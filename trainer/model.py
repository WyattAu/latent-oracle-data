"""ChessNet: the latent-oracle control policy/value network.

Architecture contract (mirrored bit-for-bit by the engine's C++ reference
inference in latent-oracle/src/nn/ — do not change one side alone):

  tokens:  t[i] = piece_emb[code_i] + square_emb[i] + side_emb[side]
  blocks:  L x pre-LN transformer (heads, GELU erf, no causal mask)
  policy:  LN -> E_from / E_to projections (d -> dpol)
           score(u, v) = <E_from[u], E_to[v]> / sqrt(dpol)
           promotion moves add promo_logit[p] from a pooled linear head
  value:   LN -> mean-pool -> linear(d,128) -> GELU -> linear(128,3) WDL

Weight blob layout (f32 LE, row-major, magic "LONW", version 1) is written by
export_blob() and consumed by the engine.
"""
from __future__ import annotations

import math
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

PIECE_CODES = 15  # 0 empty, 1-6 white, 9-14 black (table has spare room)
BLOB_MAGIC = 0x574E4F4C  # "LONW" LE
BLOB_VERSION = 1
BLOB_VERSION_GAB = 2
GAB_BUCKETS = 8  # 0 same, 1 knight, 2 file, 3 rank, 4 diag, 5 cheb1, 6 cheb2, 7 other


def gab_bucket_table() -> torch.Tensor:
    """(64, 64) int64 bucket ids from square geometry (Chessformer GAB-lite)."""
    t = torch.full((64, 64), 7, dtype=torch.long)
    for i in range(64):
        fi, ri = i % 8, i // 8
        for j in range(64):
            fj, rj = j % 8, j // 8
            if i == j:
                t[i, j] = 0
            elif {abs(fi - fj), abs(ri - rj)} == {1, 2}:
                t[i, j] = 1  # knight offset
            elif fi == fj:
                t[i, j] = 2
            elif ri == rj:
                t[i, j] = 3
            elif abs(fi - fj) == abs(ri - rj):
                t[i, j] = 4  # same diagonal
            elif max(abs(fi - fj), abs(ri - rj)) == 1:
                t[i, j] = 5  # king-adjacent (residual after diag check)
            elif max(abs(fi - fj), abs(ri - rj)) == 2:
                t[i, j] = 6
    return t


_GAB_BUCKETS_CONST = None  # lazily built, registered as non-persistent buffer

# File-mirror lookup: square a1(=0) -> h1(=7); sq ^ 7 flips the file bits.
MIRROR_IDX = torch.tensor([sq ^ 7 for sq in range(64)], dtype=torch.long)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dff: int):
        super().__init__()
        self.d, self.heads = d, heads
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wo = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.W1 = nn.Linear(d, dff)
        self.W2 = nn.Linear(dff, d)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(D // self.heads)
        if attn_bias is not None:
            logits = logits + attn_bias.unsqueeze(0)  # (1,H,T,T)
        att = torch.softmax(logits, dim=-1)
        x = x + self.Wo((att @ v).transpose(1, 2).reshape(B, T, D))
        h = self.ln2(x)
        x = x + self.W2(F.gelu(self.W1(h)))
        return x


class ChessNet(nn.Module):
    def __init__(self, d: int = 256, layers: int = 8, heads: int = 8, dff: int = 1024, dpol: int = 128,
                 gab: bool = False):
        super().__init__()
        self.d, self.layers, self.heads, self.dff, self.dpol = d, layers, heads, dff, dpol
        self.gab = gab
        self.piece_emb = nn.Embedding(PIECE_CODES, d)
        self.square_emb = nn.Embedding(64, d)
        self.side_emb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(Block(d, heads, dff) for _ in range(layers))
        if gab:
            # Zero-init: with bias 0 the GAB model is functionally identical
            # to the v1 model, so v1 checkpoints warm-start losslessly.
            self.gab_table = nn.Parameter(torch.zeros(heads, GAB_BUCKETS))
            self.register_buffer("gab_buckets", self._build_buckets(), persistent=False)
        self.lnP = nn.LayerNorm(d, eps=1e-5)
        self.Wfrom = nn.Linear(d, dpol)
        self.Wto = nn.Linear(d, dpol)
        self.promo = nn.Linear(d, 4)
        self.lnV = nn.LayerNorm(d, eps=1e-5)
        self.V1 = nn.Linear(d, 128)
        self.V2 = nn.Linear(128, 3)

    @staticmethod
    def _build_buckets() -> torch.Tensor:
        return gab_bucket_table()

    def _gab_bias(self) -> torch.Tensor | None:
        if not self.gab:
            return None
        # (H, T, T): per-head lookup of bucket bias
        return self.gab_table[:, self.gab_buckets]  # (H, 64, 64)

    # ------------------------------------------------------------ utilities
    @staticmethod
    def mirror_batch(codes: torch.Tensor, side: torch.Tensor, mask: torch.Tensor | None,
                     tgt: torch.Tensor | None):
        """File-mirror augmentation (a<->h). Lc0-style: the only legal chess
        symmetry without a color swap. Mirrors board, move indices, and mask.
        GAB buckets are invariant under file mirror, so no bias change."""
        codes = codes.view(-1, 64)[:, MIRROR_IDX]
        if mask is not None:
            mask = mask[:, MIRROR_IDX, :][:, :, MIRROR_IDX]
        if tgt is not None:
            u, v = tgt // 64, tgt % 64
            tgt = MIRROR_IDX[u] * 64 + MIRROR_IDX[v]
        return codes, side, mask, tgt

    def forward(self, codes: torch.Tensor, side: torch.Tensor):
        """codes (B,64) u8, side (B,) u8 ->
           scores (B,64,64) (scaled dot), promo_logits (B,4), wdl (B,3)."""
        B = codes.shape[0]
        sq = torch.arange(64, device=codes.device)
        x = self.piece_emb(codes) + self.square_emb(sq).unsqueeze(0) + self.side_emb(side).unsqueeze(1)
        bias = self._gab_bias()
        for blk in self.blocks:
            x = blk(x, bias)
        return self._heads(x)

    def _heads(self, x: torch.Tensor):
        hp = self.lnP(x)
        ef = self.Wfrom(hp) / math.sqrt(self.dpol)
        et = self.Wto(hp)
        scores = torch.einsum("bud,bvd->buv", ef, et)
        pooled = hp.mean(dim=1)
        promo = self.promo(pooled)
        vh = self.lnV(pooled)
        wdl = self.V2(F.gelu(self.V1(vh)))
        return scores, promo, wdl

    def forward_recycle(self, codes: torch.Tensor, side: torch.Tensor, R: int = 2):
        """Recycling trunk (RESEARCH-NOVEL.md N2): R passes over the shared
        block stack, per-pass policy scores returned for the RCT loss.
        Pass r reads the residual stream left by pass r-1 (engine parity:
        src/nn/net.cpp run_trunk loop)."""
        B = codes.shape[0]
        sq = torch.arange(64, device=codes.device)
        x = self.piece_emb(codes) + self.square_emb(sq).unsqueeze(0) + self.side_emb(side).unsqueeze(1)
        bias = self._gab_bias()
        pass_scores = []
        for _ in range(max(1, R)):
            for blk in self.blocks:
                x = blk(x, bias)
            scores, promo, wdl = self._heads(x)
            pass_scores.append((scores, promo, wdl))
        return pass_scores

    # ---------------------------------------------------------------- export
    def blob_tensors(self) -> list[torch.Tensor]:
        ts = [self.piece_emb.weight.flatten(), self.square_emb.weight.flatten(),
              self.side_emb.weight.flatten()]
        for blk in self.blocks:
            ts += [blk.ln1.weight, blk.ln1.bias,
                   blk.Wq.weight.flatten(), blk.Wq.bias,
                   blk.Wk.weight.flatten(), blk.Wk.bias,
                   blk.Wv.weight.flatten(), blk.Wv.bias,
                   blk.Wo.weight.flatten(), blk.Wo.bias,
                   blk.ln2.weight, blk.ln2.bias,
                   blk.W1.weight.flatten(), blk.W1.bias,
                   blk.W2.weight.flatten(), blk.W2.bias]
        ts += [self.lnP.weight, self.lnP.bias,
               self.Wfrom.weight.flatten(), self.Wfrom.bias,
               self.Wto.weight.flatten(), self.Wto.bias,
               self.promo.weight.flatten(), self.promo.bias,
               self.lnV.weight, self.lnV.bias,
               self.V1.weight.flatten(), self.V1.bias,
               self.V2.weight.flatten(), self.V2.bias]
        if self.gab:
            ts += [self.gab_table.flatten()]
        return ts

    def export_blob(self, path: str):
        # Header stays 7 ints (28 bytes) for C++ compatibility. GAB nets use
        # BLOB_VERSION_GAB and append the gab_table (heads*8 f32) after the
        # standard tensor stream — v1 blobs remain a strict prefix.
        version = BLOB_VERSION_GAB if self.gab else BLOB_VERSION
        hdr = struct.pack("<IIIIIII", BLOB_MAGIC, version,
                          self.d, self.layers, self.heads, self.dff, self.dpol)
        with open(path, "wb") as f:
            f.write(hdr)
            for t in self.blob_tensors():
                f.write(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
