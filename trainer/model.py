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
BLOB_VERSION_V3 = 3
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
                 gab: bool = False, v3: bool = False):
        super().__init__()
        self.d, self.layers, self.heads, self.dff, self.dpol = d, layers, heads, dff, dpol
        self.gab = gab
        self.v3 = v3
        self.piece_emb = nn.Embedding(PIECE_CODES, d)
        self.square_emb = nn.Embedding(64, d)
        self.side_emb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(Block(d, heads, dff) for _ in range(layers))
        if gab or v3:
            # Zero-init: with bias 0 the GAB model is functionally identical
            # to the v1 model, so v1 checkpoints warm-start losslessly.
            # v3 blobs carry the table unconditionally (version-3 layout).
            self.gab_table = nn.Parameter(torch.zeros(heads, GAB_BUCKETS))
            self.register_buffer("gab_buckets", self._build_buckets(), persistent=False)
        if v3:
            # SPEC-BLOB-V3.md. Every new input path is zero-initialized so a
            # v1/v2 checkpoint warm-starts bit-identically (load via
            # load_v1_into_v3, which also tiles the value head across buckets).
            self.castle_emb = nn.Embedding(16, d)      # castling-rights bitmask
            self.ep_emb = nn.Embedding(9, d)           # ep file (0 = none)
            self.king_bucket_emb = nn.Embedding(16, d) # 4x4 board quadrants
            self.rating_emb = nn.Embedding(16, d)      # rating bucket (default 0)
            nn.init.zeros_(self.castle_emb.weight)
            nn.init.zeros_(self.ep_emb.weight)
            nn.init.zeros_(self.king_bucket_emb.weight)
            nn.init.zeros_(self.rating_emb.weight)
            # HiCo additive square-history (implemented variant of N1): the
            # last 3 plies add gated embeddings on their from/to squares.
            self.hist_emb = nn.Parameter(torch.zeros(3, 2, d))  # (ply, from/to)
            self.hist_gate = nn.Parameter(torch.zeros(3))       # zero-init gate
            # Material buckets: 8 value-head tails. Warm-start tiles the v1
            # head across buckets (feature-factorizer analog).
            self.V2 = nn.Linear(128, 3 * 8)
            nn.init.zeros_(self.V2.bias)
        self.lnP = nn.LayerNorm(d, eps=1e-5)
        self.Wfrom = nn.Linear(d, dpol)
        self.Wto = nn.Linear(d, dpol)
        self.promo = nn.Linear(d, 4)
        self.lnV = nn.LayerNorm(d, eps=1e-5)
        self.V1 = nn.Linear(d, 128)
        if not v3:
            self.V2 = nn.Linear(128, 3)

    @staticmethod
    def king_bucket(sq: int) -> int:
        return (sq // 8 // 4) * 4 + (sq % 8 // 4)

    @staticmethod
    def material_bucket(codes: torch.Tensor) -> torch.Tensor:
        """8 buckets by non-king piece count (codes: B,64)."""
        pc = ((codes > 0) & (codes != 6) & (codes != 14)).sum(dim=1)
        return (pc * 8 // 30).clamp(max=7)

    @staticmethod
    def mirror_castle_bits(mask: int) -> int:
        """File mirror swaps K<->Q and k<->q rights (bits 1<->2, 4<->8)."""
        return ((mask & 1) << 1) | ((mask & 2) >> 1) | ((mask & 4) << 1) | ((mask & 8) >> 1)

    @staticmethod
    def _build_buckets() -> torch.Tensor:
        return gab_bucket_table()

    def _gab_bias(self) -> torch.Tensor | None:
        if not (self.gab or self.v3):
            return None
        # (H, T, T): per-head lookup of bucket bias
        return self.gab_table[:, self.gab_buckets]  # (H, 64, 64)

    @staticmethod
    def mirror_batch(codes: torch.Tensor, side: torch.Tensor, mask: torch.Tensor | None,
                     tgt: torch.Tensor | None):
        """File-mirror augmentation (a<->h). Lc0-style: the only legal chess
        symmetry without a color swap. GAB buckets are file-mirror invariant.
        v3 note: the CALLER must apply mirror_castle_bits to castle masks."""
        codes = codes.view(-1, 64)[:, MIRROR_IDX]
        if mask is not None:
            mask = mask[:, MIRROR_IDX, :][:, :, MIRROR_IDX]
        if tgt is not None:
            u, v = tgt // 64, tgt % 64
            tgt = MIRROR_IDX[u] * 64 + MIRROR_IDX[v]
        return codes, side, mask, tgt

    def forward(self, codes: torch.Tensor, side: torch.Tensor,
                castle: torch.Tensor | None = None, ep: torch.Tensor | None = None,
                rating: torch.Tensor | None = None, history: list | None = None):
        """codes (B,64), side (B,) -> scores (B,64,64), promo (B,4), wdl (B,3).
        v3 extras are optional with safe defaults, so all v1 call sites work
        unchanged: castle (B,) bitmask, ep (B,) file-or-0, rating (B,) bucket,
        history = list of up to 3 (from,to) tuples, oldest first."""
        device = codes.device
        B = codes.shape[0]
        sq = torch.arange(64, device=device)
        x = self.piece_emb(codes) + self.square_emb(sq).unsqueeze(0) + self.side_emb(side).unsqueeze(1)
        if self.v3:
            if castle is None:
                castle = torch.zeros(B, dtype=torch.long, device=device)
            if ep is None:
                ep = torch.zeros(B, dtype=torch.long, device=device)
            if rating is None:
                rating = torch.zeros(B, dtype=torch.long, device=device)
            x = x + self.castle_emb(castle).unsqueeze(1)
            x = x + self.ep_emb(ep).unsqueeze(1)
            ksq = (codes == torch.where(side == 0, 6, 14).unsqueeze(1)).float().argmax(dim=1)
            kb = torch.tensor([self.king_bucket(int(k)) for k in ksq], device=device)
            x = x + self.king_bucket_emb(kb).unsqueeze(1)
            x = x + self.rating_emb(rating).unsqueeze(1)
            if history:
                hist = history[-3:]  # align: oldest ply -> index 0
                off = 3 - len(hist)
                for i, (u, v) in enumerate(hist):
                    p = off + i
                    g = self.hist_gate[p]
                    x[:, u] = x[:, u] + g * self.hist_emb[p, 0]
                    x[:, v] = x[:, v] + g * self.hist_emb[p, 1]
        bias = self._gab_bias()
        for blk in self.blocks:
            x = blk(x, bias)
        mb = self.material_bucket(codes) if self.v3 else None
        return self._heads(x, mb)

    def _heads(self, x: torch.Tensor, mbucket: torch.Tensor | None = None):
        hp = self.lnP(x)
        ef = self.Wfrom(hp) / math.sqrt(self.dpol)
        et = self.Wto(hp)
        scores = torch.einsum("bud,bvd->buv", ef, et)
        pooled = hp.mean(dim=1)
        promo = self.promo(pooled)
        vh = self.lnV(pooled)
        v2out = self.V2(F.gelu(self.V1(vh)))
        if self.v3:
            # 8 material buckets: (B, 3*8) -> gather per-row bucket, then
            # softmax-safe: return raw 3-logits for the row's bucket
            B = v2out.shape[0]
            mb = mbucket if mbucket is not None else torch.zeros(B, dtype=torch.long, device=v2out.device)
            wdl = v2out.view(B, 8, 3)[torch.arange(B, device=v2out.device), mb]
        else:
            wdl = v2out
        return scores, promo, wdl

    def forward_recycle(self, codes: torch.Tensor, side: torch.Tensor, R: int = 2):
        """Recycling trunk (RESEARCH-NOVEL.md N2): R passes over the shared
        block stack, per-pass policy scores returned for the RCT loss.
        Pass r reads the residual stream left by pass r-1 (engine parity:
        src/nn/net.cpp run_trunk loop)."""
        sq = torch.arange(64, device=codes.device)
        x = self.piece_emb(codes) + self.square_emb(sq).unsqueeze(0) + self.side_emb(side).unsqueeze(1)
        bias = self._gab_bias()
        pass_scores = []
        mb = self.material_bucket(codes) if self.v3 else None
        for _ in range(max(1, R)):
            for blk in self.blocks:
                x = blk(x, bias)
            scores, promo, wdl = self._heads(x, mb)
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
        # v1-layout V2 slot: for v3 this stays v1-SHAPED (bucket 0) so every
        # later offset (gab table, v3 tail) lands where the C++ reader
        # expects. The full bucketed head is emitted in the v3 tail below.
        if self.v3:
            ts += [self.lnP.weight, self.lnP.bias,
                   self.Wfrom.weight.flatten(), self.Wfrom.bias,
                   self.Wto.weight.flatten(), self.Wto.bias,
                   self.promo.weight.flatten(), self.promo.bias,
                   self.lnV.weight, self.lnV.bias,
                   self.V1.weight.flatten(), self.V1.bias,
                   self.V2.weight.view(8, 3, 128)[0].flatten(), self.V2.bias.view(8, 3)[0]]
        else:
            ts += [self.lnP.weight, self.lnP.bias,
                   self.Wfrom.weight.flatten(), self.Wfrom.bias,
                   self.Wto.weight.flatten(), self.Wto.bias,
                   self.promo.weight.flatten(), self.promo.bias,
                   self.lnV.weight, self.lnV.bias,
                   self.V1.weight.flatten(), self.V1.bias,
                   self.V2.weight.flatten(), self.V2.bias]
        if self.gab or self.v3:
            ts += [self.gab_table.flatten()]
        if self.v3:
            ts += [self.castle_emb.weight.flatten(),
                   self.ep_emb.weight.flatten(),
                   self.king_bucket_emb.weight.flatten(),
                   self.rating_emb.weight.flatten(),
                   self.hist_emb.flatten(),
                   self.hist_gate,
                   self.V2.weight.flatten(), self.V2.bias]
        return ts

    def export_blob(self, path: str):
        # Header stays 7 ints (28 bytes) for C++ compatibility. Versions:
        # 1 = base, 2 = +GAB table, 3 = +castle/ep/king/rating/hist/buckets.
        # Each version's tensors append after the previous layout — v1 blobs
        # remain a strict byte-prefix of the tensor stream.
        version = BLOB_VERSION
        if self.gab:
            version = BLOB_VERSION_GAB
        if self.v3:
            version = BLOB_VERSION_V3
        hdr = struct.pack("<IIIIIII", BLOB_MAGIC, version,
                          self.d, self.layers, self.heads, self.dff, self.dpol)
        with open(path, "wb") as f:
            f.write(hdr)
            for t in self.blob_tensors():
                f.write(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())


def load_v1_into_v3(model_v3: ChessNet, v1_sd: dict) -> None:
    """Warm-start a v3 model from v1/v2 weights (SPEC-BLOB-V3.md §1).
    New v3 inputs are zero-init (constructor); the value head's v1 tail is
    TILED across all 8 material buckets so the loaded model is functionally
    identical to v1 regardless of bucket index."""
    sd = {k: v for k, v in v1_sd.items() if not k.startswith("V2.")}
    missing, unexpected = model_v3.load_state_dict(sd, strict=False)
    v3_only = ("castle_emb.", "ep_emb.", "king_bucket_emb.", "rating_emb.",
               "hist_emb", "hist_gate", "gab_table", "V2.")
    real_missing = [k for k in missing if not k.startswith(v3_only)]
    assert not real_missing, f"missing keys: {real_missing[:4]}"
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    with torch.no_grad():
        model_v3.V2.weight.copy_(v1_sd["V2.weight"].repeat(8, 1))  # (3,128) -> (24,128)
        model_v3.V2.bias.copy_(v1_sd["V2.bias"].repeat(8))

    @staticmethod
    def _build_buckets() -> torch.Tensor:
        return gab_bucket_table()

    def _gab_bias(self) -> torch.Tensor | None:
        if not (self.gab or self.v3):
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
