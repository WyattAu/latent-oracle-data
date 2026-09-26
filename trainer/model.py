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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        k = self.Wk(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        v = self.Wv(h).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // self.heads), dim=-1)
        x = x + self.Wo((att @ v).transpose(1, 2).reshape(B, T, D))
        h = self.ln2(x)
        x = x + self.W2(F.gelu(self.W1(h)))
        return x


class ChessNet(nn.Module):
    def __init__(self, d: int = 256, layers: int = 8, heads: int = 8, dff: int = 1024, dpol: int = 128):
        super().__init__()
        self.d, self.layers, self.heads, self.dff, self.dpol = d, layers, heads, dff, dpol
        self.piece_emb = nn.Embedding(PIECE_CODES, d)
        self.square_emb = nn.Embedding(64, d)
        self.side_emb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList(Block(d, heads, dff) for _ in range(layers))
        self.lnP = nn.LayerNorm(d, eps=1e-5)
        self.Wfrom = nn.Linear(d, dpol)
        self.Wto = nn.Linear(d, dpol)
        self.promo = nn.Linear(d, 4)
        self.lnV = nn.LayerNorm(d, eps=1e-5)
        self.V1 = nn.Linear(d, 128)
        self.V2 = nn.Linear(128, 3)

    def forward(self, codes: torch.Tensor, side: torch.Tensor):
        """codes (B,64) u8, side (B,) u8 ->
           scores (B,64,64) (scaled dot), promo_logits (B,4), wdl (B,3)."""
        B = codes.shape[0]
        sq = torch.arange(64, device=codes.device)
        x = self.piece_emb(codes) + self.square_emb(sq).unsqueeze(0) + self.side_emb(side).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        hp = self.lnP(x)
        ef = self.Wfrom(hp) / math.sqrt(self.dpol)
        et = self.Wto(hp)
        scores = torch.einsum("bud,bvd->buv", ef, et)
        pooled = hp.mean(dim=1)
        promo = self.promo(pooled)
        vh = self.lnV(pooled)
        wdl = self.V2(F.gelu(self.V1(vh)))
        return scores, promo, wdl

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
        return ts

    def export_blob(self, path: str):
        hdr = struct.pack("<IIIIIII", BLOB_MAGIC, BLOB_VERSION,
                          self.d, self.layers, self.heads, self.dff, self.dpol)
        with open(path, "wb") as f:
            f.write(hdr)
            for t in self.blob_tensors():
                f.write(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
