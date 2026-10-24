"""LONW (FP32) blob reader + numpy forward — mirrors loqw_sim.py's LOQW
structure with plain f32 arrays, for the FP32-path parity probe.
"""
import math
import struct

import numpy as np
import torch

_erf_vec = np.vectorize(math.erf, otypes=[np.float32])


class Lonw:
    def __init__(self, path):
        blob = open(path, "rb").read()
        self.magic, self.ver, self.d, self.L, self.H, self.dff, self.dpol = struct.unpack(
            "<IIIIIII", blob[:28])
        assert self.magic == 0x574E4F4C, hex(self.magic)
        self.buf = blob
        self.off = 28

    def f32(self, n):
        a = np.frombuffer(self.buf, dtype="<f4", count=n, offset=self.off)
        self.off += 4 * n
        return a.astype(np.float32)

    def parse(self):
        # Layout mirrors src/nn/net.cpp exactly: every Linear stores its
        # bias immediately after its weight block (q/k/v, Wo, W1, W2,
        # Wfrom, Wto, V1 all have biases).
        d = self.d
        self.piece_emb = self.f32(15 * d)
        self.square_emb = self.f32(64 * d)
        self.side_emb = self.f32(2 * d)
        self.layers = []
        for _ in range(self.L):
            lay = {}
            lay["ln1w"] = self.f32(d); lay["ln1b"] = self.f32(d)
            lay["Wq"] = self.f32(d * d); lay["bq"] = self.f32(d)
            lay["Wk"] = self.f32(d * d); lay["bk"] = self.f32(d)
            lay["Wv"] = self.f32(d * d); lay["bv"] = self.f32(d)
            lay["wow"] = self.f32(d * d); lay["wob"] = self.f32(d)
            lay["ln2w"] = self.f32(d); lay["ln2b"] = self.f32(d)
            lay["W1"] = self.f32(self.dff * d); lay["b1"] = self.f32(self.dff)
            lay["W2"] = self.f32(d * self.dff); lay["b2"] = self.f32(d)
            self.layers.append(lay)
        self.lnPw = self.f32(d); self.lnPb = self.f32(d)
        self.Wfrom = self.f32(self.dpol * d); self.bfrom = self.f32(self.dpol)
        self.Wto = self.f32(self.dpol * d); self.bto = self.f32(self.dpol)
        self.promow = self.f32(4 * d); self.promob = self.f32(4)
        self.lnVw = self.f32(d); self.lnVb = self.f32(d)
        self.V1 = self.f32(128 * d); self.bV1 = self.f32(128)
        self.V2w = self.f32(3 * 128); self.V2b = self.f32(3)
        if self.off != len(self.buf):
            # v2-era blobs carry extra tail arrays (king_bucket-like, ~67KB);
            # move selection only needs the prefix, so warn and continue.
            print(f"[lonw_sim] note: {len(self.buf) - self.off} unread tail bytes")


def lin(x, w, b):
    # w is (out, in) row-major; b may be None (weights carry their own bias
    # already fused or are bias-free q/k/v projections)
    out = x @ torch.from_numpy(np.ascontiguousarray(w)).float().T
    if b is not None:
        out = out + torch.from_numpy(b).float()
    return out


def ln(x, w, b):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5) * torch.from_numpy(w).float() \
        + torch.from_numpy(b).float()


def forward(m: Lonw, codes: torch.Tensor, side: int):
    d = m.d
    x = (torch.from_numpy(m.piece_emb).float().reshape(15, d)[codes] +
         torch.from_numpy(m.square_emb).float().reshape(64, d) +
         torch.from_numpy(m.side_emb).float().reshape(2, d)[side]).squeeze(0)
    B, T, D = 1, 64, d
    heads = m.H
    hd = D // heads
    for lay in m.layers:
        h = ln(x, lay["ln1w"], lay["ln1b"])
        q = lin(h, lay["Wq"].reshape(D, D), lay["bq"]).view(B, T, heads, hd).transpose(1, 2)
        k = lin(h, lay["Wk"].reshape(D, D), lay["bk"]).view(B, T, heads, hd).transpose(1, 2)
        v = lin(h, lay["Wv"].reshape(D, D), lay["bv"]).view(B, T, heads, hd).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + lin(ctx, lay["wow"].reshape(D, D), lay["wob"])
        h2 = ln(x, lay["ln2w"], lay["ln2b"])
        w1 = lin(h2, lay["W1"].reshape(m.dff, D), lay["b1"])
        g = torch.from_numpy(0.5 * w1.numpy() * (1.0 + _erf_vec(w1.numpy() * 0.70710678118654752)))
        x = x + lin(g, lay["W2"].reshape(D, m.dff), lay["b2"])
    x = x.reshape(64, d)
    hp = ln(x, m.lnPw, m.lnPb)
    ef = lin(hp, m.Wfrom.reshape(m.dpol, D), m.bfrom)
    et = lin(hp, m.Wto.reshape(m.dpol, D), m.bto)
    scores = (ef @ et.transpose(0, 1)) / math.sqrt(m.dpol)
    pooled = hp.mean(0)
    promo = torch.from_numpy(m.promow).float().reshape(4, d) @ pooled + torch.from_numpy(m.promob).float()
    hv = ln(pooled.unsqueeze(0), m.lnVw, m.lnVb)
    vhid = (hv @ torch.from_numpy(m.V1.reshape(128, d)).float().T + torch.from_numpy(m.bV1).float()).numpy()
    vhid = 0.5 * vhid * (1.0 + _erf_vec(vhid * 0.70710678118654752))
    wdl = torch.from_numpy(m.V2w).float().reshape(3, 128) @ torch.from_numpy(
        vhid.astype(np.float32)).squeeze() + torch.from_numpy(m.V2b).float()
    return scores, promo, wdl
