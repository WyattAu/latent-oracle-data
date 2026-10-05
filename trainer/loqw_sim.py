"""LOQW blob reader + quantized forward simulation (parity reference for
src/nn/netq.cpp). Single source of truth: the blob itself.
"""
import struct

import numpy as np
import torch


class Loqw:
    def __init__(self, path):
        blob = open(path, "rb").read()
        self.magic, self.ver, self.d, self.L, self.H, self.dff, self.dpol = struct.unpack(
            "<IIIIIII", blob[:28])
        assert self.magic == 0x57514F4C
        self.buf = blob
        self.off = 28

    def f32(self, n):
        a = np.frombuffer(self.buf, dtype="<f4", count=n, offset=self.off)
        self.off += 4 * n
        return a.astype(np.float32)

    def s8(self, n):
        a = np.frombuffer(self.buf, dtype=np.int8, count=n, offset=self.off)
        self.off += n
        return a

    def s32(self, n):
        a = np.frombuffer(self.buf, dtype="<i4", count=n, offset=self.off)
        self.off += 4 * n
        return a

    def ql(self, cols, rows):
        act_s = struct.unpack_from("<f", self.buf, self.off)[0]
        self.off += 4
        zp = self.buf[self.off]
        self.off += 1
        w = self.s8(cols * rows).reshape(rows, cols)
        w_s = self.f32(1)[0]
        b = self.f32(rows)
        rs = self.s32(rows)
        return (act_s, zp, w, w_s, b, rs)

    def parse(self):
        d = self.d
        self.piece_emb = self.f32(15 * d)
        self.square_emb = self.f32(64 * d)
        self.side_emb = self.f32(2 * d)
        self.layers = []
        for _ in range(self.L):
            lay = {}
            lay["ln1w"] = self.f32(d); lay["ln1b"] = self.f32(d)
            lay["Wq"] = self.ql(d, d); lay["Wk"] = self.ql(d, d); lay["Wv"] = self.ql(d, d)
            lay["wow"] = self.f32(d * d); lay["wob"] = self.f32(d)
            lay["ln2w"] = self.f32(d); lay["ln2b"] = self.f32(d)
            lay["W1"] = self.ql(d, self.dff); lay["W2"] = self.ql(self.dff, d)
            self.layers.append(lay)
        self.lnPw = self.f32(d); self.lnPb = self.f32(d)
        self.Wfrom = self.ql(d, self.dpol); self.Wto = self.ql(d, self.dpol)
        self.promow = self.f32(4 * d); self.promob = self.f32(4)
        self.lnVw = self.f32(d); self.lnVb = self.f32(d)
        self.V1 = self.ql(d, 128); self.V2w = self.f32(3 * 128); self.V2b = self.f32(3)
        assert self.off == len(self.buf), (self.off, len(self.buf))


import math

_erf_vec = np.vectorize(math.erf, otypes=[np.float32])


def ql_np(x: torch.Tensor, ql, dbg=""):
    act_s, zp, w, w_s, b, rs = ql
    a = np.clip(np.rint(x.numpy() / act_s).astype(np.int64) + zp, 0, 255).astype(np.int32)
    acc = a @ w.T.astype(np.int32)                     # raw: sum(a_u8 * w)
    total_centered = acc - 128 * rs[None, :]                 # sum((a-128)*w)
    eff = total_centered + (128 - zp) * rs[None, :]          # sum((a-zp)*w)
    out = eff.astype(np.float32) * (act_s * w_s) + b[None, :]
    if dbg:
        print(f"  {dbg}: out[0,:3] = {out[0,:3]}")
    return torch.from_numpy(out)


def ln_np(x, w, b):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) / torch.sqrt(var + 1e-5) * torch.from_numpy(w).float() + torch.from_numpy(b).float()


def forward(m: Loqw, codes: torch.Tensor, side: int, dbg=False):
    d = m.d
    d = m.d
    x = (torch.from_numpy(m.piece_emb).float().reshape(15, d)[codes] +
         torch.from_numpy(m.square_emb).float().reshape(64, d) +
         torch.from_numpy(m.side_emb).float().reshape(2, d)[side]).squeeze(0)
    if dbg:
        pass
    sq_idx = torch.arange(64)
    for li, lay in enumerate(m.layers):
        h = ln_np(x, lay["ln1w"], lay["ln1b"])
        q = ql_np(h, lay["Wq"]); k = ql_np(h, lay["Wk"]); v = ql_np(h, lay["Wv"])
        B, T, D = 1, 64, d
        heads = m.H; hd = D // heads
        q4 = q.view(B, T, heads, hd).transpose(1, 2)
        k4 = k.view(B, T, heads, hd).transpose(1, 2)
        v4 = v.view(B, T, heads, hd).transpose(1, 2)
        att = torch.softmax(q4 @ k4.transpose(-2, -1) / math.sqrt(hd), dim=-1)
        ctx = (att @ v4).transpose(1, 2).reshape(B, T, D)
        wo_w = torch.from_numpy(lay["wow"]).float().reshape(D, D)
        wo_b = torch.from_numpy(lay["wob"]).float()
        x = x + ctx @ wo_w.transpose(-1, -2) + wo_b  # blob Wo is (out,in) row-major
        h2 = ln_np(x, lay["ln2w"], lay["ln2b"])
        w1 = ql_np(h2, lay["W1"])
        # libm-erf gelu — bit-matches the engine's gelu. torch.erf differs
        # from libm erf at ~1e-7, which flips int8 quantization boundaries
        # a few times per board (visible as 1e-3..1e-2 policy noise).
        w1n = w1.numpy()
        g = torch.from_numpy(0.5 * w1n * (1.0 + _erf_vec(w1n * 0.70710678118654752)))
        mlp = ql_np(g, lay["W2"])
        x = x + mlp
        if dbg and li == 0:
            print("after_block0[0][0..4] =", " ".join(f"{v:.6f}" for v in x[0, 0, :4].tolist()))
    x = x.reshape(64, m.d)
    hp = ln_np(x, m.lnPw, m.lnPb)
    ef = ql_np(hp, m.Wfrom) / math.sqrt(m.dpol)
    et = ql_np(hp, m.Wto)
    scores = ef @ et.transpose(0, 1) / 1.0
    pooled = hp.mean(0)
    promo = torch.from_numpy(m.promow).float().reshape(4, d) @ pooled + torch.from_numpy(m.promob).float()
    hv = ln_np(pooled.unsqueeze(0), m.lnVw, m.lnVb)
    vhid_q = ql_np(hv, m.V1).numpy()
    vhid = torch.from_numpy(0.5 * vhid_q * (1.0 + _erf_vec(vhid_q * 0.70710678118654752))).reshape(128)
    wdl = torch.from_numpy(m.V2w).float().reshape(3, 128) @ vhid + torch.from_numpy(m.V2b).float()
    if dbg:
        print("wdl logits:", wdl.tolist())
    return scores, promo, wdl


import math  # noqa: E402
