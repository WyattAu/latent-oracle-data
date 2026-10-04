"""INT8 quantization export for ChessNet ("LOQW" v1 blob).  DEFINITIVE LAYOUT.

Scheme (mirrored exactly by src/nn/netq.cpp — single source of truth is the
blob; the parity harness parses the blob, never recalibrates):

- Quantized linears (Wq, Wk, Wv, W1, W2, Wfrom, Wto, V1):
    act quant:  a_u8 = clamp(round(in / act_s) + act_zp, 0, 255)
    weights:    symmetric s8, w = round(w_f32 / w_s)
    integer dot (centered madd): acc = sum((a_u8 - 128) * w)
    correction:  acc_eff = acc + (128 - act_zp) * rowsum
    dequant:     out = acc_eff * act_s * w_s + b_f32
  (the -128 centering keeps every s16 partial product within range, so the
  integer path is EXACT — no saturation analysis needed)
- Wo, promo, V2 stay FP32 (attention epilogue + small heads)
- Embeddings, LayerNorms, GELU, softmax stay FP32

Blob layout (all little-endian):
  u32 magic "LOQW", u32 ver=1, u32 d, u32 layers, u32 heads, u32 dff, u32 dpol
  f32 piece_emb[15d], square_emb[64d], side_emb[2d]
  per layer:
    f32 ln1.w[d], ln1.b[d]
    QL(Wq), QL(Wk), QL(Wv)          # QL = f32 act_s, u8 act_zp,
    f32 wo.w[d*d], wo.b[d]          #      s8 w[rows*cols], f32 w_s,
    f32 ln2.w[d], ln2.b[d]          #      f32 b[rows], s32 rowsum[rows]
    QL(W1) [cols=d, rows=dff], QL(W2) [cols=dff, rows=d]
  f32 lnP.w[d], lnP.b[d]
  QL(Wfrom) [cols=d, rows=dpol], QL(Wto)
  f32 promo.w[4d], promo.b[4]
  f32 lnV.w[d], lnV.b[d]
  QL(V1) [cols=d, rows=128]
  f32 v2.w[3*128], v2.b[3]

Calibration: per-tensor activation min/max over `--calib` shard samples run
through the FP32 net (forward hooks on each quantized Linear's input).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ChessNet, load_v1_into_v3  # noqa: E402
from format import iter_records  # noqa: E402

BLOB_MAGIC = 0x57514F4C  # "LOQW" LE
VER_1, VER_2_GAB, VER_3 = 1, 2, 3


def quant_s8(t: torch.Tensor):
    scale = float(t.abs().max()) / 127.0 or 1.0
    q = torch.round(t / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def quant_u8_params(lo: float, hi: float):
    scale = (hi - lo) / 255.0 or 1.0
    zp = int(round(-lo / scale))
    return scale, max(0, min(255, zp))


class Calibrator:
    def __init__(self):
        self.lo_hi = {}
        self.hooks = []

    def attach(self, model: ChessNet):
        names = []
        for li, blk in enumerate(model.blocks):
            names += [(blk.Wq, f"l{li}.Wq"), (blk.Wk, f"l{li}.Wk"),
                      (blk.Wv, f"l{li}.Wv"), (blk.W1, f"l{li}.W1"),
                      (blk.W2, f"l{li}.W2")]
        names += [(model.Wfrom, "Wfrom"), (model.Wto, "Wto"), (model.V1, "V1")]
        cal = self

        def make(name):
            def hook(mod, inp, out):
                t = inp[0].detach()
                lo, hi = cal.lo_hi.get(name, (float("inf"), float("-inf")))
                cal.lo_hi[name] = (min(lo, float(t.min())), max(hi, float(t.max())))
            return hook

        for mod, name in names:
            self.hooks.append(mod.register_forward_hook(make(name)))

    def detach(self):
        for h in self.hooks:
            h.remove()


def qlinear_bytes(linear: torch.nn.Linear, name: str, cal: Calibrator,
                  out: bytearray, cols: int, rows: int):
    lo, hi = cal.lo_hi[name]
    act_s, act_zp = quant_u8_params(lo, hi)
    wq, w_s = quant_s8(linear.weight.detach())
    # rowsum MUST be the sum of the QUANTIZED weights — the kernel's zero-point
    # correction is (128 - act_zp) * rowsum over the s8 values. (Using the
    # fp32 sums here was the v0 parity divergence.)
    rowsum = wq.to(torch.int64).sum(dim=1).clamp(-(2**31), 2**31 - 1).to(torch.int32)

    out += struct.pack("<f", act_s)
    out += struct.pack("<B", act_zp)
    out += wq.contiguous().numpy().tobytes()
    out += struct.pack("<f", w_s)
    out += linear.bias.detach().to(torch.float32).contiguous().numpy().tobytes()
    out += rowsum.contiguous().numpy().tobytes()
    assert cols * rows == linear.weight.numel(), (name, cols, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help=".pt checkpoint of the trained net")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--dpol", type=int, default=128)
    ap.add_argument("--gab", action="store_true", help="net has GAB (blob ver 2)")
    ap.add_argument("--v3", action="store_true",
                    help="net is v3 (SPEC-BLOB-V3.md): castle/ep/king/rating inputs, "
                         "material-bucketed value head, HiCo tail. Implies --gab layout "
                         "prefix; writes blob version 3.")
    ap.add_argument("--warm-v1", action="store_true",
                    help="checkpoint is v1/v2-shaped; load into the v3 model via "
                         "load_v1_into_v3 (zero-init tail, tiled value head)")
    ap.add_argument("--shard", default="", help="shard for calibration samples")
    ap.add_argument("--calib", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = ChessNet(args.d, args.layers, args.heads, args.dff, args.dpol,
                     gab=args.gab or args.v3, v3=args.v3).eval()
    sd = torch.load(args.net, map_location="cpu", weights_only=True)
    if args.v3 and args.warm_v1:
        load_v1_into_v3(model, sd)
    else:
        model.load_state_dict(sd)

    cal = Calibrator()
    cal.attach(model)
    codes_list, sides, castles, eps = [], [], [], []
    if args.shard and os.path.exists(args.shard):
        for s in iter_records(args.shard):
            codes_list.append(s.board_codes)
            sides.append(s.side)
            castles.append(s.castling & 15)
            eps.append(0 if s.ep >= 64 else 1 + (s.ep % 8))
            if len(codes_list) >= args.calib:
                break
    if not codes_list:
        g = torch.Generator().manual_seed(42)
        codes_list = [torch.randint(0, 15, (64,), generator=g).numpy().astype(np.uint8)
                      for _ in range(args.calib)]
        sides = list(np.zeros(args.calib, dtype=np.int64))
        castles = list(np.random.RandomState(42).randint(0, 16, args.calib))
        eps = list(np.random.RandomState(43).randint(0, 9, args.calib))
    with torch.no_grad():
        for i in range(0, len(codes_list), 128):
            chunk = codes_list[i:i + 128]
            n = len(chunk)
            if args.v3:
                model(torch.from_numpy(np.stack(chunk)).long(),
                      torch.tensor(sides[i:i + n], dtype=torch.long),
                      castle=torch.tensor(castles[i:i + n], dtype=torch.long),
                      ep=torch.tensor(eps[i:i + n], dtype=torch.long))
            else:
                model(torch.from_numpy(np.stack(chunk)).long(),
                      torch.tensor(sides[i:i + n], dtype=torch.long))
    cal.detach()
    print("calibrated tensors:", len(cal.lo_hi))

    d, dff, dpol = args.d, args.dff, args.dpol
    version = VER_3 if args.v3 else (VER_2_GAB if args.gab else VER_1)
    out = bytearray()
    out += struct.pack("<IIIIIII", BLOB_MAGIC, version, d, args.layers, args.heads, dff, dpol)

    def f32(t: torch.Tensor):
        out.extend(t.detach().to(torch.float32).contiguous().numpy().tobytes())

    f32(model.piece_emb.weight.flatten())
    f32(model.square_emb.weight.flatten())
    f32(model.side_emb.weight.flatten())

    for li, blk in enumerate(model.blocks):
        f32(blk.ln1.weight); f32(blk.ln1.bias)
        for nm, lin in ((f"l{li}.Wq", blk.Wq), (f"l{li}.Wk", blk.Wk), (f"l{li}.Wv", blk.Wv)):
            qlinear_bytes(lin, nm, cal, out, d, d)
        f32(blk.Wo.weight.flatten()); f32(blk.Wo.bias)
        f32(blk.ln2.weight); f32(blk.ln2.bias)
        qlinear_bytes(blk.W1, f"l{li}.W1", cal, out, d, dff)
        qlinear_bytes(blk.W2, f"l{li}.W2", cal, out, dff, d)

    f32(model.lnP.weight); f32(model.lnP.bias)
    qlinear_bytes(model.Wfrom, "Wfrom", cal, out, d, dpol)
    qlinear_bytes(model.Wto, "Wto", cal, out, d, dpol)
    f32(model.promo.weight.flatten()); f32(model.promo.bias)
    f32(model.lnV.weight); f32(model.lnV.bias)
    qlinear_bytes(model.V1, "V1", cal, out, d, 128)
    if args.v3:
        # v1-slot V2 carries bucket 0 (v1-shaped) so the C++ reader's offsets
        # stay valid; the full bucketed head goes to the v3 tail below.
        f32(model.V2.weight.view(8, 3, 128)[0].flatten()); f32(model.V2.bias.view(8, 3)[0])
    else:
        f32(model.V2.weight.flatten()); f32(model.V2.bias)

    if args.gab or args.v3:
        f32(model.gab_table.flatten())
    if args.v3:
        f32(model.castle_emb.weight.flatten())
        f32(model.ep_emb.weight.flatten())
        f32(model.king_bucket_emb.weight.flatten())
        f32(model.rating_emb.weight.flatten())
        f32(model.hist_emb.flatten())
        f32(model.hist_gate)
        f32(model.V2.weight.flatten()); f32(model.V2.bias)

    with open(args.out, "wb") as f:
        f.write(out)
    print(f"exported {args.out}: {len(out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
