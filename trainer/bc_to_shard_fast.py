#!/usr/bin/env python3
"""Fast BC-bag converter — phase 1 (no python-chess in the hot path).

DeepMind's BC action space is factorized: "<from-rank digit><to-file><to-rank>"
(e.g. "5e4"), verified empirically 344/344. Records carry targets =
[255, to_square, from_rank]; resolve_targets.py (phase 3) fixes them from
mask sidecars. Value targets neutral (bag has no results).
"""
import argparse
import mmap
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE
from bag_to_shard import parse_fen
from chessbench_data import _read_str


def iter_bagz_slice(path, start, stop):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        (index_start,) = struct.unpack_from("<Q", mm, len(mm) - 8)
        num = (len(mm) - index_start) // 8 - 1
        stop = min(stop, num)
        prev = 0
        if start > 0:
            (prev,) = struct.unpack_from("<q", mm, index_start + (start - 1) * 8)
        for i in range(start, stop):
            (stop_i,) = struct.unpack_from("<q", mm, index_start + i * 8)
            yield mm[prev:stop_i]
            prev = stop_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0, help="exclusive; 0 = to end")
    args = ap.parse_args()

    out = open(args.out, "wb")
    out.write(struct.pack("<IHHQ", 0x48534F4C, 1, 16, 0))
    count = 0
    buf = []
    append = buf.append
    for raw in iter_bagz_slice(args.bag, args.start, args.stop or (1 << 62)):
        fen, pos = _read_str(raw, 0)
        move, _ = _read_str(raw, pos)
        if len(move) != 3:
            continue
        d0, d1, d2 = move[0], move[1], move[2]
        if d0 < "1" or d0 > "8" or d1 < "a" or d1 > "h" or d2 < "1" or d2 > "8":
            continue
        want_rank = ord(d0) - 49
        to_sq = (ord(d1) - 97) + 8 * (ord(d2) - 49)
        codes, packed, side, rights, ep, hm, fm = parse_fen(fen)
        rec = np.zeros(1, dtype=REC_DTYPE)
        rec[0]["side"] = side
        rec[0]["castling"] = rights
        rec[0]["ep"] = ep
        rec[0]["halfmove"] = hm
        rec[0]["fullmove"] = fm
        rec[0]["n_targets"] = 1
        rec[0]["board"] = packed
        rec[0]["eval_cp"] = 0
        rec[0]["targets"][0:3] = [255, to_sq, want_rank]
        rec[0]["wdl"] = (1/3, 1/3, 1/3)
        append(rec)
        count += 1
        if len(buf) >= 65536:
            out.write(b"".join(r.tobytes() for r in buf)); buf = []
    if buf:
        out.write(b"".join(r.tobytes() for r in buf))
    out.seek(8)
    out.write(struct.pack("<Q", count))
    out.close()
    print(f"[{args.out}] wrote {count}", flush=True)


if __name__ == "__main__":
    main()
