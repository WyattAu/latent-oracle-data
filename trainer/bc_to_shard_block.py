#!/usr/bin/env python3
"""Phase 1, block-read variant: payload streamed in big sequential blocks
(no full-file mmap — page-cache charging OOM-killed the mmap variant);
index region as a small memmap. Records [start, stop) are payload-contiguous.
"""
import argparse
import mmap
import os
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE
from bag_to_shard import parse_fen
from chessbench_data import _read_str

BLOCK = 64 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0)
    args = ap.parse_args()

    with open(args.bag, "rb") as f:
        f.seek(-8, 2)
        (index_start,) = struct.unpack("<Q", f.read(8))
        f.seek(0, 2)
        num = (f.tell() - 8 - index_start) // 8 - 1
        start = args.start
        stop = min(args.stop or num, num)
        # boundary offsets for [start, stop]
        idx_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ, offset=0)
        bounds = np.frombuffer(idx_mm, dtype="<i8", count=stop - start + 1,
                               offset=index_start + start * 8)
        p_start = int(bounds[0]); p_stop = int(bounds[-1])
        out = open(args.out, "wb")
        out.write(struct.pack("<IHHQ", 0x48534F4C, 1, 16, 0))
        count = 0
        buf = []
        pos = p_start
        k = 0
        while pos < p_stop:
            nread = min(BLOCK, p_stop - pos)
            f.seek(pos)
            block = f.read(nread)
            os.posix_fadvise(f.fileno(), pos, nread, os.POSIX_FADV_DONTNEED)
            base = pos
            end_in_block = min(p_stop, pos + nread)
            # records fully ended within this block; a record straddling the
            # boundary (started in the previous block) is fetched directly
            j = k
            while j < stop - start and int(bounds[j + 1]) <= end_in_block:
                if int(bounds[j]) >= base:
                    rec_bytes = block[int(bounds[j]) - base:int(bounds[j + 1]) - base]
                else:
                    f.seek(int(bounds[j]))
                    rec_bytes = f.read(int(bounds[j + 1]) - int(bounds[j]))
                fen, p2 = _read_str(rec_bytes, 0)
                move, _ = _read_str(rec_bytes, p2)
                if len(move) == 3:
                    d0, d1, d2 = move[0], move[1], move[2]
                    if "1" <= d0 <= "8" and "a" <= d1 <= "h" and "1" <= d2 <= "8":
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
                        buf.append(rec)
                        count += 1
                        if len(buf) >= 65536:
                            out.write(b"".join(r.tobytes() for r in buf)); buf = []
                j += 1
            k = j
            pos = end_in_block
        if buf:
            out.write(b"".join(r.tobytes() for r in buf))
        out.seek(8)
        out.write(struct.pack("<Q", count))
        out.close()
        print(f"[{args.out}] wrote {count}", flush=True)


if __name__ == "__main__":
    main()
