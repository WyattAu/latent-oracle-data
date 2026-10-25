#!/usr/bin/env python3
"""Phase 3: resolve factorized BC actions via mask sidecars.

For each record: candidates = legal flats with from-rank == stored rank and
to == stored to-square. Exactly one -> write target; 0 or 2+ -> n_targets=0
(trainer skips empty-target records).
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE, MaskSidecar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--mask", required=True)
    args = ap.parse_args()

    RECSZ = REC_DTYPE.itemsize
    sc = MaskSidecar(args.mask)
    n = sc.count
    f = open(args.shard, "r+b")
    kept = dropped = 0
    for i in range(n):
        flat = sc.mask_indices(i).tolist()
        f.seek(16 + i * RECSZ)
        rec = np.frombuffer(f.read(RECSZ), dtype=REC_DTYPE)[0].copy()
        t = rec["targets"]
        to_sq = int(t[1]); want_rank = int(t[2])
        cands = [x for x in flat if (x & 63) == to_sq and ((x >> 6) // 8) == want_rank]
        f.seek(16 + i * RECSZ)
        if len(cands) == 1:
            frm = cands[0] >> 6
            rec["targets"][0:3] = [frm, to_sq, 255]
            rec["n_targets"] = 1
            f.write(rec.tobytes())
            kept += 1
        else:
            rec["n_targets"] = 0
            f.write(rec.tobytes())
            dropped += 1
        if i % 5000000 == 0:
            print(f"[{args.shard}] {i}/{n} kept={kept} dropped={dropped}", flush=True)
    f.close()
    print(f"[{args.shard}] DONE kept={kept} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
