"""Per-record volatility sidecar: how much the evaluation swung on the move
played at each position.

Loss attribution (RESULTS.md, 2026-10-06) showed the searchless net loses
primarily to middlegame tactics — a ~320cp median drop is a hung piece or a
missed threat. Volatility is a free proxy for "tactically critical": records
are stored in game order, so the eval delta between consecutive records is
the swing caused by the played move. No extra Stockfish work is needed.

A record's volatility is |eval[i+1] - eval[i]| (both white-POV) ONLY when the
pair is actually consecutive game positions; otherwise it is 0. Connectivity
is checked with a cheap conservative filter (sides alternate, piece count
changes by at most one, and the board differs in 2-4 squares) — game
boundaries fail it, so their meaningless deltas never leak in.

Output: `<shard>.vol.jsonl` rows {"idx": i, "vol": centipawns} for records
with vol > 0, plus a summary histogram.

Usage:
  python trainer/make_volatility.py --shard <labeled.shard> [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from format import iter_records  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out_path = args.out or (args.shard + ".vol.jsonl")

    prev = None      # (board_codes, side, eval)
    n = connected = 0
    hist = np.zeros(6, dtype=np.int64)   # buckets: 0, <=25, <=75, <=200, <=600, >600
    tmp = out_path + f".tmp-{os.getpid()}"
    with open(tmp, "w") as out:
        for idx, rec in enumerate(iter_records(args.shard)):
            cur = (np.array(rec.board_codes, dtype=np.int8),
                   int(rec.side), int(rec.eval_cp))
            vol = 0
            if prev is not None:
                pc, ps, pe = prev
                cc, cs, ce = cur
                # conservative connectivity: real consecutive plies alternate
                # sides and differ by exactly one move (quiet = 2 squares,
                # capture/ep = 3, castle = 4). Anything else is a game
                # boundary, and its delta is meaningless.
                if cs != ps:
                    d = int((cc != pc).sum())
                    dcount = abs(int((cc > 0).sum()) - int((pc > 0).sum()))
                    if 2 <= d <= 4 and dcount <= 1:
                        connected += 1
                        vol = abs(ce - pe)
            prev = cur
            n += 1
            if vol > 0:
                b = (0 if vol == 0 else
                     1 if vol <= 25 else
                     2 if vol <= 75 else
                     3 if vol <= 200 else
                     4 if vol <= 600 else 5)
                hist[b] += 1
                out.write(json.dumps({"idx": idx - 1, "vol": int(vol)}) + "\n")
            if n % 500000 == 0:
                print(f"  scanned {n}, connected {connected}", flush=True)

    os.replace(tmp, out_path)
    total = int(hist.sum())
    print(f"volatility sidecar -> {out_path}")
    print(f"- records {n}, connected pairs {connected} ({100*connected/max(1,n-1):.1f}%)")
    labels = ["0", "<=25", "<=75", "<=200", "<=600", ">600"]
    for lab, c in zip(labels, hist):
        print(f"  vol {lab:6s} {c:8d} ({100*c/max(1,total):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
