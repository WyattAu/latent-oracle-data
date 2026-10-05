"""Syzygy ground-truth endgame labels (RESEARCH-ENDGAME-RL E1).

Walks a shard, probes <=5-piece positions against the local Syzygy files,
and writes a sidecar: record_index -> (wdl -2..2, dtz, best_move_uci).
The AV trainer consumes this to replace approximate value targets with
exact game-theoretic ones (and DTZ-optimal policy labels) in endgames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import chess.syzygy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records  # noqa: E402
from make_puzzles import codes_to_board  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--tb", default="/home/wyatt/data/syzygy")
    ap.add_argument("--out", required=True, help="sidecar .jsonl")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    tb = chess.syzygy.open_tablebase(args.tb)
    out = open(args.out, "w")
    n = probed = labeled = 0
    for idx, s in enumerate(iter_records(args.shard)):
        n += 1
        if args.max_records and n > args.max_records:
            break
        if n % 500000 == 0:
            print(f"  scanned {n}, probed {probed}, labeled {labeled}", flush=True)
        pc = sum(1 for c in s.board_codes if c)
        if pc > 5:
            continue
        probed += 1
        board = codes_to_board(s.board_codes, s.side)
        if board.is_game_over() or chess.popcount(board.occupied) > 5:
            continue
        try:
            wdl = tb.probe_wdl(board)      # -2..2, side-to-move POV
            dtz = tb.probe_dtz(board)
        except (chess.syzygy.MissingTableError, KeyError):
            continue
        if wdl is None or dtz is None:
            continue
        # best move: for winning side, any child with WDL keeping the win and
        # minimal |dtz|; for losing side, maximal |dtz|; draw side: keep draw
        best = None
        for mv in board.legal_moves:
            board.push(mv)
            try:
                w2 = -tb.probe_wdl(board)   # child POV -> mover POV
                d2 = -tb.probe_dtz(board) if tb.probe_dtz(board) is not None else None
            except (chess.syzygy.MissingTableError, KeyError):
                board.pop()
                continue
            board.pop()
            if w2 is None:
                continue
            score = (w2 * 10000 - abs(d2 or 0)) if wdl > 0 else \
                    (-w2 * 10000 - (abs(d2 or 0) if wdl < 0 else 0))
            if best is None or score > best[0]:
                best = (score, mv)
        if best is None:
            continue
        out.write(json.dumps({
            "idx": idx, "wdl": int(wdl), "dtz": int(dtz),
            "best": best[1].uci(), "npieces": pc,
        }) + "\n")
        labeled += 1
    out.close()
    print(f"done: scanned {n}, probed {probed}, labeled {labeled} -> {args.out}")


if __name__ == "__main__":
    main()
