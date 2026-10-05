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



def _gate(tb, args, want: int) -> list:
    """Probe a small slice and assert the invariants a bulk run depends on:
    WDL in range, DTZ present for decisive positions, the chosen best move
    legal, and the row schema consumable by the trainer."""
    import json as _json
    rows, probed = [], 0
    for idx, s in enumerate(iter_records(args.shard)):
        pc = sum(1 for c in s.board_codes if c)
        if not (1 <= pc <= 5):
            continue
        board = codes_to_board(s.board_codes, s.side)
        if board.is_game_over():
            continue
        probed += 1
        try:
            wdl = tb.probe_wdl(board)
            dtz = tb.probe_dtz(board)
        except (chess.syzygy.MissingTableError, KeyError):
            continue
        if wdl is None:
            continue
        assert wdl in (-2, -1, 0, 1, 2), f"wdl out of range: {wdl}"
        assert chess.popcount(board.occupied) == pc, (
            f"record {idx}: piece count {pc} != board {chess.popcount(board.occupied)}")
        assert dtz is not None or wdl == 0, (
            f"record {idx}: decisive wdl {wdl} with no dtz")
        legal = {m.uci() for m in board.legal_moves}
        assert legal, f"record {idx}: no legal moves in a non-terminal position"
        rows.append({"idx": idx, "wdl": int(wdl),
                     "dtz": int(dtz) if dtz is not None else None,
                     "best": sorted(legal)[0], "npieces": pc})
        _json.dumps(rows[-1])  # schema must serialize
        if len(rows) >= want:
            break
    assert rows, (
        f"TB gate found no probeable positions in the first pass over "
        f"{args.shard} (syzygy dir={args.tb}); is the tablebase mount present?")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--tb", default="/home/wyatt/data/syzygy")
    ap.add_argument("--out", required=True, help="sidecar .jsonl")
    ap.add_argument("--max-records", type=int, default=0, help="0 = all")
    ap.add_argument("--gate-only", action="store_true",
                    help="run the slice validation and exit")
    ap.add_argument("--gate-size", type=int, default=200,
                    help="positions to validate before the bulk scan")
    args = ap.parse_args()

    tb = chess.syzygy.open_tablebase(args.tb)

    # Standing policy (2026-10-05): validate a small slice BEFORE committing to
    # a full-shard probe. Every bulk builder in this repo does this; the TB
    # sidecar was the last one missing it.
    if args.gate_only or args.gate_size:
        rows = _gate(tb, args, int(args.gate_size or 200))
        print(f"TB gate PASSED ({len(rows)} probed positions)")
        if args.gate_only:
            return

    tmp_out = args.out + f".tmp-{os.getpid()}"
    out = open(tmp_out, "w")
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
    # atomic: an interrupted bulk run must never leave a half-written sidecar
    os.replace(tmp_out, args.out)
    print(f"done: scanned {n}, probed {probed}, labeled {labeled} -> {args.out}")


if __name__ == "__main__":
    main()
