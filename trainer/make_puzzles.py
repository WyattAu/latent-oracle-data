"""Build the 500-position puzzle suite from a labeled shard (RESEARCH-EVAL.md
§3.2): 200 quiet (|eval|<50cp), 200 decisive (|eval|>200cp), 100 endgames
(<=12 pieces). Each puzzle: FEN + SF-best move. Fixed slice for HPO triage
and cross-generation strength ranking."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records, move_to_uci  # noqa: E402

PIECE_TO_PC = {1: ("P", 0), 2: ("N", 0), 3: ("B", 0), 4: ("R", 0), 5: ("Q", 0), 6: ("K", 0),
               9: ("P", 1), 10: ("N", 1), 11: ("B", 1), 12: ("R", 1), 13: ("Q", 1), 14: ("K", 1)}


# castling bitmask convention (shard + engine types.hpp CR_*): WK=1, WQ=2, BK=4, BQ=8
CASTLE_FEN = [("K" if m & 1 else "") + ("Q" if m & 2 else "")
              + ("k" if m & 4 else "") + ("q" if m & 8 else "") for m in range(16)]
NO_EP = 255


def codes_to_board(codes: np.ndarray, side: int, castling: int = 0, ep: int = NO_EP):
    """Reconstruct a full chess position. castling (4-bit mask) and ep
    (square index, 255=none) MUST be passed for move generation to match the
    engine: without them castling/en-passant moves are absent from
    legal_moves, which silently breaks sample runs (found 2026-10-05)."""
    import chess
    board = chess.Board(None)
    for sq, code in enumerate(codes):
        if code in PIECE_TO_PC:
            letter, color = PIECE_TO_PC[code]
            board.set_piece_at(sq, chess.Piece.from_symbol(letter if color == 0 else letter.lower()))
    board.turn = chess.WHITE if side == 0 else chess.BLACK
    fen_castle = CASTLE_FEN[castling & 15]
    if fen_castle:
        board.set_castling_fen(fen_castle)
    if ep != NO_EP:
        board.ep_square = ep
    return board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="labeled shard (needs SF targets)")
    ap.add_argument("--out", default="puzzles_500.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-scan", type=int, default=2_000_000,
                    help="records to scan (shards are already game-randomized)")
    args = ap.parse_args()

    random.seed(args.seed)
    quiet, decisive, endgame = [], [], []
    scanned = 0
    for s in iter_records(args.shard):
        scanned += 1
        if scanned > args.max_scan:
            break
        if len(s.targets) < 1 or len(s.targets) < 3:
            continue  # need full multipv trio for clean labels
        npieces = int(np.count_nonzero(s.board_codes))
        if npieces > 32:
            continue
        board = codes_to_board(s.board_codes, s.side)
        if board.is_game_over():
            continue
        u, v, promo = s.targets[0]
        mv = None
        for m in board.legal_moves:
            if m.from_square == u and m.to_square == v:
                mv = m
                break
        if mv is None:
            continue
        puz = {"fen": board.fen(), "best": move_to_uci(s.targets[0]),
               "eval_cp": int(s.eval_cp), "npieces": npieces}
        if npieces <= 12 and len(endgame) < 100:
            endgame.append(puz)
        elif abs(s.eval_cp) < 50 and len(quiet) < 200:
            quiet.append(puz)
        elif abs(s.eval_cp) > 200 and len(decisive) < 200:
            decisive.append(puz)
        if len(quiet) == 200 and len(decisive) == 200 and len(endgame) == 100:
            break

    puzzles = quiet + decisive + endgame
    with open(args.out, "w") as f:
        for p in puzzles:
            f.write(json.dumps(p) + "\n")
    print(f"scanned {scanned} records -> {len(puzzles)} puzzles "
          f"(quiet {len(quiet)}, decisive {len(decisive)}, endgame {len(endgame)}) -> {args.out}")


if __name__ == "__main__":
    main()
