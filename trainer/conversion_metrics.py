"""Conversion + repetition metrics over fastchess PGNs (RESEARCH-ENDGAME-RL
E4). Material-based conversion proxy: among games where one side reached
>=3 material points advantage (excluding promotions noise: pawns=1, N/B=3,
R=5, Q=9, min 5 pieces each side), fraction converted to a win. Also:
average repetitions per game, time-forfeit counts."""
from __future__ import annotations

import argparse
import collections

import chess
import chess.pgn

VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(b: chess.Board) -> tuple[int, int]:
    w = r = 0
    for pt, v in VALS.items():
        w += v * len(b.pieces(pt, chess.WHITE))
        r += v * len(b.pieces(pt, chess.BLACK))
    return w, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgns", nargs="+")
    ap.add_argument("--adv", type=int, default=3)
    args = ap.parse_args()

    conv_won = conv_total = 0
    rep_total = games = 0
    forfeits = collections.Counter()
    for path in args.pgns:
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                games += 1
                node = g
                hit_side = None
                board_hist = []
                while node.variations:
                    b = node.board()
                    board_hist.append(b)
                    if hit_side is None and b.fullmove_number >= 10:
                        w, r = material(b)
                        if w - r >= args.adv and w >= 5:
                            hit_side = chess.WHITE
                        elif r - w >= args.adv and r >= 5:
                            hit_side = chess.BLACK
                    node = node.variations[0]
                if hit_side is not None:
                    conv_total += 1
                    res = g.headers.get("Result", "*")
                    won = (hit_side == chess.WHITE and res == "1-0") or \
                          (hit_side == chess.BLACK and res == "0-1")
                    conv_won += int(won)
                # repetitions: count positions seen 3+ times via key set
                keys = collections.Counter(b._transposition_key() if hasattr(b, "_transposition_key") else b.fen() for b in board_hist)
                rep_total += sum(1 for v in keys.values() if v >= 3)
                term = g.headers.get("Termination", "")
                if "time forfeit" in term.lower():
                    forfeits[g.headers.get("Result", "?")] += 1
    print(f"games={games}")
    print(f"conversion (>= {args.adv} material at move 10+, min 5 pieces): "
          f"{conv_won}/{conv_total} = {conv_won / max(1, conv_total):.3f}")
    print(f"games with a 3-fold repetition: {rep_total} ({rep_total / max(1, games):.3f}/game)")
    print(f"time forfeits by result: {dict(forfeits)}")


if __name__ == "__main__":
    main()
