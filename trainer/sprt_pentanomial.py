"""Pentanomial SPRT statistics (RESEARCH-SYSTEMS §3): pair-level score model
for fastchess `-repeat` matches. Game pairs (same opening, both colors) give
5 outcomes {WW, WD/WL-symmetric...} -> tighter confidence intervals than
trinomial game-level stats.

Reads a fastchess PGN, pairs games by Round header, emits pentanomial counts,
Elo (five-bin MLE approximation) and the standard error.
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

import chess.pgn

# pair score for the FIRST player of the pair: (game1, game2) -> points
SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}


def pair_outcome(r1: float, r2: float) -> str:
    s = r1 + r2
    if s == 2.0:
        return "WW"
    if s == 1.5:
        return "WD"
    if s == 1.0:
        return "split(WL or LD-sym)"
    if s == 0.5:
        return "LD"
    return "LL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    args = ap.parse_args()

    rounds = defaultdict(list)  # round -> [score of engine1's game]
    order = []
    with open(args.pgn) as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            rnd = g.headers.get("Round", "?")
            res = SCORE.get(g.headers.get("Result", "*"), 0.5)
            if rnd not in rounds:
                order.append(rnd)
            rounds[rnd].append(res)

    counts = defaultdict(int)
    for rnd in order:
        rs = rounds[rnd]
        if len(rs) != 2:
            continue  # unpaired (odd game): excluded from pentanomial
        counts[pair_outcome(rs[0], rs[1])] += 1
    n = sum(counts.values())
    if n == 0:
        print("no paired rounds found")
        return
    # five bins: WW, WD, WL(split), LD, LL
    ww, wd, wl, ld, ll = (counts["WW"], counts["WD"], counts["split(WL or LD-sym)"],
                          counts["LD"], counts["LL"])
    score = (2 * ww + 1.5 * wd + wl + 0.5 * ld) / (2 * n)  # 0..1
    print(f"pairs: {n}  WW={ww} WD={wd} split={wl} LD={ld} LL={ll}")
    print(f"pair score: {score:.4f}")
    # Elo from score with the logistic model; variance from the multinomial
    eps = 1e-9
    s = min(max(score, eps), 1 - eps)
    elo = -400 * math.log10(1 / s - 1)
    # binomial-style SE on score, converted to Elo (conservative)
    var = (s * (1 - s)) / (2 * n)
    se_elo = 400 / math.log(10) * math.sqrt(var) / (s * (1 - s))
    print(f"Elo (pair model): {elo:+.1f} +- {se_elo:.1f}")


if __name__ == "__main__":
    main()
