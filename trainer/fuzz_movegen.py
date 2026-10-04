"""Movegen fuzz test (RESEARCH-SYSTEMS §4 gap): random legal positions,
compare the engine's legal-move list (via nnpar) against python-chess.
Closes the last movegen risk beyond the fixed-position perft suite.

Usage: python fuzz_movegen.py --engine <latent-oracle> --weights <LONW blob> \
         [--n 500] [--seed 99]
Exit code 0 iff every position's move set matches exactly.
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys

import chess


def random_position(rng: random.Random) -> chess.Board:
    b = chess.Board()
    ply = rng.randint(1, 60)
    for _ in range(ply):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return b


def engine_moves(engine: str, weights: str, fen: str) -> set[str] | None:
    out = subprocess.run([engine, "nnpar", "--weights", weights, "--fen"] + fen.split(),
                         capture_output=True, text=True)
    moves = set()
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("WDL"):
            moves.add(parts[0])
    return moves if out.returncode == 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--weights", required=True, help="any LONW blob (v1 ok)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bad = 0
    tested = 0
    for i in range(args.n):
        b = random_position(rng)
        fen = b.fen()
        want = {m.uci() for m in b.legal_moves}
        got = engine_moves(args.engine, args.weights, fen)
        if got is None:
            print(f"[{i}] ENGINE ERROR on {fen}")
            bad += 1
            continue
        tested += 1
        if got != want:
            bad += 1
            print(f"[{i}] MISMATCH {fen}")
            print(f"  missing: {sorted(want - got)[:6]}")
            print(f"  extra:   {sorted(got - want)[:6]}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n} positions ok so far (bad={bad})", flush=True)
    print(f"fuzz complete: {tested} positions, {bad} mismatches")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
