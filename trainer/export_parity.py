"""Parity harness: export a random small net + sample positions, and emit the
reference scores the C++ side must reproduce within 1e-4.

  python3 export_parity.py --out parity/ --d 32 --layers 2 --dpol 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import square_name  # noqa: E402
from model import ChessNet  # noqa: E402


def codes_from_fen(fen: str) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    m = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
    parts = fen.split()
    ranks = parts[0].split("/")
    for r, row in enumerate(ranks):
        rank = 7 - r
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                upper = ch.upper()
                code = m[upper] + (8 if ch.islower() else 0)
                codes[rank * 8 + f] = code
                f += 1
    return codes


def reference_scores(model: ChessNet, fen: str):
    codes = torch.from_numpy(codes_from_fen(fen)).long().unsqueeze(0)
    side = torch.tensor([0 if fen.split()[1] == "w" else 1]).long()
    with torch.no_grad():
        scores, promo, wdl = model(codes, side)
    return scores[0], promo[0], torch.softmax(wdl[0], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="parity")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dff", type=int, default=64)
    ap.add_argument("--dpol", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    model = ChessNet(args.d, args.layers, args.heads, args.dff, args.dpol).eval()
    model.export_blob(os.path.join(args.out, "net.bin"))

    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ]
    expected = []
    for fen in fens:
        s, p, w = reference_scores(model, fen)
        codes = codes_from_fen(fen)
        # legal-move mask via python-chess (training-time GPL tool)
        import chess
        board = chess.Board(fen)
        moves = []
        for mv in board.legal_moves:
            u, v = mv.from_square, mv.to_square
            pr = 255
            if mv.promotion:
                pr = {"n": 0, "b": 1, "r": 2, "q": 3}[chess.piece_symbol(mv.promotion).lower()]
            moves.append((u, v, pr))
        expected.append({
            "fen": fen,
            "moves": moves,
            "scores": {f"{square_name(u)}{square_name(v)}": float(s[u, v]) for (u, v, _) in moves},
            "promo": [float(x) for x in p],
            "wdl": [float(x) for x in w],
        })

    with open(os.path.join(args.out, "expected.txt"), "w") as f:
        f.write(f"{len(fens)}\n")
        for e in expected:
            f.write(e["fen"] + "\n")
            for mv in e["moves"]:
                uci = square_name(mv[0]) + square_name(mv[1]) + (
                    {0: "n", 1: "b", 2: "r", 3: "q"}[mv[2]] if mv[2] != 255 else "")
                f.write(f"{uci} {e['scores'][square_name(mv[0]) + square_name(mv[1])]:.6f}\n")
            f.write("PROMO " + " ".join(f"{x:.6f}" for x in e["promo"]) + "\n")
            f.write("WDL " + " ".join(f"{x:.6f}" for x in e["wdl"]) + "\n")
    print(f"parity inputs: {args.out}/net.bin, {args.out}/expected.txt")


if __name__ == "__main__":
    main()
