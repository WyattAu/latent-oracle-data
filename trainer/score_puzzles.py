"""Score a net on the puzzle suite (make_puzzles.py output). Reports
move-matching rate overall and per bucket - the HPO stage-A ranking metric."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_puzzles import codes_to_board  # noqa: E402
from model import ChessNet  # noqa: E402


def fen_to_codes(fen: str) -> tuple[np.ndarray, int]:
    import chess
    board = chess.Board(fen)
    codes = np.zeros(64, dtype=np.int64)
    base = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4,
            chess.QUEEN: 5, chess.KING: 6}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            codes[sq] = base[p.piece_type] + (0 if p.color == chess.WHITE else 8)
    return codes, 0 if board.turn == chess.WHITE else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help=".pt checkpoint")
    ap.add_argument("--gab", action="store_true")
    ap.add_argument("--puzzles", default="puzzles_500.jsonl")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet(gab=args.gab).to(device)
    sd = torch.load(args.net, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.eval()

    puzzles = [json.loads(l) for l in open(args.puzzles)]
    correct = {}
    total = {}

    with torch.no_grad():
        for i in range(0, len(puzzles), args.batch):
            chunk = puzzles[i:i + args.batch]
            codes_b, side_b, boards = [], [], []
            for p in chunk:
                c, s = fen_to_codes(p["fen"])
                codes_b.append(c)
                side_b.append(s)
                boards.append(codes_to_board(c.astype(np.uint8), s))
            codes = torch.from_numpy(np.stack(codes_b)).to(device)
            side = torch.tensor(side_b, dtype=torch.long, device=device)
            scores, _, _ = model(codes, side)
            flat = scores.reshape(codes.shape[0], 64 * 64)
            for j, (p, board) in enumerate(zip(chunk, boards)):
                best_uci, best_sc = None, -1e30
                for m in board.legal_moves:
                    sc = flat[j, m.from_square * 64 + m.to_square].item()
                    if m.promotion:
                        pass  # label comparison below ignores promo choice
                    if sc > best_sc:
                        best_sc, best_uci = sc, m.uci()
                b = ("endgame" if p["npieces"] <= 12
                     else ("quiet" if abs(p["eval_cp"]) < 50 else "decisive"))
                total[b] = total.get(b, 0) + 1
                # promo suffix-insensitive compare
                hit = (best_uci == p["best"]
                       or best_uci.rstrip("nbrq") == p["best"].rstrip("nbrq"))
                correct[b] = correct.get(b, 0) + int(hit)
    n_all, c_all = sum(total.values()), sum(correct.values())
    print(f"{args.net}: overall {c_all}/{n_all} = {c_all / max(1, n_all):.3f}")
    for b in ("quiet", "decisive", "endgame"):
        if total.get(b):
            print(f"  {b:9s}: {correct[b]}/{total[b]} = {correct[b] / total[b]:.3f}")


if __name__ == "__main__":
    main()
