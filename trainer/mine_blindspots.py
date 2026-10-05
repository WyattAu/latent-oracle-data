"""Blind-spot mining (RESEARCH-ROUND9 A1): score every position of archived
SPRT/self-play games with the value head; surprise = |predicted pwin for the
mover - realized game outcome|. Games containing the top-frac most surprising
positions are written to an output PGN for focused-replay retraining (whole-
game upsampling; Lc0-style game resampling). Also emits a JSONL report."""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ChessNet  # noqa: E402


def outcome_for(color: chess.Color, result: str) -> float:
    if result == "1-0":
        return 1.0 if color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else 0.0
    return 0.5


def fen_to_codes(fen: str) -> tuple[np.ndarray, int]:
    b = chess.Board(fen)
    codes = np.zeros(64, dtype=np.int64)
    base = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4,
            chess.QUEEN: 5, chess.KING: 6}
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            codes[sq] = base[p.piece_type] + (0 if p.color == chess.WHITE else 8)
    return codes, 0 if b.turn == chess.WHITE else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", nargs="+", required=True)
    ap.add_argument("--net", required=True, help=".pt checkpoint (value head source)")
    ap.add_argument("--out-pgn", default="blindspots.pgn")
    ap.add_argument("--out-report", default="blindspots.jsonl")
    ap.add_argument("--top-frac", type=float, default=0.02)
    ap.add_argument("--skip-ply", type=int, default=10, help="ignore first N plies")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessNet().to(device)
    model.load_state_dict(torch.load(args.net, map_location="cpu", weights_only=True))
    model.eval()

    rows = []  # (surprise, game_index, fen, pwin, realized)
    gi = 0
    for path in args.pgn:
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                result = g.headers.get("Result", "*")
                fens, realized = [], []
                node = g
                ply = 0
                while node.variations:
                    b = node.board()
                    if ply >= args.skip_ply:
                        fens.append(b.fen())
                        realized.append(outcome_for(b.turn, result))
                    node = node.variations[0]
                    ply += 1
                if fens:
                    rows.append((gi, fens, realized))
                gi += 1

    # batch value-head pass over all positions
    all_fens = [f for (_, fens, _) in rows for f in fens]
    all_real = [r for (_, _, real) in rows for r in realized]
    pwins = []
    with torch.no_grad():
        for i in range(0, len(all_fens), args.batch):
            chunk = all_fens[i:i + args.batch]
            codes_b, sides = [], []
            for fen in chunk:
                c, s = fen_to_codes(fen)
                codes_b.append(c)
                sides.append(s)
            codes = torch.from_numpy(np.stack(codes_b)).to(device)
            side = torch.tensor(sides, dtype=torch.long, device=device)
            _, _, wdl = model(codes, side)
            w3 = torch.softmax(wdl, -1)
            pwins += (w3[:, 0] + 0.5 * w3[:, 1]).cpu().tolist()

    surprises = []
    for f, r, p in zip(all_fens, all_real, pwins):
        surprises.append((abs(p - r), f, r, p))
    surprises.sort(reverse=True)
    n_top = max(1, int(len(surprises) * args.top_frac))
    top_set = {s[1] for s in surprises[:n_top]}
    mean_surprise = sum(s[0] for s in surprises) / max(1, len(surprises))
    print(f"positions={len(surprises)} mean surprise={mean_surprise:.4f} "
          f"top {n_top} threshold={surprises[n_top - 1][0]:.4f}")

    with open(args.out_report, "w") as rf:
        for s, fen, r, p in surprises[:n_top]:
            rf.write(json.dumps({"surprise": round(s, 4), "fen": fen,
                                 "pwin": round(p, 4), "realized": r}) + "\n")

    # whole-game replay PGN: games containing any top-surprise position
    kept = 0
    with open(args.out_pgn, "w") as out:
        gi = 0
        for path in args.pgn:
            with open(path) as f:
                while True:
                    g = chess.pgn.read_game(f)
                    if g is None:
                        break
                    hit = False
                    node = g
                    ply = 0
                    while node.variations:
                        b = node.board()
                        if ply >= args.skip_ply and b.fen() in top_set:
                            hit = True
                            break
                        node = node.variations[0]
                        ply += 1
                    if hit:
                        out.write(str(g) + "\n\n")
                        kept += 1
                    gi += 1
    print(f"replay PGN: {kept} games (of {gi}) -> {args.out_pgn}")


if __name__ == "__main__":
    main()
