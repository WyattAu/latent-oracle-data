"""Random-opening SPRT gate (RESEARCH-ROUND9 A1): a robustness leg against
off-distribution starts. Generates N random legal openings (6-12 plies,
balanced material via mirrored random plies), writes an EPD book, and the
caller runs fastchess with it. A net that gains on human openings but
collapses here has learned human priors, not chess (Wang et al. 2022
Go-attack lesson)."""
from __future__ import annotations

import argparse
import random

import chess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="random_openings.epd")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen, out_lines = set(), []
    attempts = 0
    while len(out_lines) < args.n and attempts < args.n * 50:
        attempts += 1
        b = chess.Board()
        ply = rng.randint(args.min_ply, args.max_ply)
        ok = True
        for _ in range(ply):
            legal = list(b.legal_moves)
            if not legal:
                ok = False
                break
            b.push(rng.choice(legal))
        if not ok or b.is_game_over():
            continue
        # material sanity: skip mass-capture starts
        wp = sum(len(b.pieces(pt, chess.WHITE)) for pt in
                 (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        bp = sum(len(b.pieces(pt, chess.BLACK)) for pt in
                 (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
        if abs(wp - bp) > 1:
            continue
        key = b.board_fen()
        if key in seen:
            continue
        seen.add(key)
        ep = b.ep_square if b.ep_square is not None else "-"
        ep = chess.square_name(ep) if ep != "-" else "-"
        castle = "".join(c for c, okc in zip("KQkq", [
            b.has_kingside_castling_rights(chess.WHITE),
            b.has_queenside_castling_rights(chess.WHITE),
            b.has_kingside_castling_rights(chess.BLACK),
            b.has_queenside_castling_rights(chess.BLACK)]) if okc) or "-"
        out_lines.append(f"{b.board_fen()} {('w' if b.turn else 'b')} {castle} {ep} 0 1")
    with open(args.out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {len(out_lines)} random openings -> {args.out}")


if __name__ == "__main__":
    main()
