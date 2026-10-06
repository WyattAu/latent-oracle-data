"""Loss attribution for a match PGN: where did the net actually go wrong?

For every game the net lost, find the last position whose evaluation still
favored it (the "pivot"), then classify the drop:

  tactics      a large eval swing inside the opening/middlegame
  endgame      the pivot is already a low-piece ending (<= 8 pieces)
  material     the net is materially down at the pivot
  clock        the net's clock was low when the drop happened
  promotion    a passed/promotion race went wrong

The point is to rank mechanisms by how much Elo they could plausibly be
worth, instead of guessing which lever to pull next.

Usage:
  python trainer/analyze_losses.py --pgn <file.pgn> --engine net \
      [--engine-label "BC-v1 e2"] [--depth 12] [--threads 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).parent))

# Stockfish reports mate as "score mate N"; analyze_losses maps that to a
# sentinel. Positions near it are terminal, so they are excluded from the
# attribution: a mate delivery is the end of a game, not its cause.
MATE_CP = 30000


def _side_to_move_is_net(board: chess.Board, engine_color: chess.Color) -> bool:
    return board.turn == engine_color


class Evaluator:
    """Thin Stockfish UCI wrapper; one engine reused for the whole analysis."""

    def __init__(self, path: str, depth: int, threads: int, hash_mb: int = 32):
        import subprocess
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.depth = depth
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("uci")
        self._wait("uciok")
        self._send("isready")
        self._wait("readyok")

    def _send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _wait(self, token: str) -> None:
        assert self.p.stdout is not None
        while True:
            line = self.p.stdout.readline()
            if not line or token in line:
                return

    def cp(self, board: chess.Board) -> int | None:
        """Centipawns for the side to move (Stockfish reports it that way).

        Must read until `bestmove`: returning on the first `info` line leaves
        the engine still searching, so the next `position` desyncs and every
        later score is garbage. Also takes the LAST score of the search, not
        the first (depth-1 scores are meaningless).
        """
        self._send(f"position fen {board.fen()}")
        self._send(f"go depth {self.depth}")
        assert self.p.stdout is not None
        best: int | None = None
        while True:
            line = self.p.stdout.readline()
            if not line:
                return None
            if line.startswith("bestmove"):
                return best
            if not (line.startswith("info") and " score " in line):
                continue
            toks = line.split()
            try:
                i = toks.index("score")
                kind, val = toks[i + 1], int(toks[i + 2])
            except (ValueError, IndexError):
                continue
            if kind == "cp":
                best = val
            elif kind == "mate":
                best = 30000 if val > 0 else -30000

    def close(self) -> None:
        try:
            self.p.kill()
        except Exception:  # noqa: BLE001
            pass


def classify(board: chess.Board, drop: int, ply: int) -> str:
    pieces = chess.popcount(board.occupied)
    if pieces <= 8:
        return "endgame"
    # material balance in centipawns, net's perspective
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900}
    bal = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.piece_type == chess.KING:
            continue
        bal += vals[p.piece_type] * (1 if p.color == board.turn else -1)
    if abs(bal) > 400:
        return "material"
    if drop >= 300:
        return "tactics"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--engine", default="net", help="name of the net in the PGN")
    ap.add_argument("--engine-label", default="")
    ap.add_argument("--sf", default="/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()

    ev = Evaluator(args.sf, args.depth, args.threads)
    counts: Counter = Counter()
    phases: Counter = Counter()
    drops: list[float] = []
    n_net = n_net_wins = n_net_losses = 0
    try:
        with open(args.pgn) as fh:
            game_no = 0
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                game_no += 1
                if args.max_games and game_no > args.max_games:
                    break
                net_white = game.headers.get("White", "") == args.engine
                net_color = chess.WHITE if net_white else chess.BLACK
                n_net += 1
                result = game.headers.get("Result", "*")
                if result == "1/2-1/2":
                    continue
                net_won = (result == "1-0") == net_white
                if net_won:
                    n_net_wins += 1
                    continue
                n_net_losses += 1

                # Attribute by the net's own worst move: `ev.cp` reports
                # centipawns for the side to move, so before the net moves the
                # value is already net-POV, and after it the sign must flip.
                # Pivot-based detection finds nothing against a strong engine
                # because the net is rarely ahead to begin with.
                board = game.board()
                best_drop, best_fen, best_ply = 0.0, None, 0
                for ply, move in enumerate(game.mainline_moves()):
                    if board.turn != net_color:
                        board.push(move)
                        continue
                    before = ev.cp(board)          # side to move == net
                    if before is None or abs(before) >= MATE_CP:
                        board.push(move)
                        continue
                    fen_before = board.fen()
                    board.push(move)
                    after = ev.cp(board)           # side to move == opponent
                    if after is None:
                        break
                    if abs(after) >= MATE_CP:
                        continue                   # terminal moment, not the cause
                    drop = before - (-after)       # convert to net POV
                    if drop > best_drop:
                        best_drop, best_fen, best_ply = drop, fen_before, ply
                if best_fen is not None and best_drop >= 150:
                    kind = classify(chess.Board(best_fen), best_drop, best_ply)
                    counts[kind] += 1
                    drops.append(best_drop)
                    phases["opening" if best_ply <= 20 else
                           "middlegame" if best_ply <= 60 else "endgame"] += 1
    finally:
        ev.close()

    label = args.engine_label or args.engine
    print(f"## {label} — loss attribution ({args.pgn})")
    print(f"- games with a result: {n_net}, wins {n_net_wins}, losses {n_net_losses}")
    total = sum(counts.values()) or 1
    print(f"- decisive collapses found (drop >= 150cp): {sum(counts.values())}")
    for kind, c in counts.most_common():
        print(f"  {kind:10s} {c:4d}  ({100*c/total:.0f}%)")
    print("- worst-move phase:", ", ".join(f"{k} {v}" for k, v in phases.most_common()))
    if drops:
        drops.sort()
        print(f"- drop size: median {drops[len(drops)//2]:.0f}cp, "
              f"max {drops[-1]:.0f}cp")


if __name__ == "__main__":
    main()
