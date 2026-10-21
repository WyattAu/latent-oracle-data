"""NetQ parity gate: python quantized forward vs the C++ engine's bestmove.

The champion (net_q_v2, LOQW INT8) runs on the engine's NetQ path. This gate
feeds the same positions to loqw_sim.py's bit-matched quantized forward and
to the engine (RecyclePasses=1, no Syzygy) and requires identical bestmoves.
Any mismatch is a parity bug that silently changes strength.
"""
import math
import subprocess
import sys

import chess

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from loqw_sim import Loqw, forward  # noqa: E402

ENGINE = "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle"
import os
BLOB = os.environ.get("NETQ_BLOB", "/home/wyatt/data/runs/bc/net_q_v2.bin")

PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 9, (chess.KNIGHT, chess.BLACK): 10,
    (chess.BISHOP, chess.BLACK): 11, (chess.ROOK, chess.BLACK): 12,
    (chess.QUEEN, chess.BLACK): 13, (chess.KING, chess.BLACK): 14,
}


def fen_inputs(fen):
    b = chess.Board(fen)
    codes = [0] * 64
    for sq in chess.SQUARES:
        pc = b.piece_at(sq)
        if pc is not None:
            codes[sq] = PIECE_CODE[(pc.piece_type, pc.color)]
    return codes, (0 if b.turn == chess.WHITE else 1), b


def engine_bestmove(fen):
    p = subprocess.run(
        [ENGINE], input=f"uci\nsetoption name WeightsFile value {BLOB}\n"
        f"isready\nposition fen {fen}\ngo\nquit\n",
        capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        if line.startswith("bestmove"):
            return line.split()[1]
    raise RuntimeError(f"no bestmove for {fen}: {p.stdout[-400:]}")


def python_bestmove(m, fen):
    codes, side, b = fen_inputs(fen)
    import numpy as np
    import torch
    scores, promo, _ = forward(
        m, torch.tensor(codes, dtype=torch.long).view(1, 64), side)
    best, best_sc = None, -1e30
    for mv in b.legal_moves:
        s = float(scores[mv.from_square][mv.to_square])
        if mv.promotion:
            idx = {chess.KNIGHT: 0, chess.BISHOP: 1,
                   chess.ROOK: 2, chess.QUEEN: 3}[mv.promotion]
            s += float(promo[idx])
        if s > best_sc:
            best_sc, best = s, mv.uci()
    return best


def main():
    m = Loqw(BLOB)
    m.parse()
    fens = []
    with open("/home/wyatt/tools/chess/openings.epd") as f:
        for i, line in enumerate(f):
            if i % 40 == 0:
                fens.append(line.split()[0:6] and " ".join(line.split()[:6]))
    fens = fens[:50]
    bad = 0
    for i, fen in enumerate(fens):
        try:
            e = engine_bestmove(fen)
            p = python_bestmove(m, fen)
        except Exception as ex:
            print(f"[{i}] ERROR {fen}: {ex}")
            bad += 1
            continue
        if e != p:
            bad += 1
            print(f"[{i}] MISMATCH {fen}: engine={e} python={p}")
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(fens)}] checked, {bad} mismatches", flush=True)
    print(f"PARITY {'FAIL' if bad else 'PASS'}: {bad}/{len(fens)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
