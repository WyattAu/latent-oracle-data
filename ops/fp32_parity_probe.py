"""FP32-path parity probe: is the engine's LONW (FP32) path faithful to the
torch model? Triangle on the same positions:
  python_fp32 (torch ChessNet on net_e1.pt, shard encoding black=pt+8)
  engine_fp32 (net_e1.bin)
  engine_int8 (net_q_v2.bin ~= quantized net_e1)
If python_fp32 ~ engine_int8 but python_fp32 != engine_fp32, the LONW path
is the broken one and every FP32-blob verdict in the program inherits it.
"""
import subprocess
import sys

import chess
import torch

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet  # noqa: E402

ENGINE = "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle"
PT = "/home/wyatt/data/runs/bc/net_e1.pt"
FP32_BLOB = "/home/wyatt/data/runs/bc/net_e1.bin"
INT8_BLOB = "/home/wyatt/data/runs/bc/net_q_v2.bin"

PIECE_CODE = {
    (chess.PAWN, chess.WHITE): 1, (chess.KNIGHT, chess.WHITE): 2,
    (chess.BISHOP, chess.WHITE): 3, (chess.ROOK, chess.WHITE): 4,
    (chess.QUEEN, chess.WHITE): 5, (chess.KING, chess.WHITE): 6,
    (chess.PAWN, chess.BLACK): 9, (chess.KNIGHT, chess.BLACK): 10,
    (chess.BISHOP, chess.BLACK): 11, (chess.ROOK, chess.BLACK): 12,
    (chess.QUEEN, chess.BLACK): 13, (chess.KING, chess.BLACK): 14,
}

from lonw_sim import Lonw, forward as lonw_forward  # noqa: E402

_m = Lonw(PT if False else FP32_BLOB)  # the blob IS the .pt's weights (verified)
_m.parse()


def python_bestmove(fen):
    b = chess.Board(fen)
    codes = [0] * 64
    for sq in chess.SQUARES:
        pc = b.piece_at(sq)
        if pc is not None:
            codes[sq] = PIECE_CODE[(pc.piece_type, pc.color)]
    side = 0 if b.turn == chess.WHITE else 1
    with torch.no_grad():
        scores, promo, _ = lonw_forward(_m, torch.tensor(codes).view(1, 64), side)
    best, best_sc = None, -1e30
    for mv in b.legal_moves:
        s = float(scores[mv.from_square, mv.to_square])
        if mv.promotion:
            idx = {chess.KNIGHT: 0, chess.BISHOP: 1,
                   chess.ROOK: 2, chess.QUEEN: 3}[mv.promotion]
            s += float(promo[0, idx])
        if s > best_sc:
            best_sc, best = s, mv.uci()
    return best


def engine_bestmove(fen, blob):
    p = subprocess.run(
        [ENGINE],
        input=f"uci\nsetoption name WeightsFile value {blob}\n"
              f"isready\nposition fen {fen}\ngo\nquit\n",
        capture_output=True, text=True, timeout=60)
    for line in p.stdout.splitlines():
        if line.startswith("bestmove"):
            return line.split()[1]
    raise RuntimeError(f"no bestmove: {p.stdout[-300:]}")


def main():
    fens = []
    with open("/home/wyatt/tools/chess/openings.epd") as f:
        for i, line in enumerate(f):
            if i % 40 == 0:
                fens.append(" ".join(line.split()[:6]))
    fens = fens[:20]
    t = {"py_vs_efp32": 0, "py_vs_eint8": 0, "efp32_vs_eint8": 0}
    for i, fen in enumerate(fens):
        py = python_bestmove(fen)
        efp = engine_bestmove(fen, FP32_BLOB)
        ei8 = engine_bestmove(fen, INT8_BLOB)
        t["py_vs_efp32"] += py != efp
        t["py_vs_eint8"] += py != ei8
        t["efp32_vs_eint8"] += efp != ei8
        if (i + 1) % 10 == 0:
            print(f"[{i+1}] py~efp32:{t['py_vs_efp32']} py~ei8:{t['py_vs_eint8']} "
                  f"efp32~ei8:{t['efp32_vs_eint8']}", flush=True)
    print(f"FINAL {len(fens)} positions: {t}")


if __name__ == "__main__":
    main()
