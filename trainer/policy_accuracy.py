"""Calibrate "action accuracy" against a fixed reference for every policy.

The DiffuSearch verdict showed that agreement with the *human* move is not a
strength metric. This measures something comparable across policies: how often
each policy picks Stockfish's best move, taken from the labeled shard's target
list (no extra analysis needed).

Policies compared:
  bc-greedy    the BC net's argmax over legal moves (the engine's own policy)
  bc-greedy+mirror  the same with MirrorAvg enabled
  diffusion    DiffuSearch a0 under the intended denoising schedule

Reading: if the diffusion policy trails bc-greedy against the same target, the
0-12 strength result has an independent explanation rather than being an
artifact of the playout harness.

Usage:
  python trainer/policy_accuracy.py --shard <labeled.shard> \
      --bc-net <net.pt> [--diffu-run runs/diffu_v1 --ckpt diffu_e0.pt \
      --cache <samples.pt>] [--positions 400] [--T 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import diffusion_playout as dp  # noqa: E402
from format import iter_records  # noqa: E402
from model import ChessNet  # noqa: E402
from robust_io import load_artifact  # noqa: E402
from train_diffusion import STATE_LEN  # noqa: E402


def best_target(rec) -> str | None:
    """Shard targets are (from, to, promo) triples; 255 marks an unused slot."""
    for f, t, promo in rec.targets:
        if f == 255 or t == 255:
            continue
        p = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promo] \
            if promo not in (0, 255) else None
        try:
            return chess.Move(f, t, promotion=p).uci()
        except ValueError:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--bc-net", required=True)
    ap.add_argument("--v3", action="store_true",
                    help="the net uses the v3 architecture (castle/ep/value buckets)")
    ap.add_argument("--diffu-run", default="")
    ap.add_argument("--ckpt", default="diffu_e1.pt")
    ap.add_argument("--cache", default="")
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc = ChessNet(v3=args.v3).to(device)
    bc.load_state_dict(load_artifact(args.bc_net, weights_only=True))
    bc.eval()

    dnet = None
    if args.diffu_run and args.cache:
        cfg = json.load(open(Path(args.diffu_run) / "config.json"))
        from train_diffusion import DiffuNet
        dnet = DiffuNet(cfg["d"], cfg["layers"], cfg["heads"], cfg["dff"],
                        max_len=cfg["max_len"]).to(device)
        dnet.load_state_dict(load_artifact(str(Path(args.diffu_run) / args.ckpt),
                                           weights_only=True))
        dnet.eval()

    import random
    rng = random.Random(args.seed)

    n = 0
    hits = Counter()
    legal_n = 0
    for i, rec in enumerate(iter_records(args.shard)):
        if i < args.skip:
            continue
        best = best_target(rec)
        if best is None:
            continue
        board = dp.board_from_codes_state(_state_tokens(rec))
        if board.king(chess.WHITE) is None or board.is_game_over():
            continue
        try:
            want = chess.Move.from_uci(best)
        except ValueError:
            continue
        if want not in board.legal_moves:
            continue          # shard/board mismatch: skip rather than mislead
        legal_n += 1
        n += 1

        greedy = dp.greedy_move(bc, board, v3=args.v3)
        if greedy is not None:
            hits["bc"] += (greedy == want)
        if dnet is not None:
            import random as _r
            mv = dp.diffusion_move(dnet, board, args.T, 0.0, _r.Random(0))
            if mv is not None:
                hits["diffusion"] += (mv == want)
                hits["agree_with_bc"] += (mv == greedy)
        if n >= args.positions:
            break

    print(f"## policy accuracy vs Stockfish's best move ({n} positions)")
    print(f"- bc-greedy:      {hits['bc']}/{n} = {hits['bc']/max(1,n):.3f}")
    if dnet is not None:
        print(f"- diffusion:      {hits['diffusion']}/{n} = {hits['diffusion']/max(1,n):.3f}")
        print(f"- the two agree:  {hits['agree_with_bc']}/{n} = {hits['agree_with_bc']/max(1,n):.3f}")
    return 0


def _state_tokens(rec) -> list[int]:
    """Build a source-state token run from a shard record (for the decoder)."""
    import train_diffusion as td
    codes = np.asarray(rec.board_codes, dtype=np.int64)
    toks = [td.SEP] + td.encode_state(codes, int(rec.side), int(rec.castling),
                                       int(rec.ep)) + [td.SEP]
    return toks


if __name__ == "__main__":
    sys.exit(main())
