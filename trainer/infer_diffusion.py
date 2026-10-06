"""DiffuSearch pilot inference: progressive denoising (SPEC-DIFFUSION.md §5).

Starts all target tokens as MASK and denoises over T steps with
easy-first unmasking (most-confident tokens revealed first). The a0 action
token is decoded to a move; an optional legal gate restricts the a0 softmax
to moves legal in the source position.
"""
from __future__ import annotations

import argparse
import os
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_diffusion as td  # noqa: E402
from robust_io import load_artifact  # noqa: E402
from train_diffusion import (  # noqa: E402
    MASK,
    MOVE_BASE,
    STATE_LEN,
    SYM2CODE,
    DiffuNet,
)


def id_to_move_tok(i: int):
    """Token id -> (from, to, promo) or None for non-move tokens."""
    if MOVE_BASE <= i < MOVE_BASE + 4096:
        r = i - MOVE_BASE
        return (r // 64, r % 64, 0)
    if MOVE_BASE + 4096 <= i < MOVE_BASE + 4100:
        return (255, 255, i - MOVE_BASE - 4096 + 1)
    return None


def decode_source_board(src_toks: list[int]):
    import chess
    board = chess.Board(None)
    for sq in range(64):
        ch = [k for k, v in td.VOCAB.items() if v == src_toks[1 + sq]][0]
        if ch in SYM2CODE:
            board.set_piece_at(sq, chess.Piece.from_symbol(ch))
    # side tokens are "sw"/"sb": plain "b" collided with the black bishop
    board.turn = (chess.WHITE if src_toks[1 + 64] == td.VOCAB[td.SIDE_CHARS[0]]
                 else chess.BLACK)
    return board


def legal_move_token_ids(src_toks: list[int]) -> set[int]:
    board = decode_source_board(src_toks)
    ids = set()
    for mv in board.legal_moves:
        promo = 0
        if mv.promotion:
            promo = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[mv.promotion]
        if promo:
            ids.add(td.VOCAB[f"P_{promo - 1}"])
        else:
            ids.add(td.VOCAB[f"M_{mv.from_square}_{mv.to_square}"])
    return ids


@torch.no_grad()
def denoise(model: DiffuNet, sample: list[int], src_len: int, T: int, device: str,
            legal_gate: bool = False) -> tuple[int, int]:
    """Returns (predicted a0 token, true a0 token)."""
    row = np.asarray(sample, dtype=np.int64)
    ids = torch.from_numpy(row).unsqueeze(0).to(device)
    true_a0 = int(ids[0, src_len])
    x = ids.clone()
    x[0, src_len:] = MASK
    remaining = x[0] == MASK
    for t in range(T - 1, -1, -1):
        logits = model(x)  # (1, L, V)
        probs = F.softmax(logits, dim=-1)
        c, pred = probs.max(-1)  # (1, L)
        if legal_gate:
            gate = torch.full_like(logits[0, src_len], float("-inf"))
            legal = legal_move_token_ids(sample[:src_len])
            for tid in legal:
                gate[tid] = 0.0
            a0_logits = logits[0, src_len] + gate
            a0_probs = F.softmax(a0_logits, dim=-1)
            c[0, src_len] = a0_probs.max()
            pred[0, src_len] = a0_probs.argmax()
        n_masked = int(remaining.sum())
        if n_masked == 0:
            break
        # easy-first: unmask ~1/(t+1) of the remaining, at least 1
        n_unmask = max(1, n_masked // (t + 1))
        score = torch.where(remaining, c[0], torch.full_like(c[0], -1.0))
        order = score.argsort(descending=True)
        for j in order[:n_unmask]:
            x[0, j] = pred[0, j]
            remaining[j] = False
    return int(x[0, src_len]), true_a0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with diffu_eN.pt + samples cache")
    ap.add_argument("--ckpt", default="diffu_e1.pt")
    ap.add_argument("--T", type=int, default=8, help="denoising steps (small = fast eval)")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--legal-gate", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = None
    for f in os.listdir(args.run):
        if f.endswith("_samples.pt"):
            cache = os.path.join(args.run, f)
    if cache is None:
        # trainer writes the cache next to the run dir: <run>_hN_samples.pt
        import glob
        cands = glob.glob(args.run.rstrip("/") + "_*_samples.pt")
        cache = cands[0] if cands else None
    assert cache, "samples cache not found"
    # the sample cache is a numpy array, not a state dict: weights_only=True
    # would reject it outright
    samples = load_artifact(cache, weights_only=False)
    import json
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    model = DiffuNet(cfg["d"], cfg["layers"], cfg["heads"], cfg["dff"],
                     max_len=cfg["max_len"]).to(device)
    model.load_state_dict(
        load_artifact(os.path.join(args.run, args.ckpt), weights_only=True))
    model.eval()
    src_len = 1 + STATE_LEN + 1
    hits = gate_hits = n = 0
    for sample in samples[:args.limit]:
        pred, true_a0 = denoise(model, sample, src_len, args.T, device,
                                legal_gate=args.legal_gate)
        n += 1
        hits += int(pred == true_a0)
        if args.legal_gate:
            gate_hits += int(pred == true_a0)
    print(f"n={n} T={args.T} legal_gate={args.legal_gate}: "
          f"a0 match {hits / max(1, n):.3f}"
          + (f" (gated {gate_hits / max(1, n):.3f})" if args.legal_gate else ""))


if __name__ == "__main__":
    main()
