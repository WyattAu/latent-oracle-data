"""AMZ offline pilot (RESEARCH-NOVEL2.md N5, experiment ladder step 1).

Computes amortized-minimax policy targets with the net's OWN value head:
  Q(s,m) = min over K sampled replies r of V(child(child(s,m), r))
  target(s) = softmax(beta * Q)   [mixed with SF eval via alpha if labeled]
Fine-tunes from a base checkpoint on these targets, then reports puzzle-suite
match. No search, no oracle at inference; alpha annealing is stage-3, this
pilot fixes alpha=0.3 to keep an oracle floor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records  # noqa: E402
from model import ChessNet  # noqa: E402
from make_puzzles import codes_to_board  # noqa: E402

SYM2CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
            "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}


def board_from_codes(codes: np.ndarray, side: int):
    import chess
    b = chess.Board(None)
    for sq in range(64):
        c = int(codes[sq])
        if c:
            for sym, v in SYM2CODE.items():
                if v == c:
                    b.set_piece_at(sq, chess.Piece.from_symbol(sym))
    b.turn = chess.WHITE if side == 0 else chess.BLACK
    return b


def codes_from_board(b) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    for sq in range(64):
        p = b.piece_at(sq)
        if p:
            codes[sq] = SYM2CODE[p.symbol()]
    return codes


@torch.no_grad()
def values_batch(model, codes_list, sides, device: str) -> np.ndarray:
    """Batched value_of: win prob for the side to move, per row."""
    t = torch.from_numpy(np.stack(codes_list).astype(np.int64)).to(device)
    s = torch.from_numpy(np.array(sides, dtype=np.int64)).to(device)
    for i in range(0, t.shape[0], 1024):
        _, _, wdl = model(t[i:i + 1024], s[i:i + 1024])
        w3 = torch.softmax(wdl[:, 0] if False else wdl, -1)
        v = (w3[:, 0] + 0.5 * w3[:, 1]).float().cpu().numpy()
        yield v


@torch.no_grad()
def amz_targets(model, codes: np.ndarray, side: int, device: str,
                k_replies: int = 6, beta: float = 0.01, alpha: float = 0.3,
                sf_eval: float | None = None, rng=None) -> dict | None:
    """Collect all grandchildren first, evaluate in ONE batched forward."""
    import chess
    b = board_from_codes(codes, side)
    if b.is_game_over():
        return None
    moves = list(b.legal_moves)
    if not moves:
        return None
    per_move = []   # (n_replies, terminal_value_or_None)
    gc_codes, gc_sides = [], []
    for mv in moves:
        b.push(mv)
        replies = list(b.legal_moves)
        if not replies:
            per_move.append((0, 1.0 if b.is_checkmate() else 0.5))
        else:
            if len(replies) > k_replies:
                idx = rng.sample(range(len(replies)), k_replies) if rng else range(k_replies)
                replies = [replies[i] for i in idx]
            per_move.append((len(replies), None))
            for r in replies:
                b.push(r)
                gc_codes.append(codes_from_board(b))
                gc_sides.append(0 if b.turn == chess.WHITE else 1)
                b.pop()
        b.pop()
    vals_iter = values_batch(model, gc_codes, gc_sides, device)
    vals = next(vals_iter) if gc_codes else np.zeros(0)
    Qs, off = [], 0
    for n_rep, terminal in per_move:
        if terminal is not None:
            Qs.append(terminal)
        else:
            worst = float(vals[off:off + n_rep].min())
            off += n_rep
            Qs.append(worst)
    Q = np.array(Qs, dtype=np.float64)
    p_amz = np.exp(beta * (Q - Q.max()))
    p_amz /= p_amz.sum()
    # oracle floor: blend with the played move (BC prior) via alpha on SF eval
    if sf_eval is not None and alpha > 0:
        p_sf = np.zeros_like(p_amz)
        # move played in this record is unknown here; uniform oracle blend on Q
        k = 0.00368208
        pw = math.tanh(sf_eval / 1200.0) / 2 + 0.5  # rough match to logistic
        Q_oracle = np.full_like(Q, np.clip(pw, 0.02, 0.98))
        Q_mix = alpha * Q_oracle + (1 - alpha) * Q
        p = np.exp(beta * (Q_mix - Q_mix.max()))
        p /= p.sum()
    else:
        p = p_amz
    return {"moves": [(m.from_square, m.to_square) for m in moves], "p": p.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="base .pt checkpoint")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="runs/amz_pilot")
    ap.add_argument("--positions", type=int, default=100_000)
    ap.add_argument("--k-replies", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    import random
    pyrng = random.Random(0)

    model = ChessNet().to(device)
    model.load_state_dict(torch.load(args.net, map_location="cpu", weights_only=True))
    model.eval()
    print(f"loaded {args.net} on {device}", flush=True)

    cache = os.path.join(args.out, "amz_targets.pt")
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(cache):
        samples = torch.load(cache, weights_only=False)
        print(f"loaded {len(samples)} cached target samples", flush=True)
    else:
        samples = []
        for s in iter_records(args.shard):
            if len(samples) >= args.positions:
                break
            t = amz_targets(model, s.board_codes, s.side, device,
                            k_replies=args.k_replies, beta=args.beta, rng=pyrng)
            if t is None:
                continue
            samples.append((s.board_codes.astype(np.int64), s.side,
                            t["moves"], t["p"]))
            if len(samples) % 20000 == 0:
                print(f"  {len(samples)} positions targeted", flush=True)
        torch.save(samples, cache)
        print(f"AMZ targets for {len(samples)} positions -> {cache}", flush=True)

    # fine-tune: policy CE against the AMZ soft distribution (value loss kept
    # on the played-move game result to stabilize)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    for epoch in range(args.epochs):
        order = np.random.permutation(len(samples))
        done = 0
        for i in range(0, len(samples) - args.batch, args.batch):
            idx = order[i:i + args.batch]
            codes_b = np.stack([samples[j][0] for j in idx])
            sides = np.array([samples[j][1] for j in idx])
            codes_t = torch.from_numpy(codes_b.astype(np.int64)).to(device)
            side_t = torch.from_numpy(sides.astype(np.int64)).to(device)
            scores, promo, wdl = model(codes_t, side_t)
            flat = scores.reshape(codes_t.shape[0], 64 * 64)
            loss = 0.0
            for bi, j in enumerate(idx):
                moves, p = samples[j][2], np.array(samples[j][3], dtype=np.float32)
                mv_idx = torch.tensor([u * 64 + v for u, v in moves], device=device)
                logp = F.log_softmax(flat[bi, mv_idx], dim=-1)
                pt = torch.from_numpy(p).to(device)
                loss = loss + -(pt * logp).sum()
            loss = loss / len(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            done += len(idx)
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}: amz_ce {loss.item():.4f} "
                      f"({done} positions)", flush=True)
        torch.save(model.state_dict(), os.path.join(args.out, f"amz_e{epoch}.pt"))
        print(f"epoch {epoch}: saved {args.out}/amz_e{epoch}.pt", flush=True)
    print("pilot done — evaluate with score_puzzles.py + fast SPRT", flush=True)


if __name__ == "__main__":
    main()
