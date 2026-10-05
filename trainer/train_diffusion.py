"""DiffuSearch pilot: discrete-diffusion chess policy (docs/SPEC-DIFFUSION.md).

Training data comes from CONSECUTIVE shard records: the shard worker writes
positions in game order, so record i+1 is usually the position after record
i. A sample = (source state s_t, [a_0, s_1, a_1, s_2, ...]) for as many
consecutive steps as the shard run allows (up to --horizon).

Sequence layout (h = horizon):
    [SEP] src(67) [SEP] a0 s1(67) a1 s2(67) ... a_{h-1}
Source tokens are never masked. Target tokens are masked with prob
(t+1)/T at a sampled diffusion step t; CE loss on masked positions only,
weighted 1/(t+1).
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records  # noqa: E402
from model import Block  # noqa: E402
from make_puzzles import codes_to_board, PIECE_TO_PC  # noqa: E402

SPECIALS = {"PAD": 0, "MASK": 1, "SEP": 2}
BOARD_CHARS = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
# shard piece-code -> BOARD_CHARS index (codes 9-14 are black; a direct index
# was shifted by two — black pawns encoded as bishops, queens/kings empty)
CODE_TO_CHAR = {0: ".", 1: "P", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K",
                9: "p", 10: "n", 11: "b", 12: "r", 13: "q", 14: "k"}
CASTLE_CHARS = [f"C{i}" for i in range(16)]  # castling-rights bitmask 0..15
EP_CHARS = ["-"] + [f"E{f}" for f in "abcdefgh"]
SIDE_CHARS = ["sw", "sb"]  # NOT "w"/"b": collides with piece chars in the shared vocab


MOVE_BASE = (len(SPECIALS) + len(BOARD_CHARS) + len(CASTLE_CHARS) +
             len(EP_CHARS) + len(SIDE_CHARS))


def build_vocab() -> dict[str, int]:
    v = dict(SPECIALS)
    for c in BOARD_CHARS:
        v[c] = len(v)
    for c in CASTLE_CHARS:
        v[c] = len(v)
    for c in EP_CHARS:
        v[c] = len(v)
    for c in SIDE_CHARS:
        v[c] = len(v)
    for f in range(64):
        for t in range(64):
            v[f"M_{f}_{t}"] = MOVE_BASE + f * 64 + t
    for p in range(4):
        v[f"P_{p}"] = MOVE_BASE + 4096 + p
    return v


VOCAB = build_vocab()
V = len(VOCAB)
STATE_LEN = 67  # 64 board + side + castling + ep
SEP = SPECIALS["SEP"]
MASK = SPECIALS["MASK"]
PAD = SPECIALS["PAD"]


SYM2CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
            "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}


def encode_state(board_codes: np.ndarray, side: int, castling: int, ep: int) -> list[int]:
    toks = []
    for code in board_codes:
        toks.append(VOCAB[CODE_TO_CHAR.get(int(code), ".")])
    toks.append(VOCAB[SIDE_CHARS[side]])
    toks.append(VOCAB[CASTLE_CHARS[castling & 15]])
    toks.append(VOCAB[EP_CHARS[0] if ep >= 8 else EP_CHARS[1 + ep]])
    while len(toks) < STATE_LEN:
        toks.append(PAD)
    return toks[:STATE_LEN]


def move_token(u: int, v: int, promo: int) -> int:
    if promo:
        return VOCAB[f"P_{promo - 1}"]
    return VOCAB[f"M_{u}_{v}"]


def find_connecting_move(prev_codes: np.ndarray, prev_side: int, prev_castle: int, prev_ep: int,
                         next_codes: np.ndarray, next_side: int):
    """Legal move m s.t. apply(prev, m) == next (piece placement + side)."""
    import chess
    board = codes_to_board(prev_codes, prev_side)
    if board.is_game_over():
        return None
    want_side = chess.WHITE if next_side == 0 else chess.BLACK
    for mv in board.legal_moves:
        board.push(mv)
        ok = board.turn == want_side
        if ok:
            for sq in range(64):
                p = board.piece_at(sq)
                code = 0 if p is None else SYM2CODE[p.symbol()]
                if code != next_codes[sq]:
                    ok = False
                    break
        board.pop()
        if ok:
            promo = 0
            if mv.promotion:
                promo = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[mv.promotion]
            return (mv.from_square, mv.to_square, promo)
    return None


def state_from_record(s) -> tuple[np.ndarray, int, int, int]:
    return s.board_codes, s.side, s.castling, s.ep


class DiffuNet(nn.Module):
    def __init__(self, d: int = 256, layers: int = 8, heads: int = 8, dff: int = 1024,
                 max_len: int = 256):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, heads, dff) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Linear(d, V, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device)).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))  # (B, T, V)


def build_samples(shard: str, horizon: int, max_samples: int, seed: int = 0):
    """Consecutive-record windows -> (source, targets) token lists."""
    rng = random.Random(seed)
    samples = []
    prev = None  # (codes, side, castling, ep)
    run: list = []  # list of records in the current consecutive run
    n_scanned = 0
    for s in iter_records(shard):
        n_scanned += 1
        if n_scanned % 200000 == 0:
            print(f"  scan {n_scanned}: {len(samples)} samples", flush=True)
        cur = state_from_record(s)
        if prev is not None:
            mv = find_connecting_move(*prev, cur[0], cur[1])
            if mv is not None:
                run.append((prev, mv, cur))
            else:
                run = []
        prev = cur
        if len(run) >= horizon:
            window = run[-horizon:]
            src = window[0][0]
            toks = [SEP] + encode_state(*src) + [SEP]
            ok = True
            for _, mv, nxt in window:
                toks.append(move_token(*mv))
                toks += encode_state(*nxt)
            samples.append(toks)
            run = run[-(horizon - 1):] if horizon > 1 else []
        if max_samples and len(samples) >= max_samples:
            break
    rng.shuffle(samples)
    return samples


def diffu_loss(model, batch_ids: torch.Tensor, src_len: int, T: int, device: str):
    """Sample t per row; mask targets with prob (t+1)/T; CE on masked."""
    B, L = batch_ids.shape
    t = torch.randint(0, T, (B,), device=device)
    mask_prob = ((t + 1).float() / T).unsqueeze(1)  # (B,1)
    target_mask = torch.ones(B, L, device=device)
    target_mask[:, :src_len] = 0.0  # SEP + source + SEP frozen
    drop = torch.rand(B, L, device=device) < mask_prob
    drop = drop * target_mask.bool()
    labels = batch_ids.clone()
    ids = batch_ids.clone()
    ids[drop] = MASK
    logits = model(ids)  # (B,L,V)
    logp = F.log_softmax(logits, dim=-1)
    ll = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B,L)
    w = 1.0 / (t.float() + 1.0)  # (B,)
    ll = ll * target_mask  # only target positions contribute
    per_row = ll.sum(1) / target_mask.sum(1).clamp(min=1.0)
    return -(per_row * w).sum() / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="runs/diffu")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--T", type=int, default=64, help="diffusion steps")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--max-samples", type=int, default=2_000_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=0, help="cap steps per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", type=int, default=2000, help="samples for eval")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    src_len = 1 + STATE_LEN + 1
    max_len = src_len + args.horizon * (1 + STATE_LEN)
    print(f"vocab={V} max_len={max_len} (h={args.horizon})", flush=True)

    cache = args.out + f"_h{args.horizon}_samples.pt"
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(cache):
        samples = torch.load(cache, weights_only=False)
        print(f"loaded {len(samples)} cached samples", flush=True)
    else:
        print("building samples from shard pairs (this scans the shard)...", flush=True)
        samples = build_samples(args.shard, args.horizon, args.max_samples)
        torch.save(samples, cache)
        print(f"built {len(samples)} samples -> {cache}", flush=True)

    hold = samples[:args.holdout]
    train = samples[args.holdout:]

    model = DiffuNet(args.d, args.layers, args.heads, args.dff, max_len=max_len).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M on {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def eval_match():
        model.eval()
        hits = 0
        with torch.no_grad():
            for i in range(0, len(hold), 64):
                chunk = hold[i:i + 64]
                ids = torch.tensor(chunk, dtype=torch.long, device=device)
                a0_pos = src_len  # first action token position
                # single full reveal: measure a0 accuracy from a 50%-masked pass
                t = torch.full((len(chunk),), args.T // 2 - 1, device=device)
                mask_prob = float((args.T // 2) / args.T)
                drop = (torch.rand_like(ids, dtype=torch.float) < mask_prob)
                drop[:, :src_len] = False
                noised = ids.clone()
                noised[drop] = MASK
                logits = model(noised)
                pred = logits[:, a0_pos].argmax(-1)
                hits += int((pred == ids[:, a0_pos]).sum())
        model.train()
        return hits / max(1, len(hold))

    step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(len(train))
        done = 0
        for i in range(0, len(train) - args.batch, args.batch):
            idx = order[i:i + args.batch]
            batch = torch.tensor([train[j] for j in idx], dtype=torch.long, device=device)
            loss = diffu_loss(model, batch, src_len, args.T, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            done += len(idx)
            if step % 10 == 0:
                print(f"epoch {epoch} step {step}: loss {loss.item():.4f} "
                      f"({done} samples this epoch)", flush=True)
            if args.steps and step % max(1, args.steps) == 0:
                break
            if done >= len(train):
                break
        acc = eval_match()
        print(f"epoch {epoch}: a0 match {acc:.3f}", flush=True)
        import json
        cfg = dict(d=args.d, layers=args.layers, heads=args.heads, dff=args.dff,
                   max_len=max_len, horizon=args.horizon, T=args.T)
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"))
        torch.save(model.state_dict(), os.path.join(args.out, f"diffu_e{epoch}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
