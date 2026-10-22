"""AV-head training on an engine-format shard (Loop: label-paradigm A/B).

Same records, same count, same trunk as the BC control (train.py, 8x256 v1),
but the target is the SF move's win probability (HL-Gauss over 128 bins)
instead of the one-hot move — isolating soft vs hard labels at fixed data.
Faithful to DeepNoc's action-value training: each (state, played move, win
prob) trains the move-conditional head; distributions emerge at inference
by scoring all legal moves.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from chessbench_av import (AVHead, expected_win, hl_gauss_targets,
                           NUM_BINS, PIECE_CODE)

from format import iter_records, count_records
from model import ChessNet


def shard_to_arrays(path, cap):
    """Stream the shard into preallocated CPU arrays (mover-POV win prob)."""
    n_total = count_records(path)
    n = min(n_total, cap) if cap else n_total
    X_codes = np.zeros((n, 64), dtype=np.uint8)
    X_meta = np.zeros((n, 4), dtype=np.int32)   # side, castling, ep_feat, promo
    X_from = np.zeros(n, dtype=np.int16)
    X_to = np.zeros(n, dtype=np.int16)
    Y_win = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    kept = 0
    for i, s in enumerate(iter_records(path)):
        if kept >= n:
            break
        if not s.targets:
            continue
        frm, to, promo = s.targets[0]
        if frm >= 64 or to >= 64:
            continue
        X_codes[kept] = s.board_codes
        ep_feat = 0 if s.ep >= 64 else 1 + (s.ep % 8)
        X_meta[kept] = (s.side, s.castling, ep_feat, 0)
        X_from[kept] = frm
        X_to[kept] = to
        # shard wdl is white-POV; the AV target convention is mover-POV
        Y_win[kept] = float(s.wdl[0]) if s.side == 0 else 1.0 - float(s.wdl[0])
        kept += 1
        if kept % 5_000_000 == 0:
            print(f"  loaded {kept} records ({time.time()-t0:.0f}s)", flush=True)
    print(f"loaded {kept} records in {time.time()-t0:.0f}s", flush=True)
    return (X_codes[:kept], X_meta[:kept], X_from[:kept], X_to[:kept],
            Y_win[:kept])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=0, help="0 = one epoch")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--record-limit", type=int, default=0)
    ap.add_argument("--export-every", type=int, default=20000)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    (X_codes, X_meta, X_from, X_to, Y_win) = shard_to_arrays(
        args.shard, args.record_limit)
    n = len(X_codes)
    steps = args.steps or max(1, n // args.batch)
    print(f"{n} records, {steps} steps, batch {args.batch}", flush=True)

    feats = {}
    trunk = ChessNet(v3=False).to(device)
    trunk.blocks[-1].register_forward_hook(
        lambda m, i, o: feats.__setitem__("h", o))
    with torch.no_grad():
        trunk(torch.zeros(1, 64, dtype=torch.long, device=device),
              torch.zeros(1, dtype=torch.long, device=device))
        D = feats["h"].shape[-1]
    head = AVHead(D, NUM_BINS, style="pooled").to(device)
    params = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    codes_t = torch.from_numpy(X_codes.astype(np.int64))
    side_t = torch.from_numpy(X_meta[:, 0].astype(np.int64))
    from_t = torch.from_numpy(X_from.astype(np.int64))
    to_t = torch.from_numpy(X_to.astype(np.int64))
    win_t = torch.from_numpy(Y_win)

    consec_bad = 0
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, (args.batch,))
        codes_b = codes_t[idx].to(device)
        side_b = side_t[idx].to(device)
        from_b = from_t[idx].to(device)
        to_b = to_t[idx].to(device)
        win_b = win_t[idx].to(device)
        trunk(codes_b, side_b)
        state = feats["h"].mean(dim=1)
        logits = head(state, from_b, to_b,
                      torch.zeros_like(from_b))
        target = hl_gauss_targets(win_b)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            consec_bad += 1
            opt.zero_grad(set_to_none=True)
            if consec_bad >= 50:
                print("FATAL: 50 consecutive non-finite losses", flush=True)
                return
            continue
        consec_bad = 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            with torch.no_grad():
                ew = expected_win(logits)
                acc = ((logits.argmax(-1) == target.argmax(-1)).float().mean())
                cal = (ew - win_b).abs().mean()
            print(f"step {step}: loss {loss.item():.4f} bucket-acc {acc.item():.3f} "
                  f"|E[win]-wp| {cal.item():.4f} consec_bad {consec_bad} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.export_every == 0 or step == steps:
            torch.save({"trunk": trunk.state_dict(), "head": head.state_dict(),
                        "style": "pooled"},
                       f"{args.out}/av_step{step}.pt")
            print(f"exported {args.out}/av_step{step}.pt", flush=True)


if __name__ == "__main__":
    main()
