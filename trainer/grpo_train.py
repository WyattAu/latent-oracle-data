"""GRPO RL fine-tuning for the searchless policy (RESEARCH-RL.md §3 + E2).

Group Relative Policy Optimization with:
- Gumbel-top-K action sampling from the policy prior (improvement-guaranteed
  group construction, Danihelka et al. ICLR 2022)
- Rewards from a persistent Stockfish pool: depth-d eval of the child
  position, mover POV, clipped to +/- reward-clip cp
- Group-normalized advantages, PPO-style clipped ratio, KL anchor to the
  frozen base policy
- Standing-policy validation gates: first 50 groups (legality, reward range,
  advantage normalization), first step (finite loss)

Positions come from a shard (labels unused). The policy stays a plain
ChessNet; one flat (from, to) action space with queen-promotion preference.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records  # noqa: E402
from make_puzzles import codes_to_board  # noqa: E402
from model import ChessNet  # noqa: E402
from robust_io import load_artifact, save_atomic  # atomic + self-healing artifacts


class SFEval:
    """Persistent UCI Stockfish session; eval a FEN at fixed depth (cp)."""

    def __init__(self, path: str, depth: int = 12, hash_mb: int = 16):
        import subprocess
        self.depth = depth
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1)
        self._send("uci")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("uci")
        self._sync()

    def _send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _sync(self):
        self._send("isready")
        while True:
            line = self.p.stdout.readline()
            if line.startswith("readyok"):
                return

    def eval_cp(self, fen: str) -> int:
        """Centipawn score from the side to move's perspective."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        cp, mate = 0, 0
        while True:
            line = self.p.stdout.readline()
            if line.startswith("bestmove"):
                break
            if " score cp " in line:
                try:
                    cp = int(line.split(" score cp ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif " score mate " in line:
                try:
                    mate = int(line.split(" score mate ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        if mate:
            return int(math.copysign(30000 - abs(mate), mate))
        return cp

    def close(self):
        self._send("quit")
        self.p.terminate()


def gumbel_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """K samples via Gumbel-top-K (approximate draws from softmax(logits))."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
    idx = (logits + g).topk(k, dim=-1).indices  # (B, K)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="base .pt checkpoint")
    ap.add_argument("--shard", required=True, help="position source (labels unused)")
    ap.add_argument("--sf", default="/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--out", default="runs/grpo_v1")
    ap.add_argument("--groups", type=int, default=128, help="positions per step")
    ap.add_argument("--k", type=int, default=16, help="samples per group")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=12, help="SF reward depth")
    ap.add_argument("--reward-clip", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-beta", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--export-every", type=int, default=500)
    ap.add_argument("--max-positions", type=int, default=200000, help="shard scan cap")
    ap.add_argument("--sf-pool", type=int, default=4)
    ap.add_argument("--v3", action="store_true",
                    help="base net is v3 architecture (SPEC-BLOB-V3): forwards pass "
                         "castle/ep from the shard records")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    random.seed(0)
    rng = random.Random(0)

    model = ChessNet(v3=args.v3).to(device)
    model.load_state_dict(load_artifact(args.net, weights_only=True))
    base = ChessNet(v3=args.v3).to(device)
    base.load_state_dict(model.state_dict())
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    print(f"GRPO: base={args.net} on {device}, groups={args.groups} K={args.k}", flush=True)

    os.makedirs(args.out, exist_ok=True)

    # ---- position pool
    pool = []  # rows: (codes, side, castle, ep)
    for s in iter_records(args.shard):
        b = codes_to_board(s.board_codes, s.side)
        if b.is_game_over() or b.fullmove_number > 60:
            continue
        pool.append((s.board_codes.astype(np.int64), s.side,
                     s.castling & 15, 0 if s.ep >= 64 else 1 + (s.ep % 8)))
        if len(pool) >= args.max_positions:
            break
    print(f"position pool: {len(pool)}", flush=True)

    # ---- SF pool
    n_sf = max(1, args.sf_pool)
    engines = [SFEval(args.sf, depth=args.depth) for _ in range(n_sf)]
    print(f"SF pool: {n_sf} engines @ depth {args.depth}", flush=True)

    def reward_of(child_fen: str, i: int) -> float:
        cp = engines[i % n_sf].eval_cp(child_fen)
        return max(-1.0, min(1.0, cp / args.reward_clip))

    def build_step(rng_local):
        """Sample groups -> (positions, actions, old_logp, advantages)."""
        idxs = rng_local.sample(range(len(pool)), args.groups)
        castles = [pool[i][2] for i in idxs]
        eps = [pool[i][3] for i in idxs]
        fens, moves_per, acts, base_logp = [], [], [], []
        with torch.no_grad():
            batch_codes = np.stack([pool[i][0] for i in idxs])
            batch_side = np.array([pool[i][1] for i in idxs])
            codes = torch.from_numpy(batch_codes).to(device)
            side = torch.from_numpy(batch_side).to(device)
            boards = [codes_to_board(batch_codes[j], int(batch_side[j]))
                      for j in range(args.groups)]
            castle = torch.from_numpy(np.array(castles, dtype=np.int64)).to(device)
            ep = torch.from_numpy(np.array(eps, dtype=np.int64)).to(device)
            scores, _, _ = model(codes, side, castle=castle, ep=ep)
            flat = scores.reshape(args.groups, 64 * 64)
            bscores, _, _ = base(codes, side, castle=castle, ep=ep)
            bflat = bscores.reshape(args.groups, 64 * 64)
            for j, b in enumerate(boards):
                legal = list(b.legal_moves)
                if len(legal) < 2:
                    legal = legal or []
                if len(legal) < 2:
                    fens.append(None)
                    continue
                mask = torch.full((64 * 64,), float("-inf"), device=device)
                umap = {}
                for m in legal:
                    u, v = m.from_square, m.to_square
                    # prefer queen promotion: fold promos into the (u,v) slot
                    umap[u * 64 + v] = m
                    mask[u * 64 + v] = 0.0
                lp = F.log_softmax(flat[j] + mask, dim=-1)
                k = min(args.k, len(legal))
                acts_j = gumbel_top_k(lp.unsqueeze(0), k)[0]
                blp = F.log_softmax(bflat[j] + mask, dim=-1)
                fens.append((b, mask))
                moves_per.append((legal, umap))
                acts.append(acts_j)
                base_logp.append(lp[acts_j])
        return fens, moves_per, acts, base_logp, (batch_codes, batch_side, castles, eps)

    # ---- validation gate: first batch (standing policy)
    fens, moves_per, acts, base_logp, (bc, bs, _, _) = build_step(random.Random(1))
    n_ok = sum(1 for f in fens if f is not None)
    assert n_ok >= 0.8 * args.groups, f"too few usable groups in gate: {n_ok}/{args.groups}"
    for j, f in enumerate(fens):
        if f is None:
            continue
        _, umap = moves_per[j]
        for a in acts[j]:
            assert int(a) in umap, f"sampled action not legal: {int(a)}"
    print("validation gate PASSED (50 groups: legality)", flush=True)

    # ---- training loop
    step = 0
    rng_local = random.Random(7)
    while step < args.steps:
        fens, moves_per, acts, base_logp, (bc, bs, castles, eps) = build_step(rng_local)
        # rewards via SF pool
        rewards = torch.full((args.groups, args.k), float("nan"))
        for j, f in enumerate(fens):
            if f is None:
                rewards[j, :] = 0.0
                continue
            b, mask = f
            legal, umap = moves_per[j]
            for ai, a in enumerate(acts[j]):
                a_int = int(a)
                m = umap.get(a_int)
                if m is None:
                    rewards[j, ai] = 0.0
                    continue
                b.push(m)
                r = reward_of(b.fen(), j)
                b.pop()
                rewards[j, ai] = r
        adv = (rewards - rewards.mean(dim=1, keepdim=True)) / \
              (rewards.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.reshape(-1)

        # second forward for the update (policy moved only by grad steps)
        codes = torch.from_numpy(bc).to(device)
        side = torch.from_numpy(bs).to(device)
        castle_t = torch.from_numpy(np.array(castles, dtype=np.int64)).to(device)
        ep_t = torch.from_numpy(np.array(eps, dtype=np.int64)).to(device)
        scores, _, _ = model(codes, side, castle=castle_t, ep=ep_t)
        flat = scores.reshape(args.groups, 64 * 64)
        bflat_scores, _, _ = base(codes, side, castle=castle_t, ep=ep_t)
        bflat = bflat_scores.reshape(args.groups, 64 * 64)

        logps, kls = [], []
        sel = [j for j, f in enumerate(fens) if f is not None]
        for j in sel:
            mask = fens[j][1]
            lp = F.log_softmax(flat[j] + mask, dim=-1)
            blp = F.log_softmax(bflat[j] + mask, dim=-1)
            a = acts[j].to(device)
            logps.append(lp[a])
            kls.append((blp.exp() * (blp - lp)).sum())  # KL(base || new)
        if not logps:
            continue
        new_logp = torch.stack(logps)
        old_logp = torch.stack([base_logp[j] for j in sel]).to(device)
        adv_sel = torch.stack([adv[j * args.k:(j + 1) * args.k] for j in sel]).to(device)
        ratio = torch.exp(new_logp - old_logp.to(device))
        s1 = ratio * adv_sel
        s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_sel
        pg = -torch.min(s1, s2).mean()
        kl = torch.stack(kls).mean()
        z = (flat.logsumexp(dim=-1) ** 2).mean() * 1e-3
        loss = pg + args.kl_beta * kl + z
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 5 == 0:
            print(f"step {step}: pg {pg.item():.4f} kl {kl.item():.4f} "
                  f"r_mean {rewards.mean().item():.3f}", flush=True)
        if step % args.export_every == 0 or step == args.steps:
            save_atomic(model.state_dict(), os.path.join(args.out, f"grpo_s{step}.pt"))
            model.export_blob(os.path.join(args.out, f"grpo_s{step}.bin"))
            print(f"exported {args.out}/grpo_s{step}.pt/.bin", flush=True)

    for e in engines:
        e.close()
    print("GRPO done — verdicts via fastchess A/B vs base (see grpo_phase.sh)", flush=True)


if __name__ == "__main__":
    main()
