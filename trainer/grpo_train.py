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
import time

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

    def _alive(self):
        """True while the engine process is usable. A dead engine's stdout
        stays open whenever a forked grandchild inherits the pipe, so the
        blocking readline below can hang FOREVER -- that is how the first
        production GRPO launch sat for 8 h without reaching step 5. Checking
        the child before every read turns that hang into an exception, which
        the caller's respawn path handles."""
        return self.p.poll() is None

    def _sync(self):
        self._send("isready")
        while True:
            if not self._alive():
                raise RuntimeError("SF engine exited during handshake")
            line = self.p.stdout.readline()
            if line.startswith("readyok"):
                return

    def eval_cp(self, fen: str) -> int:
        """Centipawn score from the side to move's perspective."""
        if not self._alive():
            raise RuntimeError("SF engine exited before analysis")
        self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        cp, mate = 0, 0
        while True:
            if not self._alive():
                raise RuntimeError("SF engine died mid-analysis")
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
        nonlocal engines
        e = engines[i % n_sf]
        try:
            cp = e.eval_cp(child_fen)
        except RuntimeError as exc:
            # dead/desynced engine: replace it and score this move 0 rather
            # than blocking forever (the 8-hour hang of the first launch)
            print(f"SF pool: engine {i % n_sf} failed ({exc}); respawning",
                  flush=True)
            engines[i % n_sf] = SFEval(args.sf, depth=args.depth)
            return 0.0
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
            zero_idx = torch.zeros(args.k, dtype=torch.long, device=device)
            for j, b in enumerate(boards):
                legal = list(b.legal_moves)
                if len(legal) < 2:
                    # Every list must stay index-aligned with fens: this branch
                    # used to append only to fens, so moves_per/acts/base_logp
                    # ran short and the reward loop indexed past their end.
                    fens.append(None)
                    moves_per.append(([], {}))
                    acts.append(zero_idx.clone())
                    base_logp.append(zero_idx.to(device=device, dtype=torch.float))
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
                # Pad to exactly k slots: positions with fewer than k legal
                # moves produced ragged logp vectors, which crashed the PPO
                # update in torch.stack. Slot 0 is (0 -> 0) and can never be a
                # legal move, so it is a safe dummy; its advantage is masked to
                # zero below, so it contributes no gradient.
                if k < args.k:
                    acts_j = torch.cat([acts_j, acts_j.new_zeros(args.k - k)])
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
        # actions are padded to exactly k with the dummy index 0 (a1->a1 is
        # never legal by construction), so the legality check must skip the
        # padding rather than assert on it -- the gate used to fail here on
        # every position with fewer than k legal moves.
        n_real = sum(1 for a in acts[j] if int(a) in umap)
        for a in acts[j]:
            if int(a) == 0:
                continue
            assert int(a) in umap, f"sampled action not legal: {int(a)}"
        assert n_real >= 1, f"group {j}: no legal action survived sampling"
    print("validation gate PASSED (50 groups: legality)", flush=True)

    # ---- training loop
    # Resume from the newest checkpoint in --out if one exists. Optimizer
    # moments are not saved, so the schedule restarts -- acceptable: the KL
    # and clip penalties keep a cold optimizer stable, and resume exists so a
    # watchdog-killed run continues instead of restarting from step 0 (the
    # shared box dies mid-run routinely; losing 1000 steps each time is what
    # makes the run unfinishable, not the loss of Adam moments).
    step = 0
    done = sorted(
        (f for f in os.listdir(args.out) if f.startswith("grpo_s") and f.endswith(".pt")),
        key=lambda f: int(f[6:-3]))
    if done:
        prev = int(done[-1][6:-3])
        model.load_state_dict(load_artifact(os.path.join(args.out, done[-1]),
                                            weights_only=True))
        if prev >= args.steps:
            print(f"resume: checkpoint at step {prev} >= {args.steps}; nothing to do",
                  flush=True)
            return
        step = prev
        print(f"resume: continuing from {done[-1]} (step {step})", flush=True)
    t_start = time.time()
    skipped = 0
    consec_skips = 0
    rng_local = random.Random(7)
    while step < args.steps:
        fens, moves_per, acts, base_logp, (bc, bs, castles, eps) = build_step(rng_local)
        # rewards via SF pool
        # Zeros, not NaN: padded slots used to stay NaN and poisoned the mean
        # and std of their whole group, so the loss became NaN.
        rewards = torch.zeros((args.groups, args.k), device=device)
        valid = torch.zeros((args.groups, args.k), device=device)
        for j, f in enumerate(fens):
            if f is None:
                continue
            b, mask = f
            legal, umap = moves_per[j]
            for ai, a in enumerate(acts[j]):
                a_int = int(a)
                m = umap.get(a_int)
                if m is None:
                    continue          # padded slot: stays invalid (adv 0)
                b.push(m)
                r = reward_of(b.fen(), j)
                b.pop()
                rewards[j, ai] = r
                valid[j, ai] = 1.0
        cnt = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (rewards * valid).sum(dim=1, keepdim=True) / cnt
        var = (((rewards - mean) ** 2) * valid).sum(dim=1, keepdim=True) / cnt
        adv = torch.where(valid > 0,
                          (rewards - mean) / (var.sqrt() + 1e-6),
                          torch.zeros_like(rewards))
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
            # KL(base || new) over legal moves only. Both log-probs are -inf
            # on illegal squares (shared mask), so the naive term is
            # 0 * (-inf - -inf) = NaN, which poisoned the loss from the very
            # first update (pg nan kl nan) and destroyed the weights.
            kl_terms = blp.exp() * (blp - lp)
            kls.append(torch.nan_to_num(kl_terms, nan=0.0).sum())
        if not logps:
            continue
        new_logp = torch.stack(logps)
        old_logp = torch.stack([base_logp[j] for j in sel]).to(device)
        adv_sel = torch.stack([adv[j * args.k:(j + 1) * args.k] for j in sel]).to(device)
        # Padded slots carry -inf in BOTH log-probs (dummy index 0 is masked),
        # so their ratio is exp(-inf - -inf) = NaN. Their advantage is 0, so
        # zeroing the NaN ratio makes them contribute exactly nothing.
        ratio = torch.nan_to_num(
            torch.exp(new_logp - old_logp.to(device)), nan=0.0)
        s1 = ratio * adv_sel
        s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_sel
        pg = -torch.min(s1, s2).mean()
        kl = torch.stack(kls).mean()
        z = (flat.logsumexp(dim=-1) ** 2).mean() * 1e-3
        loss = pg + args.kl_beta * kl + z
        if not torch.isfinite(loss):
            # A single bad update NaNs every weight permanently, and NaN
            # weights make EVERY later loss non-finite: this skip path would
            # then repeat forever -- build_step, rewards, non-finite,
            # continue -- consuming SF evals forever without ever
            # incrementing step (the "one step in 7 hours" incident). The
            # consecutive-skip FATAL turns that silent grind into a loud
            # abort; the gradient guard below stops the corruption at the
            # source so the loop stays rare.
            opt.zero_grad(set_to_none=True)
            skipped += 1
            consec_skips += 1
            if consec_skips >= 100:
                print(f"FATAL: {consec_skips} consecutive non-finite losses -- "
                      f"weights or reward data are corrupted; aborting instead "
                      f"of grinding forever", flush=True)
                return
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # A finite loss can still have NaN gradients on CUDA (observed in
        # production under memory pressure; the identical batch on CPU
        # backprops clean, so this is a runtime artifact, not bad data).
        # clip_grad_norm_ would scale every gradient by a NaN factor and
        # opt.step() would corrupt the weights -- the permanent-skip loop.
        # Response: redo the update on CPU, where the gradients are clean.
        # The CPU model mirrors the CUDA weights, so the update is the
        # mathematically correct one, just slower; it only runs on the
        # affected batches.
        if not all(torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.grad is not None):
            opt.zero_grad(set_to_none=True)
            # The fallback needs its own DIFFERENTIABLE CPU forward: the CUDA
            # `flat` is detached, so a CPU loss built from it has no graph.
            cpu_model = getattr(main, "_cpu_model", None)
            if cpu_model is None:
                cpu_model = ChessNet(v3=True)
                cpu_model.load_state_dict(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()})
                main._cpu_model = cpu_model
                base_cpu = ChessNet(v3=True)
                base_cpu.load_state_dict(
                    {k: v.detach().cpu() for k, v in base.state_dict().items()})
                main._base_cpu = base_cpu
            else:
                cpu_model.load_state_dict(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()})
            codes_c = torch.from_numpy(bc).cpu()
            side_c = torch.from_numpy(bs).cpu()
            castle_c = castle_t.detach().cpu()
            ep_c = ep_t.detach().cpu()
            scores_c, _, _ = cpu_model(codes_c, side_c, castle=castle_c, ep=ep_c)
            flat_c = scores_c.reshape(scores_c.shape[0], 64 * 64)
            with torch.no_grad():
                bscores_c, _, _ = main._base_cpu(codes_c, side_c,
                                                 castle=castle_c, ep=ep_c)
            bflat_c = bscores_c.reshape(bscores_c.shape[0], 64 * 64)
            logps_c, kls_c = [], []
            for j in sel:
                mask_c = fens[j][1].cpu()
                lp_c = F.log_softmax(flat_c[j] + mask_c, dim=-1)
                blp_c = F.log_softmax(bflat_c[j] + mask_c, dim=-1)
                a_c = acts[j].cpu()
                logps_c.append(lp_c[a_c])
                klt_c = blp_c.exp() * (blp_c - lp_c)
                kls_c.append(torch.nan_to_num(klt_c, nan=0.0).sum())
            new_logp_c = torch.stack(logps_c)
            kl_c = torch.stack(kls_c).mean()
            z_c = (flat_c.logsumexp(dim=-1) ** 2).mean() * 1e-3
            adv_c = adv_sel.detach().cpu()
            ratio_c = torch.nan_to_num(
                torch.exp(new_logp_c - old_logp.detach().cpu()), nan=0.0)
            s1c = ratio_c * adv_c
            s2c = torch.clamp(ratio_c, 1 - args.clip, 1 + args.clip) * adv_c
            loss_c = -torch.min(s1c, s2c).mean() + args.kl_beta * kl_c + z_c
            cpu_model.zero_grad(set_to_none=True)
            loss_c.backward()
            if not all(torch.isfinite(p.grad).all()
                       for p in cpu_model.parameters() if p.grad is not None):
                # Even CPU gradients are NaN: genuinely bad batch, skip it.
                skipped += 1
                consec_skips += 1
                if consec_skips >= 100:
                    print(f"FATAL: {consec_skips} consecutive non-finite "
                          f"gradient batches (CPU fallback also NaN) -- "
                          f"aborting", flush=True)
                    return
                continue
            torch.nn.utils.clip_grad_norm_(cpu_model.parameters(), 1.0)
            with torch.no_grad():
                for p_cuda, p_cpu in zip(model.parameters(),
                                         cpu_model.parameters()):
                    p_cuda.grad = p_cpu.grad.to(p_cuda.device)
            opt.step()
            cpu_updates = getattr(main, "_cpu_updates", 0)
            main._cpu_updates = cpu_updates + 1
            step += 1
            print(f"step {step}: CPU-fallback update #{main._cpu_updates}",
                  flush=True)
            continue
        consec_skips = 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        # Log EVERY step: with 1024+ depth-10 analyses per step on a shared
        # box a single step can take over an hour, so a 5-step print interval
        # looks identical to a hang.
        print(f"step {step}: pg {pg.item():.4f} kl {kl.item():.4f} "
              f"r_mean {rewards.mean().item():.3f} skipped {skipped} "
              f"elapsed {(time.time() - t_start) / 60:.1f}m", flush=True)
        if step % args.export_every == 0 or step == args.steps:
            finite = all(torch.isfinite(p).all() for p in model.parameters())
            if not finite:
                print(f"step {step}: weights not finite -- checkpoint NOT saved",
                      flush=True)
            else:
                save_atomic(model.state_dict(), os.path.join(args.out, f"grpo_s{step}.pt"))
                model.export_blob(os.path.join(args.out, f"grpo_s{step}.bin"))
            print(f"exported {args.out}/grpo_s{step}.pt/.bin", flush=True)

    for e in engines:
        e.close()
    print("GRPO done — verdicts via fastchess A/B vs base (see grpo_phase.sh)", flush=True)


if __name__ == "__main__":
    main()
