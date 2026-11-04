"""BC / distillation training for ChessNet on shard files.

Usage:
  python3 train.py --shard shards/bc.shard --out runs/v0 [--epochs 3] ...

Loss: masked cross-entropy over legal-move policy scores. The legal mask is
generated per sample from the shard's board codes with python-chess (a GPL
tool used purely as a training-time utility — its code never enters the
engine or the exported weights).
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import random

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import MaskSidecar, count_records, iter_records  # noqa: E402
from model import ChessNet, load_v1_into_v3  # noqa: E402
from robust_io import load_artifact, save_atomic  # atomic + self-healing artifacts

PIECE_TO_PC = {1: ("P", 0), 2: ("N", 0), 3: ("B", 0), 4: ("R", 0), 5: ("Q", 0), 6: ("K", 0),
               9: ("P", 1), 10: ("N", 1), 11: ("B", 1), 12: ("R", 1), 13: ("Q", 1), 14: ("K", 1)}
PROMO_ROLE = {0: "n", 1: "b", 2: "r", 3: "q"}


def codes_to_board(codes: np.ndarray, side: int):
    import chess  # python-chess: GPL tool, training-time only
    board = chess.Board(None)
    for sq, code in enumerate(codes):
        if code in PIECE_TO_PC:
            letter, color = PIECE_TO_PC[code]
            piece = chess.Piece.from_symbol(letter if color == 0 else letter.lower())
            board.set_piece_at(sq, piece)
    board.turn = chess.WHITE if side == 0 else chess.BLACK
    return board


def legal_mask_and_index(board, targets):
    """(64,64) bool mask over scores[from*64+to]; flat target index."""
    mask = np.zeros((64, 64), dtype=np.bool_)
    target_flat = -1
    for mv in board.legal_moves:
        u, v = mv.from_square, mv.to_square
        mask[u, v] = True  # scores[from][to]
        if targets and (u, v) == (targets[0][0], targets[0][1]):
            target_flat = u * 64 + v
    return mask, target_flat


def encode_batch(samples, device, state_aware: bool = False):
    codes = torch.from_numpy(np.stack([s.board_codes for s in samples])).long().to(device)
    side = torch.tensor([s.side for s in samples], dtype=torch.long, device=device)
    if not state_aware:
        return codes, side
    # v3 index conventions: castle = the raw 4-bit mask; ep = file + 1, with 0
    # meaning "no ep square" (the shard stores a square index, 255 = none).
    castle = torch.tensor([int(s.castling) & 15 for s in samples],
                          dtype=torch.long, device=device)
    ep = torch.tensor([0 if int(s.ep) == 255 else (int(s.ep) % 8) + 1
                       for s in samples], dtype=torch.long, device=device)
    return codes, side, castle, ep


def build_batch(stream, batch: int, device, require_targets: bool, mask_sc: MaskSidecar | None = None,
                state_aware: bool = False, volatility_weight: float = 1.0,
                volatility_threshold: int = 150,
                quality_filter: bool = False, tb_labels: dict | None = None,
                soft_labels: dict | None = None):
    """Pull one batch worth of samples with their masks.

    With a MaskSidecar the mask comes from precomputed legal-move indices
    (fast path); otherwise python-chess computes it per sample (slow path).
    With tb_labels (record idx -> (wdl -2..2, best_flat_idx), from
    make_tb_labels.py) covered records get exact Syzygy targets: the policy
    target becomes the DTZ-optimal move and the value target the exact WDL,
    both at full policy weight (RESEARCH-ENDGAME-RL E1)."""
    samples, masks, tidx, evals, labeled = [], [], [], [], []
    pol_w, tb_exact, tb_vt, fmoves, ssoft = [], [], [], [], []
    while len(samples) < batch:
        item = next(stream, None)
        if item is None:
            return None
        idx_s, s = item
        if require_targets and not s.targets:
            continue
        if mask_sc is not None and idx_s >= mask_sc.count:
            return None
        if quality_filter and len(s.targets) >= 2 and abs(s.eval_cp) > 300:
            continue
        is_tb = tb_labels is not None and idx_s in tb_labels
        s_ent = (soft_labels.get(idx_s) if soft_labels is not None else None) or None
        if is_tb:
            wdl, best_flat = tb_labels[idx_s]
            if wdl == 2:   vt = (1.0, 0.0, 0.0)
            elif wdl == 1: vt = (0.75, 0.25, 0.0)
            elif wdl == 0: vt = (0.0, 1.0, 0.0)
            elif wdl == -1: vt = (0.0, 0.25, 0.75)
            else:          vt = (0.0, 0.0, 1.0)
            pw = 1.0  # exact label: full weight
        else:
            vt = None
            pw = 1.0

        if mask_sc is not None:
            flat = mask_sc.mask_indices(idx_s)
            if len(flat) == 0:
                continue  # terminal position
            mask = np.zeros((64, 64), dtype=np.bool_)
            mask[flat >> 6, flat & 63] = True
            tflat = int(s.targets[0][0]) * 64 + int(s.targets[0][1])
            if not mask.reshape(-1)[tflat]:
                continue  # target not legal: corrupt record
            if is_tb and mask.reshape(-1)[best_flat]:
                tflat = best_flat  # DTZ-optimal policy target
            if s_ent:
                s_ent = [(mv, p) for mv, p in s_ent if mask.reshape(-1)[mv]]
                if s_ent:
                    tflat = max(s_ent, key=lambda x: x[1])[0]  # soft argmax
                else:
                    s_ent = None
        else:
            board = codes_to_board(s.board_codes, s.side)
            if board.is_game_over():
                continue
            mask, tflat = legal_mask_and_index(board, s.targets)
            if tflat < 0:
                continue
            if is_tb and mask.reshape(-1)[best_flat]:
                tflat = best_flat
            if s_ent:
                s_ent = [(mv, p) for mv, p in s_ent if mask.reshape(-1)[mv]]
                if s_ent:
                    tflat = max(s_ent, key=lambda x: x[1])[0]  # soft argmax
                else:
                    s_ent = None

        samples.append(s)
        masks.append(mask)
        tidx.append(tflat)
        evals.append(s.eval_cp)
        labeled.append(len(s.targets) >= 2)
        pol_w.append(pw)
        tb_exact.append(is_tb)
        tb_vt.append(vt if vt is not None else (0.0, 0.0, 0.0))
        fmoves.append(float(s.fullmove))
        ssoft.append(s_ent)
    encoded = encode_batch(samples, device, state_aware=state_aware)
    codes, side = encoded[0], encoded[1]
    state_kw = {}
    if state_aware:
        state_kw = {"castle": encoded[2], "ep": encoded[3]}
    # Volatility: |eval delta| vs the previous ply. Records are in game order,
    # so consecutive records are usually consecutive plies; at a game boundary
    # the delta is meaningless (affecting ~the number of games / records, a few
    # percent) — acceptable noise for a loss weight.
    volat = None
    if volatility_weight != 1.0:
        evs = [float(s_.eval_cp) for s_ in samples]
        sides = [int(s_.side) for s_ in samples]
        flags = [0.0]
        for i in range(1, len(samples)):
            if sides[i] != sides[i - 1]:
                flags.append(1.0 if abs(evs[i] - evs[i - 1]) >= volatility_threshold else 0.0)
            else:
                flags.append(0.0)   # boundary: side must alternate in a real game
        volat = flags
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    tgt = torch.tensor(tidx, dtype=torch.long, device=device)
    res_t = torch.from_numpy(np.stack([s.wdl for s in samples]).astype(np.float32)).to(device)
    tb = dict(
        pol_w=torch.tensor(pol_w, dtype=torch.float32, device=device),
        exact=torch.tensor(tb_exact, dtype=torch.bool, device=device),
        vt=torch.tensor(np.stack(tb_vt).astype(np.float32), device=device),
        fm=torch.tensor(fmoves, dtype=torch.float32, device=device),
    )
    # order must match the unpacking at the call site
    volat_t = (torch.tensor(volat, dtype=torch.float32, device=device)
               if volat is not None else torch.zeros(0, device=device))
    soft_t = None
    if soft_labels is not None and any(ssoft):
        B = len(samples)
        soft_t = torch.zeros(B, 64 * 64, device=device)
        for r, ent in enumerate(ssoft):
            if ent:
                mi = torch.tensor([m for m, _ in ent], dtype=torch.long, device=device)
                mp = torch.tensor([p for _, p in ent], dtype=torch.float32, device=device)
                soft_t[r].scatter_(0, mi, mp)
    return codes, side, mask_t, tgt, \
        torch.tensor(evals, dtype=torch.float32, device=device), \
        torch.tensor(labeled, dtype=torch.bool, device=device), \
        state_kw, volat_t, res_t, tb, soft_t


def ensure_mask(args) -> MaskSidecar | None:
    """Auto-generate a legal-move sidecar via the lo-data binary when absent.

    Generation takes minutes and makes training GPU-bound instead of
    python-chess-bound (the fallback stays available via --no-mask-auto)."""
    if args.no_mask_auto:
        return None
    import subprocess
    mask_path = args.shard + ".mask"
    total = count_records(args.shard)
    if os.path.exists(mask_path) and os.path.getsize(mask_path) > 1000:
        sc = MaskSidecar(mask_path)
        # integrity: the offset table must end inside the file. A mask
        # truncated mid-generation still carries a full-count header, so
        # count alone is not proof (2026-11-04 compound38m crash loop).
        base = 16 + (sc.count + 1) * 8
        last_off = int(sc.offsets[-1])
        sc.close()
        if last_off + base <= os.path.getsize(mask_path):
            sc = MaskSidecar(mask_path)
            if sc.count >= total:
                print(f"mask sidecar: {mask_path} ({sc.count} records)", flush=True)
                return sc
            sc.close()
            print(f"mask sidecar: {mask_path} only covers {sc.count}/{total} — regenerating", flush=True)
        else:
            print(f"mask sidecar: {mask_path} truncated (last offset {last_off}+{base} > "
                  f"{os.path.getsize(mask_path)}) — regenerating", flush=True)
    bin_ = os.environ.get("LO_LODATA_BIN", "/home/wyatt/tools/chess/lo-data")
    if not os.path.exists(bin_):
        print("mask sidecar: lo-data binary not found — python-chess fallback", flush=True)
        return None
    print(f"mask sidecar: generating {mask_path} for {total} records...", flush=True)
    r = subprocess.run([bin_, "masks", "--in", args.shard, "--out", mask_path],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(mask_path) or count_records(mask_path) < total:
        print("mask sidecar: generation failed — python-chess fallback", flush=True)
        return None
    print("mask sidecar: ready", flush=True)
    return MaskSidecar(mask_path)



class _ParamGroupSched:
    """Minimal stand-in for LambdaLR for optimizers that are not
    torch.optim.Optimizer subclasses (the joint Muon/AdamW wrapper).

    Mirrors LambdaLR semantics: lr = initial_lr * fn(step), applied once at
    construction and again on every step().
    """

    def __init__(self, opt, fn):
        self.opt = opt
        self.fn = fn
        self.base = [g.get("initial_lr", g["lr"]) for g in opt.param_groups]
        self.t = 0
        self._apply()          # LambdaLR applies fn(0) at construction

    def _apply(self):
        factor = self.fn(self.t)
        for group, base in zip(self.opt.param_groups, self.base):
            group["lr"] = base * factor

    def step(self):
        self.t += 1
        self._apply()

    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dff", type=int, default=1024)
    ap.add_argument("--dpol", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0, help="cap steps per epoch (0 = all)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--no-mask-auto", action="store_true")
    ap.add_argument("--fp32", action="store_true",
                    help="disable fp16 autocast (nan-safe)")
    ap.add_argument("--quality-filter", action="store_true",
                    help="Skip records where played move loses >300cp vs SF best")
    ap.add_argument("--decisive-weighting", action="store_true",
                    help="Lc0-style policy weighting: labeled positions with small |eval| "
                         "get reduced policy loss weight (best move is near-arbitrary "
                         "in equal positions); full weight at |eval| >= 150cp")
    ap.add_argument("--init-from", default="", help="Load pre-trained .pt checkpoint for fine-tuning")
    ap.add_argument("--lr-ft", type=float, default=1e-4, help="Fine-tuning learning rate")
    ap.add_argument("--gab", action="store_true",
                    help="Geometric Attention Bias (Chessformer GAB-lite) [see --v3 note]")
    ap.add_argument("--volatility-weight", type=float, default=1.0,
                    help="upweight positions whose eval swung vs the previous ply "
                         "(weight W when |delta| >= threshold, else 1.0). W=1.0 "
                         "disables. Targets the measured bottleneck: losses are "
                         "dominated by tactical collapses (RESULTS.md, loss "
                         "attribution). Only meaningful with --decisive-weighting.")
    ap.add_argument("--volatility-threshold", type=int, default=150,
                    help="eval delta (cp) above which a position counts as volatile")
    ap.add_argument("--prefetch", action="store_true", default=True,
                    help="build the next batch on a worker thread while the GPU "
                         "trains on the current one (default on)")
    ap.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                    help="disable batch prefetch")
    ap.add_argument("--state-aware", action="store_true",
                    help="feed the v3 castle/ep embeddings with the record's real "
                         "values. Without this the model is called as model(codes, "
                         "side), so castle/ep/rating stay pinned at index 0 and "
                         "their other rows never receive gradient: the fields are "
                         "silently inert (verified on a real run: 256/4096 nonzero "
                         "= exactly one trained row). En-passant rights are the "
                         "genuinely new information here -- the board cannot tell "
                         "you a pawn just moved two squares.")
    ap.add_argument("--v3", action="store_true",
                    help="v3 architecture (SPEC-BLOB-V3.md): castle/ep/king/rating inputs, "
                         "HiCo history, material-bucketed value head. Implies --gab layout. "
                         "With --init-from on a v1/v2 checkpoint, warm-starts via "
                         "load_v1_into_v3 (bit-identical trunk, tiled value head).")
    ap.add_argument("--gab-legacy", action="store_true",
                    help="Geometric Attention Bias (Chessformer GAB-lite): learned per-head "
                         "bias over square-relation buckets. Zero-init = v1 model exactly, "
                         "so --init-from v1 checkpoints warm-start losslessly. Exports blob v2.")
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                    help="muon: orthogonalized momentum on 2-D hidden weights + AdamW on "
                         "embeddings/heads (Muonorger paper recipe). lr applies to AdamW "
                         "params; Muon lr = lr*66 approx via --muon-lr override.")
    ap.add_argument("--muon-lr", type=float, default=0.02, help="Muon learning rate")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA shadow-weights decay (0 = off). Exports net_eN_ema.bin; "
                         "Lc0-style: EMA net is usually the stronger player.")
    ap.add_argument("--recycle", type=int, default=1,
                    help="Training-time recycling passes (RESEARCH-NOVEL N2). >1 adds "
                         "pass-consistency loss (RCT): earlier passes are pulled toward "
                         "the stop-grad final pass. 2x step cost at R=2.")
    ap.add_argument("--rct-lambda", type=float, default=0.5,
                    help="RCT consistency loss weight (used when --recycle > 1).")
    ap.add_argument("--sched", choices=["const", "cosine", "wsd"], default="const",
                    help="LR schedule: const (legacy default), cosine->10%%, or WSD "
                         "(warmup-stable-decay: stable until 90%% then linear to 10%%).")
    ap.add_argument("--opening-weight", type=float, default=1.0,
                    help="Loss weight multiplier for fullmove<=12 records (opening-phase "
                         "upweighting, RESEARCH-ROUND9 A3). 1.0 = off.")
    ap.add_argument("--tb-sidecar", default="",
                    help="JSONL from make_tb_labels.py: exact Syzygy WDL/DTZ labels for "
                         "<=5-piece records. TB records get exact value targets, "
                         "DTZ-optimal policy targets, full policy weight (E1).")
    ap.add_argument("--soft-labels", default="",
                    help="JSONL from make_soft_policy.py: SF multipv soft policy "
                         "targets {shard_idx, soft:[[uci,p]...]}. Covered rows get "
                         "CE against the soft distribution; others keep hard one-hot.")
    ap.add_argument("--qat", action="store_true",
                    help="projection QAT (quantized-projected SGD): after each step, "
                         "project the quantized-linears' weights onto the s8 grid "
                         "(per-tensor symmetric round-to-nearest). Removes most of the "
                         "INT8 export drop without STE machinery.")
    ap.add_argument("--mirror", action="store_true",
                    help="File-mirror augmentation (a<->h), prob 0.5 per batch: the only "
                         "legal chess symmetry without a color swap. Free 2x data.")
    ap.add_argument("--distill1ply", type=int, default=0,
                    help="positions per step for one-ply value distillation "
                         "(0=off): enumerates legal afterstates, evaluates them "
                         "with the same net, and pulls v(s) toward max_a(1-v_opp) "
                         "— distills one-ply search into the value head.")
    ap.add_argument("--distill-weight", type=float, default=0.1,
                    help="loss weight for the one-ply distillation term")
    ap.add_argument("--rank-mirror", action="store_true",
                    help="Rank+color mirror augmentation, prob 0.5 per batch: the true "
                         "chess Z2 symmetry (side swaps, ep/castling remap, wdl swap). "
                         "Free 2x data + enforces v(b)=1-v(mirror(b)).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = ChessNet(args.d, args.layers, args.heads, args.dff, args.dpol,
                     gab=args.gab or args.v3, v3=args.v3).to(device)
    if args.init_from and os.path.exists(args.init_from):
        sd = load_artifact(args.init_from, weights_only=True)
        missing = None
        if args.v3 and "V2.weight" in sd and sd["V2.weight"].shape[0] == 3:
            load_v1_into_v3(model, sd)
            print(f"loaded pre-trained weights from {args.init_from} (v1->v3 warm-start)", flush=True)
        else:
            missing = model.load_state_dict(sd, strict=False)
        if missing is not None and args.gab and all("gab_table" in k for k in missing.unexpected_keys) and not missing.missing_keys:
            print(f"loaded pre-trained weights from {args.init_from} (GAB table zero-init)", flush=True)
        elif missing is not None and not missing.missing_keys and not missing.unexpected_keys:
            print(f"loaded pre-trained weights from {args.init_from}", flush=True)
        elif missing is not None:
            print(f"loaded pre-trained weights from {args.init_from} "
                  f"(missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)})", flush=True)
    from muon import Muon, split_params_for_muon
    lr = args.lr_ft if args.init_from else args.lr
    if args.optimizer == "muon":
        muon_p, adamw_p = split_params_for_muon(model)
        adamw_opt = torch.optim.AdamW(adamw_p, lr=lr, weight_decay=args.wd) if adamw_p else None
        muon_opt = Muon(muon_p, lr=args.muon_lr) if muon_p else None

        class _JointOpt:
            def __init__(self, a, b):
                self.a, self.b = a, b
                self.param_groups = (a.param_groups if a else []) + \
                                    (b.param_groups if b else [])
            def zero_grad(self, set_to_none=True):
                for o in (self.a, self.b):
                    if o: o.zero_grad(set_to_none=set_to_none)
            def step(self):
                if self.b: self.b.step()
                if self.a: self.a.step()
        opt = _JointOpt(adamw_opt, muon_opt)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M on {device} "
          f"optimizer={args.optimizer}{' gab' if args.gab else ''}", flush=True)
    total_steps = None  # set after shard count known (below)
    sched_kind = args.sched

    ema_params = ([p.detach().clone() for p in model.parameters()]
                  if args.ema_decay > 0 else None)

    total = count_records(args.shard)
    os.makedirs(args.out, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    mask_sc = ensure_mask(args)
    tb_labels = None
    if args.tb_sidecar and os.path.exists(args.tb_sidecar):
        import json
        tb_labels = {}
        with open(args.tb_sidecar) as f:
            for line in f:
                r = json.loads(line)
                ff = "abcdefgh".index(r["best"][0]); fr = int(r["best"][1]) - 1
                tf = "abcdefgh".index(r["best"][2]); tr = int(r["best"][3]) - 1
                from_sq = fr * 8 + ff   # python-chess square = rank*8 + file
                to_sq = tr * 8 + tf
                tb_labels[r["idx"]] = (int(r["wdl"]), from_sq * 64 + to_sq)
        print(f"tb sidecar: {len(tb_labels)} exact endgame labels", flush=True)
    soft_labels = None
    if args.soft_labels and os.path.exists(args.soft_labels):
        import json
        import chess
        soft_labels = {}
        with open(args.soft_labels) as f:
            for line in f:
                r = json.loads(line)
                ent = []
                for uci, p in r["soft"]:
                    mv = chess.Move.from_uci(uci)
                    ent.append((mv.from_square * 64 + mv.to_square, float(p)))
                if ent:
                    soft_labels[int(r["shard_idx"])] = ent
        print(f"soft labels: {len(soft_labels)} rows from {args.soft_labels}", flush=True)
    match_ema = None
    if sched_kind != "const":
        steps_per_epoch = max(1, total // args.batch)
        total_steps = steps_per_epoch * args.epochs
        warm = max(1, int(0.05 * total_steps))
        if sched_kind == "cosine":
            lambda_lr = lambda st: min(1.0, st / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, st / max(1, total_steps)))))
        else:  # wsd: stable, then linear to 10%
            decay_start = int(0.9 * total_steps)
            lambda_lr = lambda st: min(1.0, st / warm) * (
                1.0 if st < decay_start else max(0.1, 1.0 - 0.9 * (st - decay_start) / max(1, total_steps - decay_start)))
        # The Muon path wraps AdamW+Muon in _JointOpt, which is not a
        # torch.optim.Optimizer, so LambdaLR rejects it. The unguarded call
        # that used to sit here raised TypeError and killed the AV stage-1 run
        # at startup -- after 40 h of labeling. Use a param-group shim so the
        # schedule is actually applied instead of silently skipped.
        scheduler = None
        if not isinstance(opt, dict):
            try:
                scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_lr)
            except TypeError:
                scheduler = _ParamGroupSched(opt, lambda_lr)
                print("scheduler: param-group shim (joint Muon/AdamW wrapper)",
                      flush=True)
    # Batch prefetch. Building a batch is python-side work (record decode,
    # legal-mask lookup, tensor assembly) and the GPU sat at 36% utilization
    # waiting for it on the 130M-position run -- i.e. two thirds of the step
    # time was the data pipeline idling. One worker thread builds the next
    # batch while the current one trains. Only one thread ever touches the
    # stream: the worker is a single-slot pool and we submit exactly one batch
    # ahead, so ordering is preserved.
    pool = None
    if args.prefetch:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1)

    def _make_batch(stream_):
        return build_batch(stream_, args.batch, device, require_targets=True,
                           mask_sc=mask_sc, quality_filter=args.quality_filter,
                           tb_labels=tb_labels, state_aware=args.state_aware,
                           volatility_weight=args.volatility_weight,
                           volatility_threshold=args.volatility_threshold,
                           soft_labels=soft_labels)

    for epoch in range(args.epochs):
        stream = enumerate(iter_records(args.shard))
        done = 0
        pending = pool.submit(_make_batch, stream) if pool is not None else None
        while True:
            batch = pending.result() if pending is not None else _make_batch(stream)
            if batch is None:
                break
            pending = pool.submit(_make_batch, stream) if pool is not None else None
            codes, side, mask, tgt, evals, labeled, state_kw, volat, res_t, tb, soft_t = batch
            if args.mirror and torch.rand(1).item() < 0.5:
                from model import MIRROR_IDX
                codes = codes.view(-1, 64)[:, MIRROR_IDX.to(codes.device)]
                mask = mask[:, MIRROR_IDX.to(mask.device), :][:, :, MIRROR_IDX.to(mask.device)]
                u, v = tgt // 64, tgt % 64
                tgt = MIRROR_IDX.to(tgt.device)[u] * 64 + MIRROR_IDX.to(tgt.device)[v]
                if soft_t is not None:
                    soft_t = soft_t.view(-1, 64, 64)[:, MIRROR_IDX.to(soft_t.device), :] \
                        [:, :, MIRROR_IDX.to(soft_t.device)].reshape(-1, 64 * 64)
            if args.rank_mirror and torch.rand(1).item() < 0.5:
                # rank+color mirror: sq -> sq^56, color swap (1-6 <-> 9-14),
                # side flip, wdl triple reversed (white-POV swap), eval flip.
                mir = torch.arange(64, device=codes.device) ^ 56
                cs = codes.view(-1, 64)[:, mir]
                code_swapped = torch.where((cs >= 1) & (cs <= 14), cs ^ 8, cs)
                codes = code_swapped
                side = 1 - side
                mask = mask[:, mir, :][:, :, mir]
                u, v = tgt // 64, tgt % 64
                tgt = (u ^ 56) * 64 + (v ^ 56)
                res_t = res_t[:, [2, 1, 0]]
                if evals.numel(): evals = -evals
                if soft_t is not None:
                    soft_t = soft_t.view(-1, 64, 64)[:, mir][:, :, mir].reshape(-1, 64 * 64)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda" and not args.fp32):
                if args.recycle > 1:
                    passes = model.forward_recycle(codes, side, R=args.recycle, **state_kw)
                    scores, promo, wdl = passes[-1]
                    # RCT: pull earlier passes toward the stop-grad final policy
                    rct = scores.new_zeros(())
                    with torch.no_grad():
                        final_logp = F.log_softmax(
                            passes[-1][0].reshape(codes.shape[0], -1), dim=-1)
                    for ps, _, _ in passes[:-1]:
                        pass_logp = F.log_softmax(ps.reshape(codes.shape[0], -1), dim=-1)
                        rct = rct + F.kl_div(pass_logp, final_logp,
                                             log_target=True, reduction="batchmean")
                    rct = rct / (len(passes) - 1)
                else:
                    scores, promo, wdl = model(codes, side, **state_kw)
                    rct = scores.new_zeros(())
                flat = scores.reshape(codes.shape[0], 64 * 64)
                flat = flat.masked_fill(~mask.reshape(codes.shape[0], -1), float("-inf"))
                pl_raw = F.cross_entropy(flat, tgt, reduction="none")
                if soft_t is not None:
                    # Soft SF-policy rows: CE against the teacher's move
                    # distribution; uncovered rows keep the hard one-hot.
                    soft_t = soft_t * mask.reshape(codes.shape[0], -1).float()
                    has_soft = soft_t.sum(1) > 0
                    if has_soft.any():
                        # mask logp to 0 on illegal entries: soft_t is 0 there,
                        # and 0 * (-inf) is NaN, not 0 (IEEE)
                        logp = F.log_softmax(flat, dim=1).masked_fill(
                            ~mask.reshape(codes.shape[0], -1), 0.0)
                        soft_ce = -(soft_t * logp).sum(1)
                        pl_raw = torch.where(has_soft, soft_ce, pl_raw)
                with torch.no_grad():
                    batch_match = (flat.argmax(1) == tgt).float().mean().item()
                match_ema = batch_match if match_ema is None else 0.98 * match_ema + 0.02 * batch_match
                if args.decisive_weighting:
                    # Lc0-style: weight policy loss by position decisiveness.
                    # Equal positions (|eval| < 150cp) contribute proportionally
                    # less; BC records keep full weight.
                    pw_weight = torch.where(
                        labeled,
                        torch.clamp(evals.abs() / 150.0, max=1.0),
                        torch.ones_like(evals),
                    )
                    if args.opening_weight != 1.0:
                        ow = torch.where(tb["fm"] <= 12,
                                         torch.full_like(tb["fm"], args.opening_weight),
                                         torch.ones_like(tb["fm"]))
                        pw_weight = pw_weight * ow
                    if tb_labels:
                        pw_weight = torch.where(tb["exact"], tb["pol_w"], pw_weight)
                    if volat.numel():
                        pw_weight = pw_weight * torch.where(
                            volat > 0,
                            torch.full_like(volat, args.volatility_weight),
                            torch.ones_like(volat))
                    pl = (pl_raw * pw_weight).sum() / pw_weight.sum().clamp(min=1.0)
                else:
                    pl = pl_raw.mean()
                # Value target: game results for BC records; for SF-labeled
                # records, a soft WDL distribution from eval_cp via the
                # classic logistic model (k = 0.00368208), which carries the
                # teacher's judgment instead of noisy single-game outcomes.
                k = 0.00368208
                pw = torch.sigmoid(k * evals)
                pd_ = torch.clamp(1.0 - pw - torch.sigmoid(-k * evals), min=0.0)
                pl_loss = torch.sigmoid(-k * evals)
                soft = torch.stack([pw, pd_, pl_loss], dim=1)
                soft = soft / soft.sum(dim=1, keepdim=True)
                soft_t = torch.where(labeled.unsqueeze(1), soft, res_t)
                if tb_labels:
                    # exact Syzygy value targets override everything else
                    soft_t = torch.where(tb["exact"].unsqueeze(1), tb["vt"], soft_t)
                vl = F.cross_entropy(wdl.float(), soft_t)
                distill_loss = scores.new_zeros(())
                loss = pl + 0.5 * vl + args.rct_lambda * rct
                if args.distill1ply:
                    # Loop B: one-ply value distillation (RESEARCH-CROSSFIELD §2)
                    sel = random.sample(range(codes.shape[0]),
                                        min(args.distill1ply, codes.shape[0]))
                    aft_codes, aft_sides, owner = [], [], []
                    mir = torch.arange(64, device=codes.device) ^ 56
                    for b_i in sel:
                        legal = (mask[b_i].reshape(-1) != 0).nonzero().flatten()
                        if legal.numel() == 0:
                            continue
                        frs, tos = legal // 64, legal % 64
                        bd = codes[b_i]
                        piece = bd[frs]
                        empty = torch.zeros_like(piece)
                        mover_white = (side[b_i].item() == 0)
                        pawn = piece == (1 if mover_white else 9)
                        to_occ = bd[tos] != 0
                        keep = ~(pawn & (~to_occ) & ((frs % 8) != (tos % 8)))
                        frs, tos, piece, empty = frs[keep], tos[keep], piece[keep], empty[keep]
                        if frs.numel() == 0:
                            continue
                        aft = bd.clone()
                        aft[tos] = piece
                        aft[frs] = empty
                        is_castle = ((piece == (6 if mover_white else 14))
                                     & ((frs % 8 - tos % 8).abs() == 2))
                        for ci in is_castle.nonzero().flatten().tolist():
                            fr_, to_ = int(frs[ci]), int(tos[ci])
                            rk_white = mover_white
                            r_from = (7 if to_ % 8 == 6 else 0) + (0 if rk_white else 56)
                            r_to = (5 if to_ % 8 == 6 else 3) + (0 if rk_white else 56)
                            aft[r_to] = 4 + (0 if rk_white else 8)
                            aft[r_from] = 0
                        aft_codes.append(aft)
                        aft_sides.append(torch.tensor([1 - side[b_i].item()],
                                                      dtype=torch.long, device=codes.device))
                        owner.append(torch.tensor([b_i], dtype=torch.long,
                                                  device=codes.device))
                    if aft_codes:
                        ac = torch.cat(aft_codes).view(-1, 64)
                        asd = torch.cat(aft_sides).view(-1)
                        owner = torch.cat(owner).view(-1)
                        _, _, w_opp = model(ac, asd, **state_kw)
                        p_opp = torch.softmax(w_opp.float(), dim=1)
                        v_opp = p_opp[:, 0] + 0.5 * p_opp[:, 1]
                        m_target = torch.zeros(codes.shape[0], device=codes.device)
                        m_target = m_target.scatter_reduce(0, owner, 1.0 - v_opp,
                                                           reduce="amax", include_self=False)
                        wl_all = torch.softmax(wdl.float(), dim=1)
                        v_all = wl_all[:, 0] + 0.5 * wl_all[:, 1]
                        sel_t = torch.tensor(sel, device=codes.device)
                        distill_loss = args.distill_weight * F.mse_loss(
                            v_all[sel_t], m_target[sel_t].detach())
                        loss = loss + distill_loss + distill_loss
            if not torch.isfinite(loss):
                main.consec_bad = getattr(main, "consec_bad", 0) + 1
                print(f"non-finite loss at step {step} (consecutive {main.consec_bad})", flush=True)
                if main.consec_bad >= 50:
                    print("FATAL: 50 consecutive non-finite losses", flush=True)
                    return
                opt.zero_grad(set_to_none=True)
                continue
            main.consec_bad = 0
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched_kind != "const" and scheduler is not None:
                scheduler.step()
            if ema_params is not None:
                with torch.no_grad():
                    d = args.ema_decay
                    for e, p in zip(ema_params, model.parameters()):
                        e.mul_(d).add_(p.detach(), alpha=1.0 - d)
            if args.qat:
                with torch.no_grad():
                    lins = [b.Wq for b in model.blocks] + [b.Wk for b in model.blocks] + \
                           [b.Wv for b in model.blocks] + [b.W1 for b in model.blocks] + \
                           [b.W2 for b in model.blocks] + \
                           [model.Wfrom, model.Wto, model.V1]
                    for lin in lins:
                        w = lin.weight.data
                        scale = (w.abs().max() / 127.0).clamp(min=1e-12)
                        w.copy_(torch.round(w / scale).clamp(-127, 127) * scale)
            step += 1
            done += int(tgt.numel())
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}: policy {pl.item():.4f} value {vl.item():.4f} "
                      f"match {match_ema:.3f}{' rct ' + format(rct.item(), '.4f') if args.recycle > 1 else ''} "
                      f"({done} positions this epoch)", flush=True)
            if args.steps and step % max(1, args.steps) == 0:
                break
            if done >= total:
                break
        model.export_blob(os.path.join(args.out, f"net_e{epoch}.bin"))
        save_atomic(model.state_dict(), os.path.join(args.out, f"net_e{epoch}.pt"))
        print(f"epoch {epoch}: exported {args.out}/net_e{epoch}.bin", flush=True)
        # EMA export: swap in shadow weights, export, restore raw weights.
        if ema_params is not None:
            raw_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema_sd = {k: e.clone() for k, e in zip(raw_sd.keys(), ema_params)}
            model.load_state_dict(ema_sd)
            model.export_blob(os.path.join(args.out, f"net_e{epoch}_ema.bin"))
            save_atomic(ema_sd, os.path.join(args.out, f"net_e{epoch}_ema.pt"))
            model.load_state_dict(raw_sd)
            print(f"epoch {epoch}: exported {args.out}/net_e{epoch}_ema.bin", flush=True)


if __name__ == "__main__":
    main()
