"""Action-value distillation from DeepMind ChessBench (loop 1).

Trains a move-conditional 128-bin win% head on the GAB transformer trunk:
the input is the board plus the CANDIDATE MOVE (embedded), the output is a
HL-Gauss soft distribution over win probability buckets. At inference the
engine scores each legal move (~35 forwards) and picks the argmax expected
win%. This is the DeepNoc supervision (action-value beats best-move BC by
83.3% vs 65.7% puzzle accuracy at 9M params) that our GRPO static-eval
reward-hacking failure pointed to.

Data: ChessBench .bag files (see chessbench_data.py) -- (fen, move,
win_prob) triples, Stockfish-16 win probabilities at 50ms/move.
"""
import argparse, math, random, struct, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn

from model import ChessNet
from chessbench_data import read_bagz, decode_av_record
from pgn_to_shard import encode_board  # validated roundtrip encoder

NUM_BINS = 128
BIN_EDGES = np.linspace(0.0, 1.0, NUM_BINS + 1)
BIN_CENTERS = ((BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2).astype(np.float32)
HL_SIGMA = (BIN_EDGES[1] - BIN_EDGES[0]) / math.sqrt(2.0)


def hl_gauss(win_prob: float) -> np.ndarray:
    """HL-Gauss soft label: Gaussian-smoothed histogram over the buckets."""
    centers = BIN_CENTERS
    weights = np.exp(-0.5 * ((centers - win_prob) / HL_SIGMA) ** 2)
    return (weights / weights.sum()).astype(np.float32)


PIECE_CODE = {}
for pt in range(1, 7):
    PIECE_CODE[(pt, chess.WHITE)] = pt
    PIECE_CODE[(pt, chess.BLACK)] = pt + 8


def fen_to_inputs(fen: str):
    """FEN -> (codes u8[64], side u1, castling u8, ep u8) in shard convention."""
    board = chess.Board(fen)
    codes = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is not None:
            codes[sq] = PIECE_CODE[(pc.piece_type, pc.color)]
    side = 0 if board.turn == chess.WHITE else 1
    rights = 0
    cr = board.clean_castling_rights()
    if cr & chess.BB_A1: rights |= 2
    if cr & chess.BB_H1: rights |= 1
    if cr & chess.BB_A8: rights |= 8
    if cr & chess.BB_H8: rights |= 4
    ep = board.ep_square if board.ep_square is not None else 255
    return codes, side, rights, ep


class AVHead(nn.Module):
    """Move-conditional win% bucket classifier over the trunk's state.

    style="pooled": mean-pool the board tokens, concat the move embedding,
    MLP -> bins. The move and the state only mix in the last MLP.
    style="xattn": the move embedding attends over all 64 board tokens
    (DeepNoc-style move conditioning, one cross-attention layer) before the
    output MLP. Keeps the spec's ~one-trunk-pass inference: the cross-attention
    runs per candidate move but is tiny next to the 6-block trunk.
    """

    def __init__(self, d_model: int, num_bins: int = NUM_BINS,
                 style: str = "pooled"):
        super().__init__()
        self.style = style
        self.from_emb = nn.Embedding(64, d_model)
        self.to_emb = nn.Embedding(64, d_model)
        self.promo_emb = nn.Embedding(5, d_model)
        if style == "xattn":
            self.xattn = nn.MultiheadAttention(d_model, 4, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(),
                nn.Linear(d_model, num_bins),
            )
        elif style == "pooled":
            self.net = nn.Sequential(
                nn.Linear(d_model * 2, d_model), nn.GELU(),
                nn.Linear(d_model, num_bins),
            )
        else:
            raise ValueError(f"unknown head style: {style}")

    def forward(self, state, from_sq, to_sq, promo):
        m = self.from_emb(from_sq) + self.to_emb(to_sq) + self.promo_emb(promo)
        if self.style == "xattn":
            # state: (B, 64, D) board tokens; query = move token (B, 1, D)
            out, _ = self.xattn(m.unsqueeze(1), state, state)
            return self.net(self.norm(out.squeeze(1)))
        return self.net(torch.cat([state, m], dim=-1))


def hl_gauss_targets(win_probs: torch.Tensor) -> torch.Tensor:
    """(B,) win probs -> (B, 128) soft bucket targets on device."""
    centers = torch.tensor(BIN_CENTERS, device=win_probs.device)
    sigma = (BIN_EDGES[1] - BIN_EDGES[0]) / math.sqrt(2.0)
    w = torch.exp(-0.5 * ((centers.unsqueeze(0) - win_probs.unsqueeze(1)) / sigma) ** 2)
    return w / w.sum(dim=1, keepdim=True)


def expected_win(bins_logits: torch.Tensor) -> torch.Tensor:
    return (F.softmax(bins_logits, dim=-1)
            * torch.tensor(BIN_CENTERS, device=bins_logits.device)).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--head-style", choices=["pooled", "xattn"],
                    default="pooled",
                    help="AV head: pooled concat (v1) or move-token "
                         "cross-attention (DeepNoc-style conditioning)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-from", default="")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-bins", type=int, default=NUM_BINS)
    ap.add_argument("--record-limit", type=int, default=0,
                    help="0 = stream the whole file")
    ap.add_argument("--export-every", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(7)
    trunk = ChessNet(v3=True).to("cuda")
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cuda", weights_only=True)
        if "trunk" in sd:  # this script's checkpoints nest trunk+head
            sd = sd["trunk"]
        trunk.load_state_dict(sd)
        print(f"trunk init from {args.init_from}", flush=True)
    # capture the residual stream leaving the LAST block: the AV head pools
    # this state (DeepNoc-style trunk pooling) instead of the policy logits
    feats = {}
    trunk.blocks[-1].register_forward_hook(
        lambda m, inp, out: feats.__setitem__("h", out))
    with torch.no_grad():
        _probe_codes = torch.zeros(1, 64, dtype=torch.long, device="cuda")
        _probe_side = torch.zeros(1, dtype=torch.long, device="cuda")
        trunk(_probe_codes, _probe_side)
        D = feats["h"].shape[-1]
    print(f"trunk state dim: {D}", flush=True)
    import os
    os.makedirs(args.out, exist_ok=True)
    head = AVHead(D, args.num_bins, style=args.head_style).to("cuda")
    params = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # streaming build into PREALLOCATED CPU arrays (uint8/int32): 20M records
    # fit in ~3GB RAM; the fen_cache approach (dict of per-fen decodes) dies
    # at this scale, and per-record decode without cache is ~100 min -- so
    # cache decoded FENs in a bounded dict (hot set) and accept re-decodes.
    cap = args.record_limit or 20_000_000
    t0 = time.time()
    X_codes = np.zeros((cap, 64), dtype=np.uint8)
    X_meta = np.zeros((cap, 4), dtype=np.int32)  # side, cast, ep_feat, promo
    X_from = np.zeros(cap, dtype=np.int16)
    X_to = np.zeros(cap, dtype=np.int16)
    Y_win = np.zeros(cap, dtype=np.float32)
    fen_cache = {}
    n_rec = 0
    skipped_bad = 0
    for path in args.bags:
        if n_rec >= cap:
            break
        for rec in read_bagz(path):
            if n_rec >= cap:
                break
            fen, move, wp = decode_av_record(rec)
            try:
                inputs = fen_cache.get(fen)
                if inputs is None:
                    if len(fen_cache) < 3_000_000:
                        inputs = fen_to_inputs(fen)
                        fen_cache[fen] = inputs
                    else:
                        inputs = fen_to_inputs(fen)
            except (ValueError, chess.InvalidMoveError):
                skipped_bad += 1
                continue
            codes, side, cast, ep = inputs
            try:
                mv = chess.Move.from_uci(move)
            except chess.InvalidMoveError:
                skipped_bad += 1
                continue
            X_codes[n_rec] = codes
            X_meta[n_rec, 0] = side
            X_meta[n_rec, 1] = cast
            X_meta[n_rec, 2] = 0 if ep >= 64 else 1 + (ep % 8)
            X_meta[n_rec, 3] = PROMO_ID.get(mv.promotion, 0)
            X_from[n_rec] = mv.from_square
            X_to[n_rec] = mv.to_square
            Y_win[n_rec] = wp
            n_rec += 1
        print(f"  {path.split('/')[-1]}: cumulative {n_rec} records "
              f"({time.time() - t0:.0f}s, cache {len(fen_cache)}, "
              f"bad {skipped_bad})", flush=True)
    n = n_rec
    X_from_s = X_from[:n].astype(np.int64)
    X_to_s = X_to[:n].astype(np.int64)
    print(f"dataset: {n} (pos, move, wp) records ({time.time() - t0:.0f}s, "
          f"bad {skipped_bad})", flush=True)

    # per-batch numpy gather -> pinned transfer; GPU never holds the table
    codes_cpu = X_codes[:n]
    meta_cpu = X_meta[:n]
    win_cpu = Y_win[:n]
    from_cpu = X_from_s
    to_cpu = X_to[:n].astype(np.int64)

    def batch_to_device(idx_np):
        codes_b = torch.from_numpy(codes_cpu[idx_np].astype(np.int64)).to("cuda")
        meta_b = torch.from_numpy(meta_cpu[idx_np].astype(np.int64)).to("cuda")
        from_b = torch.from_numpy(from_cpu[idx_np]).to("cuda")
        to_b = torch.from_numpy(to_cpu[idx_np]).to("cuda")
        promo_b = meta_b[:, 3]
        win_b = torch.from_numpy(win_cpu[idx_np]).to("cuda")
        return codes_b, meta_b[:, 0], meta_b[:, 1], meta_b[:, 2],             from_b, to_b, promo_b, win_b
    consec_bad = 0
    for step in range(1, args.steps + 1):
        idx_np = np.random.randint(0, n, (args.batch,))
        codes_b, side_b, cast_b, ep_b, from_b, to_b, promo_b, win_b = \
            batch_to_device(idx_np)
        trunk(codes_b, side_b, castle=cast_b, ep=ep_b)
        h = feats["h"]  # (B, 64, D) residual stream
        if args.head_style == "xattn":
            logits = head(h, from_b, to_b, promo_b)
        else:
            logits = head(h.mean(dim=1), from_b, to_b, promo_b)
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
        torch.nn.utils.clip_grad_norm_(
            list(trunk.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            with torch.no_grad():
                ew = expected_win(logits)
                acc = ((torch.argmax(logits, dim=-1) ==
                        torch.argmax(target, dim=-1)).float().mean())
                cal = (ew - win_b).abs().mean()
            print(f"step {step}: loss {loss.item():.4f} bucket-acc {acc.item():.3f} "
                  f"|E[win]-wp| {cal.item():.4f} consec_bad {consec_bad}", flush=True)
        if step % args.export_every == 0 or step == args.steps:
            torch.save({"trunk": trunk.state_dict(), "head": head.state_dict(),
                        "style": args.head_style},
                       f"{args.out}/av_s{step // args.export_every}.pt")
            print(f"exported {args.out}/av_s{step // args.export_every}.pt",
                  flush=True)


PROMO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

if __name__ == "__main__":
    main()
