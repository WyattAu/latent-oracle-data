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
    """Move-conditional win% bucket classifier over the trunk's state."""

    def __init__(self, d_model: int, num_bins: int = NUM_BINS):
        super().__init__()
        self.from_emb = nn.Embedding(64, d_model)
        self.to_emb = nn.Embedding(64, d_model)
        self.promo_emb = nn.Embedding(5, d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, num_bins),
        )

    def forward(self, state, from_sq, to_sq, promo):
        m = self.from_emb(from_sq) + self.to_emb(to_sq) + self.promo_emb(promo)
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
    head = AVHead(D, args.num_bins).to("cuda")
    params = list(trunk.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    # materialise the bagz records into tensors once (positions repeat across
    # their legal moves; dedupe by fen cache)
    t0 = time.time()
    fen_cache = {}
    X_codes, X_side, X_cast, X_ep = [], [], [], []
    X_from, X_to, X_promo = [], [], []
    Y_win = []
    n_rec = 0
    for path in args.bags:
        for rec in read_bagz(path):
            fen, move, wp = decode_av_record(rec)
            key = fen
            if key not in fen_cache:
                fen_cache[key] = fen_to_inputs(fen)
            codes, side, cast, ep = fen_cache[key]
            mv = chess.Move.from_uci(move)
            X_codes.append(codes); X_side.append(side)
            X_cast.append(cast); X_ep.append(ep)
            X_from.append(mv.from_square); X_to.append(mv.to_square)
            X_promo.append(PROMO_ID.get(mv.promotion, 0))
            Y_win.append(wp)
            n_rec += 1
            if args.record_limit and n_rec >= args.record_limit:
                break
        if args.record_limit and n_rec >= args.record_limit:
            break
        print(f"  {path.split('/')[-1]}: cumulative {n_rec} records "
              f"({time.time() - t0:.0f}s)", flush=True)
        if n_rec and len(fen_cache) > 6_000_000:
            break  # fen cache guard
    X_codes = np.stack(X_codes)
    X_side = np.array(X_side, dtype=np.int64)
    X_cast = np.array(X_cast, dtype=np.int64)
    X_ep = np.array(X_ep, dtype=np.int64)
    X_from = np.array(X_from, dtype=np.int64)
    X_to = np.array(X_to, dtype=np.int64)
    X_promo = np.array(X_promo, dtype=np.int64)
    Y_win = np.array(Y_win, dtype=np.float32)
    del fen_cache
    print(f"dataset: {len(Y_win)} (pos, move, wp) records, "
          f"{len(np.unique(X_codes, axis=0))} unique boards (capped view), "
          f"{time.time() - t0:.0f}s", flush=True)

    codes_t = torch.from_numpy(X_codes.astype(np.int64)).to("cuda")
    side_t = torch.from_numpy(X_side.astype(np.int64)).to("cuda")
    cast_t = torch.from_numpy(X_cast.astype(np.int64)).to("cuda")
    # ep feature = file index (trainer convention): 0 if none, else 1 + file
    ep_feat = np.where(X_ep >= 64, 0, 1 + (X_ep % 8)).astype(np.int64)
    ep_t = torch.from_numpy(ep_feat).to("cuda")
    from_t = torch.from_numpy(X_from).to("cuda")
    to_t = torch.from_numpy(X_to).to("cuda")
    promo_t = torch.from_numpy(X_promo).to("cuda")
    win_t = torch.from_numpy(Y_win).to("cuda")

    n = len(Y_win)
    consec_bad = 0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), device="cuda")
        trunk(codes_t[idx], side_t[idx], castle=cast_t[idx], ep=ep_t[idx])
        state_vec = feats["h"].mean(dim=1)  # pool the residual stream (B, D)
        logits = head(state_vec, from_t[idx], to_t[idx], promo_t[idx])
        target = hl_gauss_targets(win_t[idx])
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
                cal = (ew - win_t[idx]).abs().mean()
            print(f"step {step}: loss {loss.item():.4f} bucket-acc {acc.item():.3f} "
                  f"|E[win]-wp| {cal.item():.4f} consec_bad {consec_bad}", flush=True)
        if step % args.export_every == 0 or step == args.steps:
            torch.save({"trunk": trunk.state_dict(), "head": head.state_dict()},
                       f"{args.out}/av_s{step // args.export_every}.pt")
            print(f"exported {args.out}/av_s{step // args.export_every}.pt",
                  flush=True)


PROMO_ID = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

if __name__ == "__main__":
    main()
