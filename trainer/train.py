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
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format import iter_records, count_records, move_to_uci  # noqa: E402
from model import ChessNet  # noqa: E402

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


def encode_batch(samples, device):
    codes = torch.from_numpy(np.stack([s.board_codes for s in samples])).long().to(device)
    side = torch.tensor([s.side for s in samples], dtype=torch.long, device=device)
    return codes, side


def build_batch(stream, batch: int, device, require_targets: bool):
    """Pull one batch worth of samples with their masks (slow path: python-chess).

    `stream` is a persistent iterator owned by the caller so successive calls
    advance through the shard (a fresh iterator per call would replay the
    first batch forever)."""
    samples, masks, tidx, evals, labeled = [], [], [], [], []
    while len(samples) < batch:
        s = next(stream, None)
        if s is None:
            return None
        if require_targets and not s.targets:
            continue
        board = codes_to_board(s.board_codes, s.side)
        if board.is_game_over():
            continue
        mask, tflat = legal_mask_and_index(board, s.targets)
        if tflat < 0:
            continue
        samples.append(s)
        masks.append(mask)
        tidx.append(tflat)
        evals.append(s.eval_cp)
        labeled.append(len(s.targets) >= 2)  # SF-labeled records carry eval
    codes, side = encode_batch(samples, device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    tgt = torch.tensor(tidx, dtype=torch.long, device=device)
    res_t = torch.from_numpy(np.stack([s.wdl for s in samples]).astype(np.float32)).to(device)
    return codes, side, mask_t, tgt, \
        torch.tensor(evals, dtype=torch.float32, device=device), \
        torch.tensor(labeled, dtype=torch.bool, device=device), res_t


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = ChessNet(args.d, args.layers, args.heads, args.dff, args.dpol).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M on {device}", flush=True)

    total = count_records(args.shard)
    os.makedirs(args.out, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    for epoch in range(args.epochs):
        stream = iter_records(args.shard)
        done = 0
        while True:
            batch = build_batch(stream, args.batch, device, require_targets=True)
            if batch is None:
                break
            codes, side, mask, tgt, evals, labeled, res_t = batch
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                scores, promo, wdl = model(codes, side)
                flat = scores.reshape(codes.shape[0], 64 * 64)
                flat = flat.masked_fill(~mask.reshape(codes.shape[0], -1), float("-inf"))
                pl = F.cross_entropy(flat, tgt)
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
                vl = F.cross_entropy(wdl.float(), soft_t)
                loss = pl + 0.5 * vl
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            done += int(tgt.numel())
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}: policy {pl.item():.4f} value {vl.item():.4f} "
                      f"({done} positions this epoch)", flush=True)
            if args.steps and step % max(1, args.steps) == 0:
                break
            if done >= total:
                break
        model.export_blob(os.path.join(args.out, f"net_e{epoch}.bin"))
        torch.save(model.state_dict(), os.path.join(args.out, f"net_e{epoch}.pt"))
        print(f"epoch {epoch}: exported {args.out}/net_e{epoch}.bin", flush=True)


if __name__ == "__main__":
    main()
