"""DiffuSearch strength test: diffusion-sampled play vs greedy BC play.

The queue's DiffuSearch verdict is a0-match, which measures how well the
model predicts the move a human played. That is predictability, not playing
strength -- and the gate (0.25 / 0.40) is saturated by it. This harness
answers the question that actually matters: does sampling from the denoised
diffusion policy (with a legal-move gate) play better than the greedy policy
of the same network?

Both sides use the same BC weights, so the comparison isolates the sampling
mechanism. Games start from a spread of real positions (the sample cache's
source states) and are played with alternating colors; a ply cap decides
otherwise-unfinished games by material.

Usage:
  python trainer/diffusion_playout.py \
      --diffu-run /home/wyatt/data/runs/diffu_v1 --ckpt diffu_e1.pt \
      --bc-net /home/wyatt/data/runs/bc_v1/net_e2.pt \
      --cache /home/wyatt/data/runs/diffu_v1_h2_samples.pt \
      --games 40 --ply-cap 200 --T 16 [--temperature 0.0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

import infer_diffusion as idiff  # noqa: E402
import train_diffusion as td  # noqa: E402
from amz_pilot import codes_from_board  # noqa: E402
from model import ChessNet  # noqa: E402
from robust_io import load_artifact  # noqa: E402

SRC_LEN = 1 + td.STATE_LEN + 1


def encode_position(board: chess.Board) -> list[int]:
    codes = codes_from_board(board).astype("int64")
    castle = 0
    if board.has_castling_rights(chess.WHITE) and board.occupied_co[chess.WHITE] & chess.BB_H1:
        castle |= 1
    if board.has_castling_rights(chess.WHITE) and board.occupied_co[chess.WHITE] & chess.BB_A1:
        castle |= 2
    if board.has_castling_rights(chess.BLACK) and board.occupied_co[chess.BLACK] & chess.BB_H8:
        castle |= 4
    if board.has_castling_rights(chess.BLACK) and board.occupied_co[chess.BLACK] & chess.BB_A8:
        castle |= 8
    ep = board.ep_square if board.ep_square is not None else td.NO_EP if hasattr(td, "NO_EP") else 255
    return [td.SEP] + td.encode_state(codes, 0 if board.turn else 1, castle, ep) + [td.SEP]


def board_from_codes_state(src: list[int]) -> chess.Board:
    return idiff.decode_source_board(src)


@torch.no_grad()
def diffusion_move(model, board: chess.Board, T: int, temperature: float,
                   rng: random.Random) -> chess.Move | None:
    """One DiffuSearch move, using the intended algorithm.

    This MUST mirror infer_diffusion.denoise: mask the whole target region up
    front, then reveal tokens easy-first by confidence (roughly 1/(t+1) per
    step) and read a0 at the END. An earlier version unmasked a0 first and
    read it immediately; with no context revealed that scores 0.08 a0
    agreement against 0.33 for the real schedule, so it measures a different
    (much weaker) policy than the one the queue evaluates.
    """
    src = encode_position(board)
    # encode_position already returns [SEP] state [SEP], so it starts at index 0.
    # Placing it at index 1 inserted a duplicate SEP and shifted the a0 slot by
    # one, which collapsed agreement with the reference to 0.08.
    #
    # The row must also have the SAME LENGTH as the training samples
    # (horizon 2 => 2 + STATE_LEN + 2*(1 + STATE_LEN) = 205 tokens): the model
    # was trained on that shape and a shorter row measurably degrades it
    # (0.22 vs 0.33 agreement).
    horizon = getattr(model, "_playout_horizon", 2)
    width = 2 + td.STATE_LEN + horizon * (1 + td.STATE_LEN)
    row = np_int64(width)
    row[0:len(src)] = src
    legal = idiff.legal_move_token_ids(src)
    if not legal:
        return None
    dev = _dev(model)
    gate = torch.full((td.V,), float("-inf"), device=dev)
    for tid in legal:
        gate[tid] = 0.0

    x = torch.from_numpy(row).unsqueeze(0).to(dev).long()
    x[0, SRC_LEN:] = td.MASK
    remaining = x[0] == td.MASK
    for t in range(T - 1, -1, -1):
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0]
        conf = probs.max(-1).values
        conf = torch.where(remaining, conf, torch.full_like(conf, -1.0))
        a0_conf = conf[SRC_LEN]
        n_masked = int(remaining.sum())
        if n_masked == 0:
            break
        n_unmask = max(1, n_masked // (t + 1))
        order = conf.argsort(descending=True)
        for j in order[:n_unmask].tolist():
            if not remaining[j]:
                continue
            pick = int(probs[j].argmax())
            if j == SRC_LEN:
                # a0 must be a legal move; fall back to the best legal token
                masked = probs[j] + gate
                pick = int(masked.argmax()) if float(masked.max()) > float("-1e30") else pick
            x[0, j] = pick
            remaining[j] = False
        if t == 0 and remaining[SRC_LEN]:
            x[0, SRC_LEN] = int((probs[SRC_LEN] + gate).argmax())
            remaining[SRC_LEN] = False
    mv = idiff.id_to_move_tok(int(x[0, SRC_LEN]))
    if mv is None or mv[0] == 255:
        return None
    from_sq, to_sq, promo_idx = mv
    promo = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promo_idx] \
        if promo_idx else None
    for cand in board.legal_moves:
        if cand.from_square == from_sq and cand.to_square == to_sq \
                and (cand.promotion or None) == promo:
            return cand
    return None


def np_int64(n: int):
    import numpy as np
    out = np.zeros(n, dtype=np.int64)
    out[:] = td.PAD
    return out


def _dev(model):
    return next(model.parameters()).device


def _mirror_board(board: chess.Board) -> chess.Board:
    """File-mirrored position: square s -> s ^ 7, colors and turn unchanged."""
    m = chess.Board(None)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            m.set_piece_at(sq ^ 7, p)
    m.turn = board.turn
    # castling rights cannot survive a file mirror (the king would not be on
    # e1), and the engine clears them in the mirrored view
    return m


@torch.no_grad()
def greedy_move(bc: ChessNet, board: chess.Board, v3: bool,
                mirror: bool = False) -> chess.Move | None:
    codes = codes_from_board(board).astype("int64")   # embedding needs int64
    dev = _dev(bc)
    c = torch.from_numpy(codes).unsqueeze(0).to(dev)
    side = torch.tensor([0 if board.turn else 1], device=dev)
    kw = {}
    if v3 or getattr(bc, "v3", False):
        # real v3 state, matching train.py --state-aware: castle is the raw
        # 4-bit mask, ep is file+1 with 0 meaning "none"
        castle = 0
        if board.has_castling_rights(chess.WHITE) and board.occupied_co[chess.WHITE] & chess.BB_H1:
            castle |= 1
        if board.has_castling_rights(chess.WHITE) and board.occupied_co[chess.WHITE] & chess.BB_A1:
            castle |= 2
        if board.has_castling_rights(chess.BLACK) and board.occupied_co[chess.BLACK] & chess.BB_H8:
            castle |= 4
        if board.has_castling_rights(chess.BLACK) and board.occupied_co[chess.BLACK] & chess.BB_A8:
            castle |= 8
        ep = 0 if board.ep_square is None else (board.ep_square % 8) + 1
        kw = {"castle": torch.tensor([castle], device=dev),
              "ep": torch.tensor([ep], device=dev)}
    kw = {k: v.to(dev) for k, v in kw.items()}
    scores, promo, _ = bc(c, side, **kw)
    sm = scores[0]          # (64, 64) from -> to, matching the engine's flatten
    pm = promo[0]           # (4,) pooled promotion logits: N, B, R, Q
    mm = pmm = None
    if mirror:
        mboard = _mirror_board(board)
        mc = torch.from_numpy(codes_from_board(mboard).astype("int64")).unsqueeze(0).to(dev)
        mkw = {}
        if kw:
            mkw = {"castle": torch.zeros_like(kw["castle"])}   # rights cleared, as in the engine
            if board.ep_square is not None:
                mkw["ep"] = torch.tensor([(board.ep_square ^ 7) % 8 + 1], device=dev)
            else:
                mkw["ep"] = torch.zeros_like(kw["ep"])
        mscores, mpromo, _ = bc(mc, side, **mkw)
        mm, pmm = mscores[0], mpromo[0]
    order = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}
    best, best_score = None, -1e30
    for mv in board.legal_moves:
        s = float(sm[mv.from_square, mv.to_square])
        if mv.promotion:
            s += float(pm[order[mv.promotion]])
        if mm is not None and not (abs(chess.square_file(mv.to_square)
                                       - chess.square_file(mv.from_square)) == 2
                                   and chess.square_rank(mv.from_square) == 0
                                   and chess.square_rank(mv.to_square) == 0):
            s += float(mm[mv.from_square ^ 7, mv.to_square ^ 7])
            if mv.promotion:
                s += float(pmm[order[mv.promotion]])
        if s > best_score:
            best, best_score = mv, s
    return best


def material(board: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
            chess.QUEEN: 9, chess.KING: 0}
    return sum(vals[board.piece_at(sq).piece_type] for sq in chess.SQUARES
               if board.piece_at(sq) and board.piece_at(sq).color == color)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffu-run", required=True)
    ap.add_argument("--ckpt", default="diffu_e1.pt")
    ap.add_argument("--bc-net", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--ply-cap", type=int, default=200)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = argmax of the gated distribution")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--diffu-color", choices=["white", "black", "both"], default="both")
    ap.add_argument("--both-greedy", action="store_true",
                    help="harness control: greedy BC vs greedy BC should score ~0.500")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(Path(args.diffu_run) / "config.json"))
    from train_diffusion import DiffuNet
    dnet = DiffuNet(cfg["d"], cfg["layers"], cfg["heads"], cfg["dff"],
                    max_len=cfg["max_len"]).to(device)
    dnet.load_state_dict(load_artifact(str(Path(args.diffu_run) / args.ckpt), weights_only=True))
    dnet.eval()

    bc = ChessNet().to(device)
    bc.load_state_dict(load_artifact(args.bc_net, weights_only=True))
    bc.eval()

    samples = load_artifact(args.cache, weights_only=False)
    rng = random.Random(args.seed)

    results: Counter = Counter()
    used: list[str] = []
    g = 0
    while g < args.games:
        row = samples[rng.randrange(len(samples))].tolist()
        board = board_from_codes_state(row[:1 + td.STATE_LEN])
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            continue
        if board.is_game_over():
            continue
        if args.diffu_color == "both":
            diffu_white = (g % 2 == 0)
        else:
            diffu_white = (args.diffu_color == "white")
        history = []
        while (not board.is_game_over()) and len(history) < args.ply_cap:
            mover = board.turn
            is_diffu = (mover == chess.WHITE) == diffu_white
            if is_diffu and not args.both_greedy:
                mv = diffusion_move(dnet, board, args.T, args.temperature, rng)
            else:
                mv = greedy_move(bc, board, v3=False)
            if mv is None:
                break
            board.push(mv)
            history.append(mv)
        if board.is_checkmate():
            outcome = "win" if board.turn != (chess.WHITE if diffu_white else chess.BLACK) else "loss"
        elif board.is_stalemate() or board.is_insufficient_material():
            outcome = "draw"
        else:
            d, b = material(board, chess.WHITE), material(board, chess.BLACK)
            dm = d - b
            outcome = ("win" if (dm > 0) == diffu_white else "loss") if dm else "draw"
        results[outcome] += 1
        used.append(f"game{g}: diffu={'W' if diffu_white else 'B'} {outcome} "
                    f"plies={len(history)}")
        g += 1
        if g % 5 == 0:
            print(f"  {g}/{args.games}: {dict(results)}", flush=True)

    n = args.games
    score = (results["win"] + 0.5 * results["draw"]) / max(1, n)
    print(f"## DiffuSearch strength vs greedy BC (same weights)")
    print(f"- games {n}, T={args.T}, temperature={args.temperature}")
    print(f"- diffusion: {results['win']}W {results['draw']}D {results['loss']}L, "
          f"score {score:.3f}")
    se = (0.25 / max(1, n)) ** 0.5
    print(f"- vs 0.500 baseline: {score - 0.5:+.3f} +/- {se:.3f} (1 s.e.)")
    verdict = ("DIFFUSION WINS" if score - 0.5 > 2 * se else
               "GREEDY WINS" if 0.5 - score > 2 * se else "inconclusive")
    print(f"DIFFUSION VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
