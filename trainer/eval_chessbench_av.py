"""Evaluate a ChessBench AV checkpoint on the public puzzles.csv.

Scores every legal move of each puzzle position with the AV head and counts
argmax-expected-win matches against the puzzle's first solution move.
Reference points (DeepNoc paper, 10k puzzles): GC-270M 95.4%, GC-9M 88.9%,
their BC-9M 65.7%, LC0-T82-policy 88.6%, AZ-policy 74.4% (approx), random ~2%.
"""
import argparse, csv, sys
import numpy as np
import torch
import chess
from model import ChessNet
from chessbench_av import AVHead, expected_win, PIECE_CODE
from chessbench_av import NUM_BINS


def fen_to_inputs(fen):
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
    ep_feat = 0 if ep >= 64 else 1 + (ep % 8)
    return codes, side, rights, ep_feat, board


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--puzzles", default="/home/wyatt/data/chessbench/puzzles.csv")
    ap.add_argument("--num-puzzles", type=int, default=2000)
    args = ap.parse_args()

    trunk = ChessNet(v3=True).to("cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=True)
    style = ck.get("style", "pooled")
    head = AVHead(trunk.blocks[-1].Wq.out_features, NUM_BINS,
                  style=style).to("cuda")
    trunk.load_state_dict(ck["trunk"])
    head.load_state_dict(ck["head"])
    style = ck.get("style", "pooled")
    trunk.eval(); head.eval()

    rows = []
    with open(args.puzzles) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= args.num_puzzles:
                break
            rows.append(row)
    print(f"puzzles: {len(rows)}", flush=True)

    correct = total = 0
    feats = {}
    handled = trunk.blocks[-1].register_forward_hook(
        lambda m, i, o: feats.__setitem__("h", o))
    with torch.no_grad():
        for row in rows:
            board = chess.Board(row["FEN"])
            legal = list(board.legal_moves)
            sol = row["Moves"].split()[0]
            if not legal:
                continue
            codes, side, cast, ep_feat, _ = fen_to_inputs(row["FEN"])
            codes_t = torch.from_numpy(codes.astype(np.int64)).to("cuda")
            side_t = torch.tensor([side], dtype=torch.long, device="cuda")
            cast_t = torch.tensor([cast], dtype=torch.long, device="cuda")
            ep_t = torch.tensor([ep_feat], dtype=torch.long, device="cuda")
            # score every legal move: one batched forward per candidate
            ew_best, best_uci = -1.0, None
            trunk(codes_t.unsqueeze(0), side_t, castle=cast_t, ep=ep_t)
            h = feats["h"]
            state = h if style == "xattn" else h.mean(dim=1)
            cand = []
            for mv in legal:
                promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                         chess.ROOK: 3, chess.QUEEN: 4}.get(mv.promotion, 0)
                cand.append((mv.uci(), mv.from_square, mv.to_square, promo))
            from_t = torch.tensor([c[1] for c in cand], device="cuda")
            to_t = torch.tensor([c[2] for c in cand], device="cuda")
            pr_t = torch.tensor([c[3] for c in cand], device="cuda")
            state_b = state.expand(len(cand), -1, -1) if style == "xattn" \
                else state.expand(len(cand), -1)
            logits = head(state_b, from_t, to_t, pr_t)
            ew = expected_win(logits)
            best_i = int(torch.argmax(ew).item())
            total += 1
            if cand[best_i][0] == sol:
                correct += 1
    print(f"AV puzzle accuracy: {correct}/{total} = {correct/total*100:.2f}% "
          f"(DeepNoc refs: GC-9M 88.9%, BC-9M 65.7%, random ~2%)", flush=True)


if __name__ == "__main__":
    main()
