"""Minimal UCI wrapper for a ChessBench AV checkpoint.

Move selection: one trunk forward, then the AV head scored over ALL legal
moves (batched); argmax expected-win. ~30-60ms/move on GPU.
"""
import sys
import numpy as np
import torch
import chess
from model import ChessNet
from chessbench_av import AVHead, expected_win, PIECE_CODE, NUM_BINS
from eval_chessbench_av import fen_to_inputs


def best_move(fen):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    codes, side, cast, ep_feat, _ = fen_to_inputs(fen)
    codes_t = torch.from_numpy(codes.astype(np.int64)).to("cuda").unsqueeze(0)
    side_t = torch.tensor([side], dtype=torch.long, device="cuda")
    cast_t = torch.tensor([cast], dtype=torch.long, device="cuda")
    ep_t = torch.tensor([ep_feat], dtype=torch.long, device="cuda")
    feats = {}
    h = trunk.blocks[-1].register_forward_hook(
        lambda m, i, o: feats.__setitem__("h", o))
    with torch.no_grad():
        trunk(codes_t, side_t, castle=cast_t, ep=ep_t)
        state = feats["h"].mean(dim=1)
        cand = []
        for mv in legal:
            promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4}.get(mv.promotion, 0)
            cand.append((mv, mv.from_square, mv.to_square, promo))
        from_t = torch.tensor([c[1] for c in cand], device="cuda")
        to_t = torch.tensor([c[2] for c in cand], device="cuda")
        pr_t = torch.tensor([c[3] for c in cand], device="cuda")
        logits = head(state.expand(len(cand), -1), from_t, to_t, pr_t)
        ew = expected_win(logits)
        best = cand[int(torch.argmax(ew).item())][0]
    h.remove()
    return best


def main():
    global trunk, head
    ckpt = sys.argv[1]
    trunk = ChessNet(v3=True).to("cuda")
    head = AVHead(trunk.blocks[-1].Wq.out_features, NUM_BINS).to("cuda")
    ck = torch.load(ckpt, map_location="cuda", weights_only=True)
    trunk.load_state_dict(ck["trunk"])
    head.load_state_dict(ck["head"])
    trunk.eval(); head.eval()

    board = chess.Board()
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue
        cmd = line.split()[0]
        if cmd == "uci":
            print("id name ChessBench-AV", flush=True)
            print("uciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            parts = line.split()
            if parts[1] == "startpos":
                board = chess.Board()
                moves = parts[3:] if len(parts) > 3 and parts[2] == "moves" else []
            elif parts[1] == "fen":
                fen = " ".join(parts[2:8])
                board = chess.Board(fen)
                moves = parts[9:] if len(parts) > 9 and parts[8] == "moves" else []
            for mv in moves:
                board.push(chess.Move.from_uci(mv))
        elif cmd == "go":
            bm = best_move(board.fen())
            print(f"bestmove {bm.uci() if bm else '(none)'}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
