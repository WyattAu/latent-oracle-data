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


style = "pooled"
import sys as _sys
DEV = "cuda"


def best_move(fen):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]
    codes, side, cast, ep_feat, _ = fen_to_inputs(fen)
    codes_t = torch.from_numpy(codes.astype(np.int64)).to(DEV).unsqueeze(0)
    side_t = torch.tensor([side], dtype=torch.long, device=DEV)
    cast_t = torch.tensor([cast], dtype=torch.long, device=DEV)
    ep_t = torch.tensor([ep_feat], dtype=torch.long, device=DEV)
    feats = {}
    handle = trunk.blocks[-1].register_forward_hook(
        lambda m, i, o: feats.__setitem__("h", o))
    with torch.no_grad():
        trunk(codes_t, side_t, castle=cast_t, ep=ep_t)
        h = feats["h"]
        state = h if style == "xattn" else h.mean(dim=1)
        cand = []
        for mv in legal:
            promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4}.get(mv.promotion, 0)
            cand.append((mv, mv.from_square, mv.to_square, promo))
        from_t = torch.tensor([c[1] for c in cand], device=DEV)
        to_t = torch.tensor([c[2] for c in cand], device=DEV)
        pr_t = torch.tensor([c[3] for c in cand], device=DEV)
        state_b = state.expand(len(cand), -1, -1) if style == "xattn" \
            else state.expand(len(cand), -1)
        logits = head(state_b, from_t, to_t, pr_t)
        ew = expected_win(logits)
        best = cand[int(torch.argmax(ew).item())][0]
    handle.remove()
    return best


def main():
    global trunk, head, style
    ckpt = None
    for _a in _sys.argv[1:]:
        if _a.startswith("--device="):
            globals()["DEV"] = _a.split("=", 1)[1]
        elif ckpt is None:
            ckpt = _a
    ckpt = ckpt or "/home/wyatt/data/runs/bc_avlabels40m/net_e0_ema.pt"
    trunk = ChessNet(v3=False).to(DEV)
    ck = torch.load(ckpt, map_location=DEV, weights_only=True)
    style = ck.get("style", "pooled")
    head = AVHead(trunk.blocks[-1].Wq.out_features, NUM_BINS,
                  style=style).to(DEV)
    sd = ck["trunk"] if "trunk" in ck else ck
    trunk.load_state_dict(sd)
    head.load_state_dict(ck["head"])
    style = ck.get("style", "pooled")
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
