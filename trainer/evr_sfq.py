#!/usr/bin/env python3
"""SF-Q game-edge (EVR) measurement across nets.

For each sampled position: SF (fixed nodes) evals the position (cp_before,
mover POV) and its best move. Each net picks a move; SF evals the resulting
position from the opponent's POV (cp_after_opp); the mover's cp after =
−cp_after_opp. cp_lost = cp_before − cp_after_mover. Reported in cp and
win-prob (400-logistic). The before-eval is shared across nets.

Usage: evr_sfq.py --shard S --lo A --hi B --positions K --nodes N --nets n=p[,n=p]
"""
import argparse, subprocess, random, sys
import numpy as np, torch, chess
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE
from model import ChessNet

def unpack(board_u4):
    b = board_u4.astype("<u4").view(np.uint8)
    c = np.empty(64, dtype=np.uint8)
    c[0::2] = b & 0x0F; c[1::2] = b >> 4
    return c

def to_board(codes, side, ep):
    b = chess.Board(None)
    for sq, c in enumerate(codes):
        if c:
            pt = c if c < 8 else c - 8
            b.set_piece_at(sq, chess.Piece(pt, chess.WHITE if c < 8 else chess.BLACK))
    b.turn = chess.WHITE if side == 0 else chess.BLACK
    if ep < 64: b.ep_square = ep
    return b

class SF:
    def __init__(self, path, nodes):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)
        self.nodes = nodes
        self.p.stdin.write("uci\nisready\n"); self.p.stdin.flush()
        for line in self.p.stdout:
            if line.startswith("readyok"): break
    def eval_fen(self, fen):
        self.p.stdin.write(f"position fen {fen}\ngo nodes {self.nodes}\n")
        self.p.stdin.flush()
        best, cp = None, None
        for line in self.p.stdout:
            if line.startswith("bestmove"):
                best = line.split()[1]; break
            if " score cp " in line:
                t = line.split(" score cp ")[1].split()
                if "lowerbound" in line or "upperbound" in line: continue
                cp = int(t[0])
        return best, cp
    def close(self):
        self.p.stdin.write("quit\n"); self.p.stdin.flush(); self.p.terminate()

def cp2wp(cp):
    return 1 / (1 + 10 ** (-cp / 400))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--lo", type=int, default=430_000_000)
    ap.add_argument("--hi", type=int, default=526_000_000)
    ap.add_argument("--positions", type=int, default=400)
    ap.add_argument("--nodes", type=int, default=800_000)
    ap.add_argument("--sf", default="/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--nets", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(4)
    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(args.lo, args.hi), args.positions))

    sf = SF(args.sf, args.nodes)
    f = open(args.shard, "rb")

    nets = []
    for part in args.nets.split(","):
        name, path = part.split("=", 1)
        net = ChessNet(256, 8, 8, 1024, 128).eval()
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "trunk" in sd: sd = sd["trunk"]
        net.load_state_dict(sd)
        nets.append((name, net))

    # sample valid positions and get the SF before-eval + best move
    rows = []
    want = idxs
    fi = 0
    i = 0
    while fi < len(want) and i < args.hi:
        i = want[fi]
        f.seek(16 + i * 64)
        rec = np.frombuffer(f.read(64), dtype=REC_DTYPE)[0]
        if rec["n_targets"] != 0:
            codes = unpack(rec["board"])
            side = int(rec["side"])
            ep = int(rec["ep"])
            b = to_board(codes, side, ep)
            if not b.is_game_over():
                best, cp = sf.eval_fen(b.fen())
                if best is not None and cp is not None:
                    cp_mover = cp if b.turn == chess.WHITE else -cp
                    rows.append((codes, side, ep, b, best, cp_mover))
                    fi += 1
                    continue
        fi += 1
        if fi % 20 == 0: print(f"sampled {fi}/{len(want)}", flush=True)
    f.close()
    print(f"{len(rows)} positions with SF evals", flush=True)

    # per net: pick moves, then SF the after-positions
    per_net = {n: [] for n, _ in nets}
    for name, net in nets:
        picked = []
        k = 0
        for (codes, side, ep, b, best, cp_mover) in rows:
            codes_t = torch.tensor(codes).long()[None]
            side_t = torch.tensor([side])
            with torch.no_grad():
                scores, _, _ = net(codes_t, side_t)
            flat = scores[0].reshape(-1)
            mask = np.full(4096, -1e30, dtype=np.float32)
            for m in b.legal_moves:
                mask[m.from_square * 64 + m.to_square] = 0.0
            ft = torch.tensor(mask)
            idx = int((flat + ft).argmax())
            picked.append((idx // 64, idx % 64))
        # eval the after-positions
        for (codes, side, ep, b, best, cp_mover), (fr, to) in zip(rows, picked):
            mv = chess.Move(fr, to)
            if mv not in b.legal_moves:
                for m2 in b.legal_moves:
                    if m2.from_square == fr and m2.to_square == to:
                        mv = m2; break
            b.push(mv)
            _, cp = sf.eval_fen(b.fen())
            cp_after_mover = -cp if cp is not None else 0
            per_net[name].append(cp_mover - cp_after_mover)
            b.pop()
        arr = np.array(per_net[name])
        print(f"{name:<10} n={len(arr)} mean_cp_lost={arr.mean():.1f} "
              f"median={np.median(arr):.1f} mean|max(0,cp)|={np.mean(np.maximum(arr,0)):.1f}",
              flush=True)
    sf.close()

if __name__ == "__main__":
    main()
