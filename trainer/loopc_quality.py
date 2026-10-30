import sys, random
import numpy as np, torch
sys.path.insert(0, "trainer")
import chess
from format import REC_DTYPE, MaskSidecar
from model import ChessNet

torch.set_num_threads(6)
DEV = "cpu"
A = "/home/wyatt/data/shards/bc_sub/slice1.full.shard"
AM = A + ".mask"
sc = MaskSidecar(AM)
f = open(A, "rb")
rng = random.Random(11)
positions = []
for lo, n in [(1_000_000, 300), (75_000_000, 300)]:
    for i in range(lo, lo + n):
        f.seek(16 + i * 64)
        rec = np.frombuffer(f.read(64), dtype=REC_DTYPE)[0]
        if rec["n_targets"] == 0: continue
        b = chess.Board(None)
        bdis = rec["board"].astype("<u4").view(np.uint8)
        codes = np.empty(64, dtype=np.uint8)
        codes[0::2] = bdis & 0x0F; codes[1::2] = bdis >> 4
        for sq, c in enumerate(codes):
            if c:
                pt = c if c < 8 else c - 8
                b.set_piece_at(sq, chess.Piece(pt, chess.WHITE if c < 8 else chess.BLACK))
        b.turn = chess.WHITE if int(rec["side"]) == 0 else chess.BLACK
        if rec["ep"] < 64: b.ep_square = int(rec["ep"])
        if b.is_game_over(): continue
        flat = sc.mask_indices(i).tolist()
        mask = np.full(4096, -1e30, dtype=np.float32); mask[flat] = 0.0
        if mask[int(rec["targets"][0])*64+int(rec["targets"][1])] != 0: continue
        positions.append((codes, int(rec["side"]), mask,
                          int(rec["targets"][0])*64+int(rec["targets"][1]),
                          int(rec["targets"][2]) if rec["targets"][2] not in (0,255) else 0))

net = ChessNet(256, 8, 8, 1024, 128).to(DEV).eval()
net.load_state_dict(torch.load("/home/wyatt/data/runs/bcpool_1ep/net_e0.pt",
                               map_location=DEV, weights_only=True))

stats = []
with torch.no_grad():
    for i in range(0, len(positions), 256):
        chunk = positions[i:i+256]
        codes = torch.tensor(np.stack([p[0] for p in chunk]).astype(np.int64)).to(DEV)
        sides = torch.tensor([p[1] for p in chunk]).to(DEV)
        p1 = net.forward_recycle(codes, sides, R=1)[-1][0]
        p2 = net.forward_recycle(codes, sides, R=2)[-1][0]
        for j, (codes_i, side_i, mask, tgt, promo) in enumerate(chunk):
            s1 = p1[j].reshape(-1) + torch.tensor(mask).to(DEV)
            s2 = p2[j].reshape(-1) + torch.tensor(mask).to(DEV)
            lp1 = torch.log_softmax(s1, 0); lp2 = torch.log_softmax(s2, 0)
            top2 = torch.topk(lp1, 2)
            margin = float(top2.values[0] - top2.values[1])
            m1 = int(s1.argmax()); m2 = int(s2.argmax())
            def hit(m):
                fr, to = m // 64, m % 64
                if promo and (fr, to) == (tgt // 64, tgt % 64):
                    return True
                return m == tgt
            stats.append((margin, hit(m1), hit(m2), m1 != m2))

import numpy as np
rows = np.array(stats)
ch = rows[:, 3] == 1
print(f"n={len(rows)}  changed: {int(ch.sum())} ({100*ch.mean():.1f}%)")
print(f"target-match on CHANGED moves: RP1 {100*rows[ch,1].mean():.1f}%  RP2 {100*rows[ch,2].mean():.1f}%")
# gating simulation at the margin threshold: run RP2 only when margin < tau
for tau in [0.5, 1.0, 1.5, 2.0]:
    gated = rows[:, 0] < tau
    spend = 1 + gated.mean()          # avg passes per position
    # gated positions get RP2 move; ungated keep RP1 move
    hit = np.where(gated, rows[:, 2], rows[:, 1]).mean()
    print(f"tau={tau}: RP2 on {100*gated.mean():.0f}% of positions ({spend:.2f}x compute) -> target-match {100*hit:.1f}% (always-RP2: {100*rows[:,2].mean():.1f}%, always-RP1: {100*rows[:,1].mean():.1f}%)")
