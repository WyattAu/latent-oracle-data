import sys, random
import numpy as np, torch, chess
sys.path.insert(0, "trainer")
from format import REC_DTYPE, MaskSidecar
from model import ChessNet

torch.set_num_threads(6)
DEV = "cpu"

A = "/home/wyatt/data/shards/bc_sub/slice1.full.shard"
AM = A + ".mask"
sc = MaskSidecar(AM)
f = open(A, "rb")
rng = random.Random(11)
blocks = [(1_000_000, 300), (75_000_000, 300)]
positions = []
for lo, n in blocks:
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
        t = rec["targets"]
        tgt = int(t[0]) * 64 + int(t[1])
        if mask[tgt] != 0: continue
        positions.append((codes, int(rec["side"]), mask, tgt))
print(f"{len(positions)} positions", flush=True)

net = ChessNet(256, 8, 8, 1024, 128).to(DEV).eval()
net.load_state_dict(torch.load("/home/wyatt/data/runs/bcpool_1ep/net_e0.pt",
                               map_location=DEV, weights_only=True))

rows = []
B = 256
with torch.no_grad():
    for i in range(0, len(positions), B):
        chunk = positions[i:i+B]
        codes = torch.tensor(np.stack([p[0] for p in chunk]).astype(np.int64)).to(DEV)
        sides = torch.tensor([p[1] for p in chunk]).to(DEV)
        p1 = net.forward_recycle(codes, sides, R=1)[-1][0]
        p2 = net.forward_recycle(codes, sides, R=2)[-1][0]
        for j, (codes_i, side_i, mask, tgt) in enumerate(chunk):
            s1 = p1[j].reshape(-1) + torch.tensor(mask).to(DEV)
            s2 = p2[j].reshape(-1) + torch.tensor(mask).to(DEV)
            lp1 = torch.log_softmax(s1, 0); lp2 = torch.log_softmax(s2, 0)
            top2 = torch.topk(lp1, 2)
            margin = float(top2.values[0] - top2.values[1])
            ent = float(-(lp1.exp() * lp1).sum())
            mv1 = int(s1.argmax()); mv2 = int(s2.argmax())
            ce2 = float(-lp2[tgt])
            rows.append((margin, ent, mv1 != mv2, float(lp1[tgt]), ce2))
import math
rows = np.array([(r[0], r[1], float(r[2]), r[3], r[4]) for r in rows])
changed = rows[:, 2] == 1
print(f"n={len(rows)}  RP1->RP2 move changes: {int(rows[:,2].sum())} ({100*changed.mean():.1f}%)")
for name, col in [("top1 margin", rows[:,0]), ("policy entropy", rows[:,1])]:
    lo = rows[~changed][:, col.idx] if False else None
    a = rows[~changed][:, 0 if col is rows[:,0] else 1]
    b_ = rows[changed][:, 0 if col is rows[:,0] else 1]
    print(f"{name}: unchanged mean={a.mean():.4f}  changed mean={b_.mean():.4f}")
# quantile table: move-change rate by margin/entropy quintile
for name, colidx in [("margin", 0), ("entropy", 1)]:
    qs = np.quantile(rows[:, colidx], [0.2, 0.4, 0.6, 0.8])
    ids = np.digitize(rows[:, colidx], qs)
    line = []
    for q in range(5):
        sel = ids == q
        line.append(f"{100*rows[sel,2].mean():.1f}%")
    print(f"change-rate by {name} quintile (low->high): {' '.join(line)}")
