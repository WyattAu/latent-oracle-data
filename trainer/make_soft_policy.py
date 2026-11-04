#!/usr/bin/env python3
"""Soft SF-policy labels for the BC pool positions.

For each BC slice position: SF multipv=K at fixed nodes produces the top-K
moves with cp scores; softmax(cp/τ) gives the soft target distribution.
The BC trainer's policy CE uses these soft targets instead of the single
hard move — SF sharpness where positions demand it, learnability where
they don't.

Output: jsonl {fen_idx, soft: [[uci, p], ...], best_uci} per position.
The trainer reads these as CE targets (distribution over legal moves).
"""
import argparse, subprocess, json, sys, random, math
import numpy as np, chess
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--sf", default="/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2")
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--positions", type=int, default=0, help="0 = all in range")
    ap.add_argument("--nodes", type=int, default=200_000)
    ap.add_argument("--multipv", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sc = __import__("format").MaskSidecar(args.mask)
    sf = subprocess.Popen([args.sf], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1)
    sf.stdin.write("uci\nsetoption name MultiPV value %d\nisready\n" % args.multipv)
    sf.stdin.flush()
    for line in sf.stdout:
        if line.startswith("readyok"): break

    f = open(args.shard, "rb")
    RECSZ = 64
    rng = random.Random(args.seed)
    lo_s = max(args.lo, 0)
    hi_s = min(args.hi, 526_359_028)
    span = hi_s - lo_s
    if args.positions and args.positions < span:
        idxs = sorted(rng.sample(range(lo_s, hi_s), args.positions))
    else:
        idxs = list(range(lo_s, hi_s))

    out_f = open(args.out, "w")
    n = 0
    for i in idxs:
        f.seek(16 + i * RECSZ)
        rec = np.frombuffer(f.read(RECSZ), dtype=REC_DTYPE)[0]
        if rec["n_targets"] == 0: continue
        bdis = rec["board"].astype("<u4").view(np.uint8)
        codes = np.empty(64, dtype=np.uint8)
        codes[0::2] = bdis & 0x0F; codes[1::2] = bdis >> 4
        b = chess.Board(None)
        for sq, c in enumerate(codes):
            if c:
                pt = c if c < 8 else c - 8
                b.set_piece_at(sq, chess.Piece(pt, chess.WHITE if c < 8 else chess.BLACK))
        b.turn = chess.WHITE if int(rec["side"]) == 0 else chess.BLACK
        ep = int(rec["ep"])
        if ep < 64: b.ep_square = ep
        # castling rights from the record
        cr = int(rec["castling"])
        b.castling_rights = ((cr & 1) * chess.BB_H1 | (cr & 2) * chess.BB_A1 |
                             (cr & 4) * chess.BB_H8 | (cr & 8) * chess.BB_A8)
        if b.is_game_over(): continue

        sf.stdin.write(f"position fen {b.fen()}\ngo nodes {args.nodes}\n")
        sf.stdin.flush()
        # SF emits one info line per multipv index per depth: keep the LAST
        # (deepest) line per index, else intermediate depths pollute the mix.
        multipv = {}
        for line in sf.stdout:
            if line.startswith("bestmove"): break
            if " multipv " not in line:
                continue
            try:
                pv_idx = int(line.split(" multipv ")[1].split()[0])
                if " score cp " in line:
                    cp = int(line.split(" score cp ")[1].split()[0])
                elif " score mate " in line:
                    m = int(line.split(" score mate ")[1].split()[0])
                    cp = 10000 if m > 0 else -10000
                else:
                    continue
                parts = line.split(" pv ")
                pv = parts[1].split()[0] if len(parts) > 1 else None
                if pv:
                    multipv[pv_idx] = (cp, pv)
            except (ValueError, IndexError):
                continue
        if not multipv: continue
        ordered = [multipv[k] for k in sorted(multipv)]
        best_cp = ordered[0][0]
        # dedup moves (keep the max weight per move), then softmax-normalize
        weights = {}
        for cp, mv in ordered:
            w = math.exp(-(cp - best_cp) / args.temperature)
            if mv not in weights or w > weights[mv]:
                weights[mv] = w
        z = sum(weights.values()) or 1.0
        soft = sorted(((mv, round(w / z, 5)) for mv, w in weights.items()),
                      key=lambda x: -x[1])
        rec_out = {"shard_idx": i, "fen": b.fen(), "soft": soft,
                   "best": soft[0][0] if soft else None}
        out_f.write(json.dumps(rec_out) + "\n")
        n += 1
        if n % 10000 == 0:
            print(f"{n} positions labelled", flush=True)
        if args.positions and n >= args.positions:
            break
    out_f.close(); sf.stdin.write("quit\n"); sf.stdin.flush(); sf.terminate()
    f.close()
    print(f"wrote {n} soft-labelled positions -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
