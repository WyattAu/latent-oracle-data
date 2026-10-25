"""Convert ChessBench .bagz action-value records to engine-format shards.

Record: (fen, sf_move_uci, win_prob). The move becomes the BC policy target;
win_prob (DeepNoc mover-POV) becomes a white-POV wdl for the value head.
No python-chess in the hot loop -- manual FEN parse (~10x faster).
Writes ONE streaming output shard (fixed 64B records, LOSH v1 header).
"""
import argparse
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from chessbench_data import read_bagz  # noqa: E402
from format import REC_DTYPE, MAGIC, VERSION  # noqa: E402

PIECE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
         "p": 9, "n": 10, "b": 11, "r": 12, "q": 13, "k": 14}
PROMO = {"n": 0, "b": 1, "r": 2, "q": 3}
CASTLE_BIT = {"Q": 2, "K": 1, "q": 8, "k": 4}


def parse_fen(fen):
    parts = fen.split()
    board_part, side, castle_part, ep_part = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1
    codes = np.zeros(64, dtype=np.uint8)
    # FEN board part starts at RANK 8 (a8); sq = file + 8*rank, a1=0
    rank = 7
    file = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            codes[file + 8 * rank] = PIECE[ch]
            file += 1
    b = codes[0::2] | (codes[1::2] << np.uint8(4))  # byte j = sq2j | sq2j+1<<4
    packed = b.view("<u4") if b.dtype == np.uint8 else b.astype(np.uint8).view("<u4")
    side_i = 0 if side == "w" else 1
    rights = 0
    for ch in castle_part:
        rights |= CASTLE_BIT.get(ch, 0)
    ep = 255
    if ep_part != "-":
        ep = (ord(ep_part[0]) - 97) + 8 * (int(ep_part[1]) - 1)
    return (codes, packed, side_i, rights, ep,
            min(halfmove, 255), min(fullmove, 65535))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max records (0=all)")
    args = ap.parse_args()

    out = open(args.out, "wb")
    out.write(struct.pack("<IHHQ", MAGIC, VERSION, 16, 0))  # hdr const 16, u64 count
    count = 0
    buf = []
    from chessbench_data import decode_av_record
    for bag in args.bags:
        for raw in read_bagz(bag):
            fen, move, wp = decode_av_record(raw)
            codes, packed, side, rights, ep, hm, fm = parse_fen(fen)
            mv = move.strip()
            frm = (ord(mv[0]) - 97) + 8 * (int(mv[1]) - 1)
            to = (ord(mv[2]) - 97) + 8 * (int(mv[3]) - 1)
            promo = PROMO[mv[4]] if len(mv) > 4 else 0
            # DeepNoc win_prob is mover-POV -> white-POV wdl. Only an
            # expected score is available, so split the residual mass over
            # draw/loss evenly — a proper (w,d,l) triple that sums to 1.
            # The old (pw, 0, 0) collapsed the value head to always-win
            # (CE -> 0 signature) and poisoned trunk features (2026-10-25).
            pw = min(max(wp if side == 0 else 1.0 - wp, 0.0), 1.0)
            rest = (1.0 - pw) * 0.5
            wdl = (pw, rest, rest)
            rec = np.zeros(1, dtype=REC_DTYPE)
            rec[0]["side"] = side
            rec[0]["castling"] = rights
            rec[0]["ep"] = ep
            rec[0]["halfmove"] = hm
            rec[0]["fullmove"] = fm
            rec[0]["n_targets"] = 1
            rec[0]["board"] = packed
            rec[0]["eval_cp"] = 0
            rec[0]["targets"][0:3] = [frm, to, promo if promo else 255]
            rec[0]["wdl"] = wdl
            buf.append(rec)
            count += 1
            if len(buf) >= 65536:
                out.write(b"".join(r.tobytes() for r in buf))
                buf = []
            if args.limit and count >= args.limit:
                break
        if args.limit and count >= args.limit:
            break
    if buf:
        out.write(b"".join(r.tobytes() for r in buf))
    out.seek(8)
    out.write(struct.pack("<Q", count))
    out.close()
    print(f"wrote {count} records -> {args.out}")


if __name__ == "__main__":
    main()
