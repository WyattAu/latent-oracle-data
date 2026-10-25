#!/usr/bin/env python3
"""Convert DeepMind ChessBench behavioral_cloning .bag (fen, move) into a
LOSH BC shard. Human/GM moves are the natural BC target — the AV bags'
deep-Stockfish single moves have high conditional entropy and fit ~12x
worse per sample (2026-10-25 audit). Value targets are neutral: the bag
carries no result; the engine plays policy argmax and ignores value.

Streaming via mmap: the 36GB bag cannot be read() into RAM.
"""
import argparse
import mmap
import struct
import sys

import numpy as np

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE
from bag_to_shard import parse_fen, PROMO
from chessbench_data import _read_str


def iter_bagz_mmap(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        (index_start,) = struct.unpack_from("<Q", mm, len(mm) - 8)
        num = (len(mm) - index_start) // 8 - 1
        prev = 0
        for i in range(num):
            (stop,) = struct.unpack_from("<q", mm, index_start + i * 8)
            yield mm[prev:stop]
            prev = stop


def decode_bc_record(buf):
    fen, pos = _read_str(buf, 0)
    move, pos = _read_str(buf, pos)
    return fen, move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    MAGIC = 0x48534F4C
    out = open(args.out, "wb")
    out.write(struct.pack("<IHHQ", MAGIC, 1, 16, 0))
    count = 0
    buf = []
    import chess
    n_ambig = 0
    for raw in iter_bagz_mmap(args.bag):
        fen, code = decode_bc_record(raw)
        # DeepMind's BC action space is factorized: from-rank (1-indexed
        # digit) + to-square (file+rank). Resolve against legal moves; skip
        # the ~14% of records where two legal moves match (their model
        # cannot express from-file either).
        codes, packed, side, rights, ep, hm, fm = parse_fen(fen)
        b = chess.Board(None)
        for sq, c in enumerate(codes):
            if c:
                pt = c if c < 8 else c - 8
                b.set_piece_at(sq, chess.Piece(pt, chess.WHITE if c < 8 else chess.BLACK))
        b.turn = chess.WHITE if side == 0 else chess.BLACK
        if ep < 64:
            b.ep_square = ep
        cr = 0
        if rights & 1 and b.piece_at(chess.H1) and b.piece_at(chess.H1).piece_type == chess.ROOK and b.piece_at(chess.E1): cr |= chess.BB_H1
        if rights & 2 and b.piece_at(chess.A1) and b.piece_at(chess.A1).piece_type == chess.ROOK and b.piece_at(chess.E1): cr |= chess.BB_A1
        if rights & 4 and b.piece_at(chess.H8) and b.piece_at(chess.H8).piece_type == chess.ROOK and b.piece_at(chess.E8): cr |= chess.BB_H8
        if rights & 8 and b.piece_at(chess.A8) and b.piece_at(chess.A8).piece_type == chess.ROOK and b.piece_at(chess.E8): cr |= chess.BB_A8
        b.castling_rights = cr
        if b.is_game_over():
            continue
        want_rank = int(code[0]) - 1
        want_to = chess.parse_square(code[1:3])
        matches = [m for m in b.legal_moves
                   if m.from_square // 8 == want_rank and m.to_square == want_to]
        if len(matches) != 1:
            n_ambig += 1
            continue
        mv = matches[0]
        frm = mv.from_square
        to = mv.to_square
        promo = mv.promotion or 0
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
        rec[0]["wdl"] = (1/3, 1/3, 1/3)
        buf.append(rec)
        count += 1
        if len(buf) >= 65536:
            out.write(b"".join(r.tobytes() for r in buf)); buf = []
        if args.limit and count >= args.limit:
            break
    if buf:
        out.write(b"".join(r.tobytes() for r in buf))
    out.seek(8)
    out.write(struct.pack("<Q", count))
    out.close()
    print(f"wrote {count} records -> {args.out} (skipped ambiguous/terminal: {n_ambig})", flush=True)


if __name__ == "__main__":
    main()
