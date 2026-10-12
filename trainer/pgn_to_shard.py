"""Convert self-play PGNs into a training shard: winner-move BC data.

For every position where the SIDE TO MOVE is the game's winner, emit one
shard record whose policy target is the move actually played in the game,
with the WDL outcome in the wdl head. Draw games are skipped (no signal
about which moves to prefer). This is the data side of self-play outcome
learning: the searchless engine makes games cheap, so outcome supervision
is affordable where game-outcome RL was not.
"""
import argparse, struct
import chess
import numpy as np
import chess.pgn

REC_DTYPE = np.dtype([
    ("halfmove", "<u1"),
    ("fullmove", "<u2"),
    ("n_targets", "<u1"),
    ("pad", "<u1"),
    ("board", "<u4", (8,)),
    ("eval_cp", "<i2"),
    ("targets", "<u1", (9,)),
    ("wdl", "<f4", (3,)),
    ("pad2", "<u1"),
])
PIECE_CODE = {}  # python-chess piece_type/color -> shard code
for pt in range(1, 7):
    PIECE_CODE[(pt, chess.WHITE)] = pt
    PIECE_CODE[(pt, chess.BLACK)] = pt + 8
PROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}


def encode_board(board):
    """Exact mirror of format.py's decode:
    board_codes[sq] = (raw[sq >> 1] >> ((sq & 1) * 4)) & 0xF, with raw the
    32 bytes of the record's board field (8 little-endian u32 words)."""
    codes = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is not None:
            codes[sq] = PIECE_CODE[(pc.piece_type, pc.color)]
    raw = np.zeros(32, dtype=np.uint8)
    for sq in range(64):
        raw[sq >> 1] |= int(codes[sq]) << ((sq & 1) * 4)
    words = np.zeros(8, dtype="<u4")
    for byte_i in range(32):
        words[byte_i // 4] |= int(raw[byte_i]) << (8 * (byte_i % 4))
    return words


def encode_record(board, move, outcome_white_pov):
    codes = encode_board(board)
    side = 0 if board.turn == chess.WHITE else 1
    rights = 0
    cr = board.clean_castling_rights()
    if cr & chess.BB_A1: rights |= 2   # Q
    if cr & chess.BB_H1: rights |= 1   # K
    if cr & chess.BB_A8: rights |= 8   # q
    if cr & chess.BB_H8: rights |= 4   # k
    ep = board.ep_square if board.ep_square is not None else 255
    promo = PROMO.get(move.promotion, 0) if move.promotion else 0
    targets = np.zeros(9, dtype=np.uint8)
    targets[0:3] = [move.from_square, move.to_square, promo]
    wdl = np.zeros(3, dtype="<f4")
    if outcome_white_pov == 1:
        wdl[0] = 1.0
    elif outcome_white_pov == -1:
        wdl[2] = 1.0
    else:
        wdl[1] = 1.0
    rec = np.zeros(1, dtype=REC_DTYPE)
    rec[0]["halfmove"] = min(board.halfmove_clock, 255)
    rec[0]["fullmove"] = min(board.fullmove_number, 65535)
    rec[0]["n_targets"] = 1
    rec[0]["board"] = codes
    rec[0]["eval_cp"] = 0
    rec[0]["targets"] = targets
    rec[0]["wdl"] = wdl
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-draws", action="store_true",
                    help="keep positions from drawn games (wdl=draw); "
                    "default skips them")
    args = ap.parse_args()

    recs = []
    games = 0
    with open(args.pgn) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            res = game.headers["Result"]
            if res == "1-0":
                outcome = 1        # white won (white POV)
            elif res == "0-1":
                outcome = -1
            elif args.include_draws:
                outcome = 0
            else:
                continue
            games += 1
            board = game.board()
            for move in game.mainline_moves():
                mover_win = (board.turn == chess.WHITE and outcome == 1) or \
                            (board.turn == chess.BLACK and outcome == -1)
                if mover_win:
                    recs.append(encode_record(board, move, outcome))
                board.push(move)

    magic, version = 0x48534f4c, 1  # LOSH -- matches the labeler header exactly
    with open(args.out, "wb") as f:
        f.write(struct.pack("<IHH", magic, version, len(recs)))
        if recs:
            f.write(np.concatenate(recs).tobytes())
    print(f"games: {games}, records: {len(recs)} -> {args.out}")


if __name__ == "__main__":
    main()
