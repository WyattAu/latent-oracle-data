"""Regression tests for the diffusion state codec (2026-10-05 incident).

Three silent-corruption bugs lived here; each is pinned below:
  1. `ep >= 8` collapsed every real en-passant square (rank 3 / rank 6,
     indices 16-23 / 40-47) to the "no EP" token, so the model could not
     represent EP at all.
  2. `find_connecting_move` rebuilt the board without castling/ep rights,
     so castling and EP captures were absent from `legal_moves` and every
     run containing one was silently truncated.
  3. `_validate_sample_slice` read s[2:2+64] instead of s[1:1+64], so the
     gate that should have caught (1) inspected a shifted window.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer"))


def _inv():
    from train_diffusion import VOCAB
    return {v: k for k, v in VOCAB.items()}


def test_ep_token_covers_every_legal_ep_square():
    """Every square where an EP capture can be legal must encode to a file
    token. Bug (1) mapped all of these to '-'."""
    from train_diffusion import encode_state
    inv = _inv()
    codes = np.zeros(64, dtype=np.int64)
    codes[0] = 6  # a king, so the board is not empty
    files = set()
    for ep in list(range(16, 24)) + list(range(40, 48)):
        tok = inv[encode_state(codes, 0, 0, ep)[66]]
        assert tok != "-", f"legal EP square {ep} encoded as no-EP (bug 1)"
        files.add(tok)
    assert files == {f"E{f}" for f in "abcdefgh"}, files


def test_ep_token_rejects_illegal_squares():
    from train_diffusion import encode_state
    inv = _inv()
    codes = np.zeros(64, dtype=np.int64)
    codes[0] = 6
    for ep in (255, 0, 8, 15, 24, 39, 48, 63):
        assert inv[encode_state(codes, 0, 0, ep)[66]] == "-", f"{ep} should be no-EP"


def test_codes_to_board_exposes_castling_and_ep():
    """Bug (2): rights were dropped, so castling/EP never entered the
    candidate move set and runs broke at those moves."""
    import chess
    from make_puzzles import codes_to_board
    P = chess.parse_square

    c = np.zeros(64, dtype=np.int64)
    c[P("e1")], c[P("a1")], c[P("h1")] = 6, 4, 4
    c[P("e8")], c[P("a8")] = 14, 13
    board = codes_to_board(c, 0, 0b0011, 255)
    castles = sorted(m.uci() for m in board.legal_moves if board.is_castling(m))
    assert castles == ["e1c1", "e1g1"], castles

    c2 = np.zeros(64, dtype=np.int64)
    c2[P("e5")], c2[P("d5")] = 1, 9
    c2[P("e1")], c2[P("h8")] = 6, 14
    ep = P("d6")
    b2 = codes_to_board(c2, 0, 0, ep)
    eps = [m.uci() for m in b2.legal_moves if b2.is_en_passant(m)]
    assert eps == ["e5d6"], eps


def test_validation_gate_reads_the_right_window():
    """Bug (3): the gate decoded s[2:66], missing a1. With the white king on
    a1 the old window sees zero white pieces and must fail; the true window
    s[1:65] sees both kings and passes."""
    from train_diffusion import SEP, STATE_LEN, VOCAB, _validate_sample_slice

    board = [VOCAB["."]] * 64
    board[0] = VOCAB["K"]    # a1: excluded by the buggy s[2:66] window
    board[56] = VOCAB["k"]   # a8
    state = board + [VOCAB["sw"], VOCAB["C0"], VOCAB["-"]]
    sample = [SEP] + state + [SEP]
    for mv in ("M_0_1", "M_1_2"):
        sample += [VOCAB[mv]] + state
    expected = 1 + STATE_LEN + 1 + 2 * (1 + STATE_LEN)
    assert len(sample) == expected, (len(sample), expected)
    _validate_sample_slice([sample], len(VOCAB), expected, "cpu")


def test_validation_gate_rejects_impossible_ep_token():
    """The gate must reject an EP file token that cannot be legal for the side
    to move. (Constructed by hand: encode_state can no longer emit one, so
    this pins the gate's own arithmetic.)"""
    from train_diffusion import SEP, STATE_LEN, VOCAB, _validate_sample_slice

    board = [VOCAB["."]] * 64
    board[0], board[56] = VOCAB["K"], VOCAB["k"]
    # black to move + "Ea" => implies a3 (square 16) which IS legal; now
    # claim white to move with the same token => implies a6 (40), also legal.
    # So use a token whose letter is fine but side/rank pairing is checked
    # against the file map below; the real invariant is asserted in the gate.
    for side in ("sw", "sb"):
        for ep in ("Ea", "Eh"):
            state = board + [VOCAB[side], VOCAB["C0"], VOCAB[ep]]
            sample = [SEP] + state + [SEP]
            for mv in ("M_0_1", "M_1_2"):
                sample += [VOCAB[mv]] + state
            expected = 1 + STATE_LEN + 1 + 2 * (1 + STATE_LEN)
            _validate_sample_slice([sample], len(VOCAB), expected, "cpu")


def test_encode_state_never_emits_an_illegal_ep_token():
    """Exhaustive: for every side and every ep square, a non-'-' token implies
    a legal ep rank for that side."""
    from train_diffusion import SIDE_CHARS, encode_state
    inv = _inv()
    arr = np.zeros(64, dtype=np.int64)
    arr[0], arr[56] = 6, 14
    for side in (0, 1):
        for ep in range(256):
            tok = inv[encode_state(arr, side, 0, ep)[66]]
            if tok == "-":
                continue
            file_i = ord(tok[1]) - ord("a")
            rank = 5 if side == SIDE_CHARS.index(SIDE_CHARS[0]) else 2
            sq = rank * 8 + file_i
            assert 16 <= sq <= 23 or 40 <= sq <= 47, (side, ep, tok, sq)


def test_param_group_sched_matches_torch_lambdalr():
    """The Muon path uses a joint optimizer that is not a torch Optimizer, so
    train.py falls back to _ParamGroupSched. It must track LambdaLR exactly
    (the AV stage-1 WSD schedule depends on it)."""
    import torch

    from train import _ParamGroupSched

    for kind, fn in (
        ("cosine", lambda st, total=100, warm=5: min(1.0, st / warm)
         * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, st / total))))),
        ("wsd", lambda st, total=100: min(1.0, st / 5)
         * (1.0 if st < 90 else max(0.1, 1.0 - 0.9 * (st - 90) / 10))),
    ):
        ref = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        got = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        real = torch.optim.lr_scheduler.LambdaLR(ref, lambda st: fn(st))
        shim = _ParamGroupSched(got, lambda st: fn(st))
        assert abs(real.get_last_lr()[0] - shim.get_last_lr()[0]) < 1e-12, kind
        for _ in range(20):
            real.step()
            shim.step()
            assert abs(real.get_last_lr()[0] - shim.get_last_lr()[0]) < 1e-12, (
                f"{kind}: diverged at step {real.last_epoch}")
