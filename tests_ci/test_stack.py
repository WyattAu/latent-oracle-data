"""CI test suite for the latent-oracle-data trainer stack.

Run: python -m pytest tests_ci/ -q  (CPU, ~1 min)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainer"))


def test_format_roundtrip(tmp_path):
    from format import Sample, count_records, iter_records
    import struct
    # build a minimal shard: header + 2 records of 64 bytes
    from format import REC_DTYPE
    rec_dtype = REC_DTYPE
    assert rec_dtype.itemsize == 64, rec_dtype.itemsize
    r0 = np.zeros(2, dtype=rec_dtype)
    r0[0]["side"] = 0
    r0[0]["eval_cp"] = -123
    r0[0]["board"] = np.frombuffer(b"\x12" * 32, dtype="<u4")
    r0[0]["wdl"] = (1.0, 0.0, 0.0)
    path = tmp_path / "mini.shard"
    with open(path, "wb") as f:
        f.write(struct.pack("<IHH", 0x48534F4C, 1, 16) + b"\x00" * 8)  # 16-byte header
        f.write(r0.tobytes())
    assert count_records(str(path)) == 2
    recs = list(iter_records(str(path)))
    assert len(recs) == 2
    assert recs[0].eval_cp == -123
    assert recs[0].side == 0
    # board is 8x u32 little-endian = 32 bytes, nibble-packed squares
    assert recs[0].board_codes[0] == 0x2 and recs[0].board_codes[1] == 0x1


def test_gab_buckets_mirror_invariant():
    from model import gab_bucket_table
    t = gab_bucket_table()
    M = torch.tensor([sq ^ 7 for sq in range(64)])  # file mirror
    assert torch.equal(t, t[M][:, M]), "GAB buckets must be file-mirror invariant"


def test_gab_zero_init_identity():
    from model import ChessNet
    torch.manual_seed(0)
    m1 = ChessNet(d=32, layers=1, heads=2, dff=32, dpol=16, gab=False)
    m2 = ChessNet(d=32, layers=1, heads=2, dff=32, dpol=16, gab=True)
    m2.load_state_dict(m1.state_dict(), strict=False)
    codes = torch.randint(0, 15, (2, 64))
    side = torch.randint(0, 2, (2,))
    with torch.no_grad():
        s1, _, _ = m1(codes, side)
        s2, _, _ = m2(codes, side)
    assert (s1 - s2).abs().max().item() == 0.0, "zero-init GAB must be exact no-op"


def test_forward_recycle_shapes_and_grad():
    from model import ChessNet
    torch.manual_seed(0)
    m = ChessNet(d=32, layers=2, heads=2, dff=32, dpol=16)
    codes = torch.randint(0, 15, (2, 64))
    side = torch.randint(0, 2, (2,))
    passes = m.forward_recycle(codes, side, R=2)
    assert len(passes) == 2
    s1, p1, w1 = passes[0]
    s2, p2, w2 = passes[1]
    assert s1.shape == s2.shape == (2, 64, 64)
    (s2.sum() + 0.1 * s1.sum()).backward()  # grads flow through both passes


def test_muon_step_reduces_loss():
    from muon import Muon, split_params_for_muon
    from model import ChessNet
    torch.manual_seed(0)
    net = ChessNet(d=32, layers=1, heads=2, dff=32, dpol=16)
    mu, ad = split_params_for_muon(net)
    assert len(mu) > 0 and len(ad) > 0
    opt = Muon(mu, lr=0.02)
    opt2 = torch.optim.AdamW(ad, lr=1e-3)
    x = torch.randint(0, 15, (4, 64))
    y = torch.randn(4, 64, 64)
    first = None
    for _ in range(20):
        s, _, _ = net(x, torch.zeros(4, dtype=torch.long))
        loss = ((s - y) ** 2).mean()
        if first is None:
            first = loss.item()
        for o in (opt, opt2):
            o.zero_grad()
        loss.backward()
        for o in (opt, opt2):
            o.step()
    assert loss.item() < first, "Muon+AdamW must reduce toy loss"


def test_rct_loss_finite():
    from model import ChessNet
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = ChessNet(d=32, layers=2, heads=2, dff=32, dpol=16)
    codes = torch.randint(0, 15, (2, 64))
    side = torch.randint(0, 2, (2,))
    passes = m.forward_recycle(codes, side, R=3)
    final_logp = F.log_softmax(passes[-1][0].reshape(2, -1), dim=-1)
    rct = passes[0][0].new_zeros(())
    for ps, _, _ in passes[:-1]:
        lp = F.log_softmax(ps.reshape(2, -1), dim=-1)
        rct = rct + F.kl_div(lp, final_logp, log_target=True, reduction="batchmean")
    assert torch.isfinite(rct), "RCT loss must be finite"


def test_diffusion_tokenizer_roundtrip():
    import train_diffusion as td
    import chess
    board = chess.Board()
    codes = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            codes[sq] = td.SYM2CODE[p.symbol()]
    toks = td.encode_state(codes, 0, 15, 255)
    assert len(toks) == td.STATE_LEN
    # rank-1 = R N B Q K B N R
    base = len(td.SPECIALS)
    assert [toks[i] for i in range(8)] == [base + 4, base + 2, base + 3, base + 5,
                                           base + 6, base + 3, base + 2, base + 4]
    # move tokens round-trip
    mt = td.move_token(12, 28, 0)
    assert td.MOVE_BASE <= mt < td.MOVE_BASE + 4096
    assert mt == td.MOVE_BASE + 12 * 64 + 28
    # promo tokens live after moves
    pt = td.move_token(12, 28, 4)
    assert pt >= td.MOVE_BASE + 4096


def test_diffusion_black_pieces_and_vocab_range():
    import train_diffusion as td
    import numpy as np
    import chess
    board = chess.Board()
    codes = np.zeros(64, dtype=np.uint8)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            codes[sq] = td.SYM2CODE[p.symbol()]
    toks = td.encode_state(codes, 1, 15, 255)
    inv = {v: k for k, v in td.VOCAB.items()}
    assert "".join(inv[t] for t in toks[56:64]) == "rnbqkbnr"
    assert "".join(inv[t] for t in toks[48:56]) == "pppppppp"
    assert inv[toks[64]] == "sb"
    assert max(td.VOCAB.values()) < td.V
    assert len(td.VOCAB) == td.V


def test_gumbel_top_k_bounds_and_uniqueness():
    import grpo_train as g
    torch.manual_seed(0)
    logits = torch.randn(4, 100)
    idx = g.gumbel_top_k(logits, 8)
    assert idx.shape == (4, 8)
    for row in idx:
        assert row.min() >= 0 and row.max() < 100
        assert len(set(row.tolist())) == 8, "gumbel-top-k must return unique indices"


def test_grpo_reward_clipping_math():
    for cp, want in [(30000, 1.0), (-30000, -1.0), (150, 0.5), (0, 0.0), (-600, -1.0)]:
        r = max(-1.0, min(1.0, cp / 300))
        assert abs(r - want) < 1e-9
