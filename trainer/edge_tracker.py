#!/usr/bin/env python3
"""Game-edge tracking: policy-weighted value regret in calibrated probability
space (docs/RESEARCH-EDGE.md, "EVR").

Subcommands:
  calibrate --pgns GLOB --games N --out calib.npz
      raw wdl-mean -> empirical score via PAV isotonic on engine-vs-engine
      games with known outcomes.
  evr --pgns GLOB --games-per N --calib calib.npz
      per-class per-move EVR / oracle-regret / drift, plus outcome AUC.
"""
from __future__ import annotations
import argparse, glob, math, sys
import numpy as np
import torch
import chess
import chess.pgn

sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet


def load_net(pt="/home/wyatt/data/runs/bc/net_e1.pt"):
    torch.set_num_threads(4)
    net = ChessNet(256, 8, 8, 1024, 128, gab=False, v3=False).eval()
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    return net


def board_codes(board: chess.Board) -> np.ndarray:
    codes = np.zeros(64, dtype=np.int64)
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            codes[sq] = p.piece_type + (0 if p.color else 8)
    return codes


def net_eval(net, batch_codes, batch_sides):
    """(w,d,l) mover-POV probabilities + mean score, list order preserved."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(batch_codes), 512):
            codes = torch.tensor(np.stack(batch_codes[i:i+512])).long()
            side = torch.tensor(batch_sides[i:i+512]).long()
            _, _, wdl = net(codes, side)
            outs.append(torch.softmax(wdl, dim=1).numpy())
    return np.concatenate(outs)


def evaluate_game(net, board_plies):
    """Returns per-ply dict list: v_s, q_played, vbar, oracle, entropy."""
    # gather all inputs
    codes_all, sides_all, tags = [], [], []   # tags: (ply, kind, uci)
    for t, (board, move) in enumerate(board_plies):
        side = 0 if board.turn else 1
        codes_all.append(board_codes(board)); sides_all.append(side)
        tags.append((t, "s", None))
        for m in board.legal_moves:
            board.push(m)
            codes_all.append(board_codes(board))
            sides_all.append(0 if board.turn else 1)
            board.pop()
            tags.append((t, "a", m.uci()[:4]))
    wdl = net_eval(net, codes_all, sides_all)          # (N,3)
    mean = wdl[:, 0] + 0.5 * wdl[:, 1]                 # mover-POV score
    # scatter back
    per_ply = [dict() for _ in board_plies]
    for k, (t, kind, uci) in enumerate(tags):
        if kind == "s":
            per_ply[t]["v_s"] = mean[k]
            per_ply[t]["wdl_s"] = wdl[k]
        else:
            per_ply[t].setdefault("q", {})[uci] = mean[k]
    out = []
    for t, (board, move) in enumerate(board_plies):
        d = per_ply[t]
        uci = move.uci()[:4]
        qs = np.array(list(d["q"].values()))
        pi = np.exp(qs - qs.max()); pi /= pi.sum()      # policy-weighted EV
        q_played = d["q"].get(uci)
        if q_played is None:
            q_played = 1.0 - (0.0 if board.turn else 1.0)  # shouldn't happen
        vbar = float(pi @ qs)
        out.append(dict(v_s=d["v_s"], q=q_played, vbar=vbar,
                        oracle=float(qs.max() - q_played),
                        evr=vbar - q_played,
                        entropy=float(-(pi * np.log(pi + 1e-12)).sum()),
                        wdl_s=d["wdl_s"]))
    return out


def game_positions(pgn_path, max_games=10**9):
    """Yield (white, black, result_white_score, [(board, move) plies])."""
    with open(pgn_path, errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                return
            res = g.headers.get("Result", "*")
            ws = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res, None)
            if ws is None:
                continue
            plies = []
            b = g.board()
            for mv in g.mainline_moves():
                plies.append((b.copy(stack=False), mv))
                b.push(mv)
            yield g.headers.get("White", "?"), g.headers.get("Black", "?"), ws, plies


def pav_isotonic(raw, outcome):
    """PAV isotonic regression; returns (thresholds, values) step model."""
    order = np.argsort(raw)
    r = np.asarray(raw)[order]; y = np.asarray(outcome, dtype=float)[order]
    blocks = [[y[i], 1, r[i], r[i]] for i in range(len(r))]  # [sum, n, lo, hi]
    i = 0
    stack = []
    for blk in blocks:
        stack.append(blk)
        while len(stack) > 1 and stack[-2][0] / stack[-2][1] > stack[-1][0] / stack[-1][1]:
            b2 = stack.pop(); b1 = stack.pop()
            stack.append([b1[0] + b2[0], b1[1] + b2[1], b1[2], b2[3]])
    th = [b[3] for b in stack[:-1]]
    vals = [b[0] / b[1] for b in stack]
    return np.array(th), np.array(vals)


def apply_isotonic(raw, th, vals):
    idx = np.searchsorted(th, raw, side="left")
    return np.array([vals[min(i, len(vals) - 1)] for i in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "evr"])
    ap.add_argument("--pgns", required=True, help="glob")
    ap.add_argument("--games", type=int, default=800)
    ap.add_argument("--per", type=int, default=60, dest="per")
    ap.add_argument("--calib", default="/home/wyatt/data/edge_calib.npz")
    ap.add_argument("--out", default="/home/wyatt/data/edge_report.json")
    args = ap.parse_args()
    net = load_net()
    files = sorted(glob.glob(args.pgns))
    if args.cmd == "calibrate":
        raw, out = [], []
        done = 0
        import os
        for path in files:
            if done >= args.games: break
            for w, b, ws, plies in game_positions(path):
                if done >= args.games: break
                if not plies: continue
                plies = plies[::2]                     # every 2nd ply suffices
                codes = [board_codes(bd) for bd, _ in plies]
                sides = [0 if bd.turn else 1 for bd, _ in plies]
                wdl = net_eval(net, codes, sides)
                mean = wdl[:, 0] + 0.5 * wdl[:, 1]
                for k, (bd, mv) in enumerate(plies):
                    mover_white = bd.turn
                    r = ws if mover_white else 1.0 - ws
                    raw.append(mean[k]); out.append(r)
                done += 1
                if done % 100 == 0:
                    print(f"calib {done}/{args.games}", flush=True)
        th, vals = pav_isotonic(raw, out)
        cal = apply_isotonic(raw, th, vals)
        ece_before = float(np.mean(np.abs(np.array(raw) - np.array(out))))
        ece_after = float(np.mean(np.abs(cal - np.array(out))))
        np.savez(args.calib, th=th, vals=vals, ece_before=ece_before, ece_after=ece_after,
                 n=len(raw))
        print(f"calibration: n={len(raw)} ECE before={ece_before:.4f} after={ece_after:.4f}")
    else:
        cal = np.load(args.calib)
        th, vals = cal["th"], cal["vals"]
        classes = {}   # name -> list of per-game dicts
        for path in files:
            count = {}
            for w, b, ws, plies in game_positions(path, args.per):
                for name in (w, b):
                    if count.get(name, 0) >= args.per: continue
                if not plies: continue
                res = evaluate_game(net, plies)
                # calibrate v_s and q through the isotonic map
                for d in res:
                    d["v_s_c"] = float(apply_isotonic([d["v_s"]], th, vals)[0])
                    d["q_c"] = float(apply_isotonic([d["q"]], th, vals)[0])
                for name in (w, b):
                    if count.get(name, 0) >= args.per: continue
                    count[name] = count.get(name, 0) + 1
                    own = name
                    dd = []
                    for k, d in enumerate(res):
                        mover_white = plies[k][0].turn
                        if (mover_white and own == w) or ((not mover_white) and own == b):
                            dd.append(d)
                    evr = [d["evr"] for d in dd]
                    orc = [d["oracle"] for d in dd]
                    res_mover = ws if own == w else 1.0 - ws
                    classes.setdefault(name, []).append(dict(
                        evr=float(np.mean(evr)), oracle=float(np.mean(orc)),
                        result=res_mover, nplies=len(dd)))
        # report
        print(f"{'class':<28}{'n':>5}{'EVR/move':>10}{'oracle/move':>13}{'score':>7}")
        rows = []
        for name in sorted(classes, key=lambda n: -np.mean([g["evr"] for g in classes[n]])):
            gs = classes[name]
            rows.append((name, len(gs), float(np.mean([g["evr"] for g in gs])),
                         float(np.mean([g["oracle"] for g in gs])),
                         float(np.mean([g["result"] for g in gs]))))
            print(f"{name:<28}{len(gs):>5}{rows[-1][2]:>10.4f}{rows[-1][3]:>13.4f}{rows[-1][4]:>7.3f}")
        # outcome AUC: feature = opponent evr - own evr  (higher = better)
        feats, labels = [], []
        for name, gs in classes.items():
            for g in gs:
                pass
        # pairwise across classes: build per-game (feature, result) only where
        # opponent class known — use games file-by-file instead: approximate
        # with per-class mean separation (reported above) and a pooled AUC
        # over all games with feature = -own evr, label = result
        allf, alll = [], []
        for name, gs in classes.items():
            for g in gs:
                allf.append(-g["evr"]); alll.append(g["result"])
        allf = np.array(allf); alll = np.array(alll)
        wins = allf[alll == 1.0]; losses = allf[alll == 0.0]
        draws = allf[alll == 0.5]
        # AUC win-vs-loss (Mann-Whitney)
        import itertools
        gt = sum(1 for x in wins for y in losses if x > y) + 0.5 * sum(1 for x in wins for y in losses if x == y)
        auc = gt / max(1, len(wins) * len(losses))
        print(f"AUC(-ownEVR -> win vs loss): {auc:.4f}  (n_w={len(wins)} n_l={len(losses)} n_d={len(draws)})")


if __name__ == "__main__":
    main()
