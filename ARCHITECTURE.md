# Architecture — latent-oracle-data

Data pipeline + research tooling for the latent-oracle searchless chess
engine. See the engine repo's ARCHITECTURE.md for the inference side and
docs/SPEC-*.md / ADR-* for decisions.

## Components

```
src/            Rust pipeline (lo-data binary)
  main.rs         CLI dispatch: shard | label | masks | openings | info
  pgn.rs          PGN/zst streaming parser (Elo/time-control filters)
  shard.rs        64-byte fixed-record shard format v1 (spec: README)
  label.rs        Stockfish multipv labeling (persistent engine pool)
  masks.rs        legal-move mask sidecars (precomputed for the trainer)
  openings.rs     EPD book extraction (SPRT pools)
  worker.rs       parallel worker pool
  zobrist.rs      zobrist keys (matches the engine's splitmix64 scheme)

trainer/        Python research tooling (flat modules by design; see
                pyproject.toml note before restructuring)
  model.py        ChessNet — the architecture contract (bit-mirrored by the
                  engine's C++ inference; do not change one side alone)
  train.py        BC/AV/distillation trainer (mask sidecars, TB sidecars,
                  RCT, QAT, EMA, mirror aug, decisive weighting, Muon, WSD)
  muon.py         Muon optimizer (orthogonalized momentum)
  export_int8.py  LOQW INT8 blob exporter (calibration + v2 GAB + v3 tail)
  loqw_sim.py     quantized forward simulation — parity reference (ADR-005)
  train_diffusion.py / infer_diffusion.py   DiffuSearch pilot (E: future-token
                  discrete diffusion; samples built from shard pairs)
  amz_pilot.py    amortized-minimax targets (RESEARCH-NOVEL2 N5)
  grpo_train.py   RL fine-tuning (Gumbel-top-K groups, SF-reward, PPO+KL)
  make_puzzles.py / score_puzzles.py   fixed 500-position triage suite
  make_tb_labels.py  Syzygy ground-truth endgame labels (E1)
  mine_blindspots.py  surprise mining for focused replay (A1)
  conversion_metrics.py  E4 endgame-conversion metrics from SPRT PGNs
  fuzz_movegen.py movegen differential fuzz vs python-chess
  random_openings.py  off-distribution SPRT book generator
  sprt_pentanomial.py  pair-level SPRT statistics
  export_parity.py  FP32 parity harness (engine contract check)

tests_ci/       CI gate: format round-trip, GAB identity, recycle grads,
                Muon, RCT, diffusion tokenizer (black pieces + vocab range)
```

## Contracts

- **Shard v1**: 64-byte records, "LOSH" LE header — trainer/format.py and
  shard.rs must agree bit-for-bit.
- **LOQW blob**: quantized net layout — export_int8.py (writer), netq.cpp
  (reader), loqw_sim.py (parity reference). Single source of truth: the
  blob itself.
- **LONW blob**: fp32 net layout — model.py export_blob (writer), net.cpp
  (reader); v1 ⊂ v2(GAB) ⊂ v3(tail) strict-prefix versions.
- **Parity**: any inference change ships with a harness run (ADR-005).

## Invariants (standing policy, RESEARCH-SYSTEMS §4.5)

- Bulk dataset builds validate a small slice (token range, decode
  round-trip, one finite training step) before the full scan.
- Differential fuzz against a reference implementation for movegen.
- Experiment verdicts are pre-registered (kill criteria) in the engine
  repo's docs/RESULTS.md ledger.
