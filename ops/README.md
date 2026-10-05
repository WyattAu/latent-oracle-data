# ops — autonomous experiment cascade

These are the orchestration scripts that drive the research chains. They were
previously unversioned (they lived only in `/home/wyatt/data`), which meant a
lost box lost the whole experiment calendar. Each is committed with the code it
drives; the deployed copies live in `$DATA` (`/home/wyatt/data`) and are the
ones that run.

## Cascade topology

```
label_mixed.sh      CPU: d10 4M + d16 1M labels (crash-safe: --resume,
                    --batch-records; every stage RAM-gated)
        │
        ├──► keep_best.sh      gate: SPRT every epoch of bc_v1 / bc_v1f /
        │                      dist1m_dw, rank by FINAL Elo, publish
        │                      net_best.bin + net_best.pt
        │            ├──► loopcd_triage.sh   recycling alpha grid
        │            └──► int8_sprt.sh       INT8 parity + match verdict
        │
        └──► av_phase.sh       GPU: AV stage1 (d10, Muon, WSD) → stage2
                               (d16, RCT+QAT) → FP32+INT8 blobs → A/B verdict
                                    └──► grpo_phase.sh   RL on the AV net

moonshot_queue.sh   GPU: DiffuSearch h=2 → a0-match gate → AMZ pilot →
                    puzzle gate → pre-scripted double-downs (0.25/0.40)
                    (waits only on recovery_seq + GPU, runs beside the above)
```

## Rules encoded in the scripts

- **RAM preflight** (`wait_for_ram`): the box is shared; another session took
  25+ GB and OOM-killed an 18 h labeling run on 2026-10-05. Stages wait for
  `MemAvailable` headroom instead of dying.
- **Atomic artifacts**: trainer saves go through `trainer/robust_io.py`
  (tmp + fsync + rename) so a kill never leaves a 0-byte file that poisons the
  next run.
- **Resumable labeling**: `lo-data label --resume --batch-records N` writes in
  durable batches; a partial output is a valid input prefix.
- **Skip guards**: every stage checks for its own output artifact, so a relaunch
  never repeats finished work.
- **Pre-registered gates**: thresholds live in the scripts (a0-match 0.25/0.40,
  puzzle +1.5 pts, Elo bounds), so verdicts are not re-litigated after seeing
  the numbers.

## Editing rules

- NEVER edit a script that is currently executing (bash re-reads by byte
  offset). Kill it, edit, relaunch — that is how the keep-best Elo fix landed.
- `sed`-edit live scripts only when the phase has not started and the script is
  blocked in a `while ... sleep` loop, and even then prefer kill/edit/relaunch.
