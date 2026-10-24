#!/bin/bash
# Loop-4 slice: full xattn-head AV run, chained behind the label-depth match.
# Controlled comparison vs av_s6 (pooled, 11.0% puzzles): same 20M records,
# same 12k steps, same batch, same init (av_s4). Only the head differs.
# Fires when the d16bc match's txt contains "Finished match".
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TXT=$DATA/sprt/av-d16bc-vs-s1.txt
LOG=$DATA/xattn_scale.log

while ! grep -q "Finished match" "$TXT" 2>/dev/null; do
  sleep 300
done
echo "[$(date '+%m-%d %H:%M')] d16bc match finished; starting xattn scale run" >> "$LOG"

mkdir -p $DATA/runs/chessbench_av_xattn
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/chessbench_av.py \
  --bags $DATA/chessbench/action_value-00000-of-02148_data.bag \
         $DATA/chessbench/action_value-00001-of-02148_data.bag \
  --out $DATA/runs/chessbench_av_xattn \
  --head-style xattn \
  --init-from $DATA/runs/chessbench_av/av_s4.pt \
  --steps 12000 --batch 384 --record-limit 20000000 \
  --export-every 2000 >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] xattn scale run done" >> "$LOG"
