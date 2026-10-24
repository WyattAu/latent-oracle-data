#!/bin/bash
# Loop-4 control #3: from-scratch AV training (no BC warm-start).
# Same 20M records, 12k steps, batch, pooled head as av_s6 (11.0%).
# Fires when the 26k-step compute-matched run exits.
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
LOG=$DATA/av_scratch.log
while pgrep -f "chessbench_av.py" > /dev/null; do sleep 300; done
echo "[$(date '+%m-%d %H:%M')] 10m run exited; starting from-scratch run" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
  $DATA/../dev/src/github.com/WyattAu/latent-oracle-data/trainer/chessbench_av.py \
  --bags $DATA/chessbench/action_value-00000-of-02148_data.bag \
         $DATA/chessbench/action_value-00001-of-02148_data.bag \
  --out $DATA/runs/chessbench_av_scratch \
  --steps 12000 --batch 384 --record-limit 20000000 \
  --export-every 2000 >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] from-scratch run done" >> "$LOG"
