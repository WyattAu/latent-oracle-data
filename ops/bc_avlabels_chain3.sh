#!/bin/bash
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
LOG=$DATA/bc_avlabels3.log
OUT=$DATA/runs/bc_avlabels40m
echo "[$(date '+%m-%d %H:%M')] training v2: fp32 + nan guard, shard+mask reused" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $DATA/shards/chessbench_bc40m.shard --out $OUT \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --batch 384 --lr 3e-4 --fp32 >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] training done" >> "$LOG"
