#!/bin/bash
# BC-at-scale chain: convert train shards 00000+00001 (~40M records) to one
# engine-format shard, then train the bc_v1 architecture from scratch on it.
# GPU work; runs alongside the CPU match pipeline.
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
LOG=$DATA/bc_avlabels.log
SHARD=$DATA/shards/chessbench_bc40m.shard
OUT=$DATA/runs/bc_avlabels40m

echo "[$(date '+%m-%d %H:%M')] converting 2 train shards" >> "$LOG"
nice -n 10 $PY $TR/bag_to_shard.py \
  --bags $DATA/chessbench/action_value-00000-of-02148_data.bag \
         $DATA/chessbench/action_value-00001-of-02148_data.bag \
  --out $SHARD >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] conversion done: $(ls -la $SHard 2>/dev/null | awk '{print $5}') bytes" >> "$LOG"

mkdir -p $OUT
echo "[$(date '+%m-%d %H:%M')] training from scratch, 1 epoch" >> "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $SHARD --out $OUT \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --batch 512 --lr 3e-4 >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] training done" >> "$LOG"
