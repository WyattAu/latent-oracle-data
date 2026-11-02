#!/bin/bash
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data/shards
R=/home/wyatt/data/runs/stack_av
LOG=/home/wyatt/data/stack_av.log
BAGS="$D/chessbench_av/action_value-00000-of-02148_data.bag $D/chessbench_av/action_value-00001-of-02148_data.bag"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
mkdir -p $R

INIT=/home/wyatt/data/runs/bc_marquee/e3/net_e0.pt
log "Stack: AV head on marquee trunk (init from $INIT)"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/chessbench_av.py \
  --bags $BAGS \
  --head-style pooled \
  --out $R \
  --init-from $INIT \
  --steps 98750 --batch 384 --lr 1e-4 \
  >> $LOG 2>&1 || { log "Stack AV training FAILED"; exit 1; }
log "Stack AV training done — net at $R/av_step98762.pt"
log "=== Stack experiment training complete ==="
