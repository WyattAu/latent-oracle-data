#!/bin/bash
# AMZ offline pilot queue (RESEARCH-NOVEL2.md N5, ladder step 1).
# Waits for the recovery_seq chain (GPU free), then computes amortized-
# minimax targets for 50k positions with BC-v1's value head and fine-tunes.
# Ladder gate: puzzle suite >= +1.5 points over the base net.
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
LOG=$DATA/amz_pilot.log
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "=== AMZ pilot queued (waits for recovery_seq chain) ==="
while pgrep -f "recovery_seq.sh" > /dev/null; do sleep 600; done
while [ "$(gpu_free_mb)" -lt 3600 ]; do sleep 300; done
log "GPU free — starting AMZ target computation (50k positions)"

NET=$DATA/runs/bc_v1/net_e2.pt
[ -f $DATA/runs/bc_v1/net_best.bin.pt ] && NET=$DATA/runs/bc_v1/net_best.bin.pt
$PY "$TR/amz_pilot.py" \
  --net $NET \
  --shard $DATA/shards/bc_v1_combined.shard \
  --out $DATA/runs/amz_pilot \
  --positions 50000 --epochs 1 --batch 256 --lr 1e-4 >> "$LOG" 2>&1

[ -f "$DATA/runs/amz_pilot/amz_e0.pt" ] || { log "FATAL: AMZ pilot failed"; exit 1; }
log "pilot trained — puzzle eval vs base:"

$PY "$TR/score_puzzles.py" --net $DATA/runs/bc_v1/net_e2.pt \
  --puzzles $DATA/puzzles_500.jsonl >> "$LOG" 2>&1
$PY "$TR/score_puzzles.py" --net $DATA/runs/amz_pilot/amz_e0.pt \
  --puzzles $DATA/puzzles_500.jsonl >> "$LOG" 2>&1
log "=== AMZ pilot complete — gate: AMZ >= base + 1.5 puzzle points ==="
