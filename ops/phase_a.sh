#!/bin/bash
# Phase-A pipeline for latent-oracle:
#   wait for Lichess download -> shard 20M BC -> BC training ->
#   label 1M checkpoint -> distilled-1M training -> label to 5M (resume) ->
#   distilled-5M training.
# Fully detached; progress in /home/wyatt/data/pipeline.log

DATA=/home/wyatt/data
LO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
SF=/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2
PY=/tmp/opencode/venv/bin/python
LOG=$DATA/pipeline.log
PGN=$DATA/lichess/lichess_db_standard_rated_2026-07.pgn.zst

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== Phase-A pipeline started (pid $$) ==="

# 1. wait for the download to finish
while pgrep -f "lichess/fetch.sh" > /dev/null; do sleep 60; done
log "download complete: $(stat -c%s "$PGN") bytes"

# 2. shard 20M BC positions (skipped when the completed shard already exists)
if [ -f $DATA/shards/bc.shard ] && [ "$(stat -c%s $DATA/shards/bc.shard)" -ge 1280000016 ]; then
  log "shard already present ($(stat -c%s $DATA/shards/bc.shard) bytes) — skipping sharding"
else
  log "sharding 20M positions..."
  $LO/target/release/lo-data shard --in "$PGN" --out $DATA/shards \
    --min-elo 2000 --min-ply 8 --max-ply 120 --max-positions 20000000 >> "$LOG" 2>&1
  [ -f $DATA/shards/bc.shard ] || { log "FATAL: sharding produced no output"; exit 1; }
  log "shard done: $(stat -c%s $DATA/shards/bc.shard) bytes"
fi

# 3. BC training (game-result WDL + played-move policy)
log "bc training..."
$PY $LO/trainer/train.py --shard $DATA/shards/bc.shard --out $DATA/runs/bc \
  --epochs 2 --batch 512 >> "$LOG" 2>&1
log "bc training done"

# 4. label 1M checkpoint
log "labeling 1M checkpoint..."
$LO/target/release/lo-data label --in $DATA/shards/bc.shard --out $DATA/shards/labeled_1m.shard \
  --sf "$SF" --depth 16 --multipv 3 --threads 5 --max-records 1000000 >> "$LOG" 2>&1
log "label 1M done"

# 5. distilled-1M training
log "distilled-1M training..."
$PY $LO/trainer/train.py --shard $DATA/shards/labeled_1m.shard --out $DATA/runs/dist1m \
  --epochs 2 --batch 512 >> "$LOG" 2>&1
log "distilled-1M done"

# 6. label to 5M (--resume skips the first 1M, already labeled)
log "labeling to 5M..."
$LO/target/release/lo-data label --in $DATA/shards/bc.shard --out $DATA/shards/labeled_5m.shard \
  --sf "$SF" --depth 16 --multipv 3 --threads 5 --max-records 5000000 --resume >> "$LOG" 2>&1
log "label 5M done"

# 7. distilled-5M training
log "distilled-5M training..."
$PY $LO/trainer/train.py --shard $DATA/shards/labeled_5m.shard --out $DATA/runs/dist5m \
  --epochs 2 --batch 512 >> "$LOG" 2>&1
log "distilled-5M done — PHASE A COMPLETE"
