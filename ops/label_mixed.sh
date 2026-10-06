#!/bin/bash
# Mixed-depth labeling (KataGo-style cost cut, user decision 2026-10-04):
#   stage 1: 4M positions at depth 10 from bc_v1_combined.shard  (~1.5 d CPU)
#   stage 2: 1M positions at depth 16 from bc.shard               (~1 d CPU)
# AV training then runs sequentially: d10 pretrain -> d16 fine-tune.
set -u
DATA=/home/wyatt/data
LO=/home/wyatt/tools/chess/lo-data
SF=/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/label_mixed.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- memory preflight (2026-10-05: OOM kills took out the d10 labeler at
# 3.9M/4M records and the DiffuSearch sample build; another session on this
# box routinely eats 25+ GB). Wait for real headroom instead of dying.
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
ram_total_mb() { awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo; }
wait_for_ram() {
  local need=${1:-3000} waited=0
  while [ "$(ram_avail_mb)" -lt "$need" ]; do
    if [ $((waited % 1800)) -eq 0 ]; then
      log "ram wait: need ${need}MB, have $(ram_avail_mb)MB of $(ram_total_mb)MB"
    fi
    sleep 300; waited=$((waited+300))
  done
  log "ram ok: $(ram_avail_mb)MB available (needed ${need}MB)"
}
log "=== mixed-depth labeling started ==="
wait_for_ram 1500

$LO label \
  --in $DATA/shards/bc_v1_combined.shard \
  --out $DATA/shards/labeled_d10_4m.shard \
  --sf "$SF" --depth 10 --threads 6 --hash 256 --max-records 2000000 \
  --resume --batch-records 10000 \
  >> "$LOG" 2>&1
[ -f "$DATA/shards/labeled_d10_4m.shard" ] || { log "FATAL: d10 stage failed"; exit 1; }
log "stage 1 complete: labeled_d10_4m.shard"

$LO label \
  --in $DATA/shards/bc.shard \
  --out $DATA/shards/labeled_d16_1m.shard \
  --sf "$SF" --depth 16 --threads 6 --hash 256 --max-records 500000 \
  --resume --batch-records 5000 \
  >> "$LOG" 2>&1
[ -f "$DATA/shards/labeled_d16_1m.shard" ] || { log "FATAL: d16 stage failed"; exit 1; }
log "=== mixed-depth labeling complete: both stages ready ==="
