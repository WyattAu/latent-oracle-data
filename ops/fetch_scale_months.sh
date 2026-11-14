#!/bin/bash
# Download 2 more lichess months for the 160M-pool scale-up run.
set -u
DATA=/home/wyatt/data
LOG=$DATA/fetch_scale.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
mkdir -p $DATA/lichess
for MONTH in 2026-06 2026-05; do
  F=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  URL=https://database.lichess.org/standard/$(basename $F)
  [ -f "$F" ] && [ "$(stat -c%s "$F")" -gt 20000000000 ] && { log "skip: $MONTH present"; continue; }
  log "downloading $MONTH"
  for attempt in $(seq 1 200); do
    curl -sS -C - --max-time 43200 -o "$F" "$URL" && break
    log "attempt $attempt for $MONTH failed (rc=$?); retry in 60s"
    sleep 60
  done
  log "$MONTH done: $(stat -c%s "$F" 2>/dev/null || echo 0) bytes"
done
log "=== scale-month downloads complete ==="
