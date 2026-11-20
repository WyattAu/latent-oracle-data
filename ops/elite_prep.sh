#!/bin/bash
# Elite-pool prep for the GM fine-tune: re-shard the on-disk archives at
# min-elo 2500 (the 2026-05/06 archives hold the strongest games we have).
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LOG=$DATA/elite_prep.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
for MONTH in 2026-06 2026-05; do
  SRC=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  DST=$DATA/shards/elite_${MONTH}
  [ -f "$SRC" ] || { log "FATAL: missing $SRC"; exit 1; }
  if [ -f "$DST/bc.shard" ] && [ "$(stat -c%s "$DST/bc.shard")" -gt 100000000 ]; then
    log "skip: elite_$MONTH present"
  else
    log "sharding $MONTH at min-elo 2500 (cap 20M)"
    nice -n 19 $TOOLS/lo-data shard --in "$SRC" --out "$DST" \
      --min-elo 2500 --min-ply 8 --max-ply 120 \
      --max-positions 20000000 >> "$LOG" 2>&1 || { log "FATAL: shard $MONTH elite"; exit 1; }
    log "elite $MONTH: $(stat -c%s "$DST/bc.shard" 2>/dev/null || echo 0) bytes"
  fi
done
log "=== elite prep complete ==="
