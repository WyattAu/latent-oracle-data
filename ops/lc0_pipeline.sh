#!/bin/bash
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LOG=$DATA/lc0_pipeline.log
LC0_URL="https://storage.lczero.org/files/match_pgns/2"
PGN_DIR=$DATA/lc0_pgns
COMBINED=$PGN_DIR/lc0_selfplay.pgn

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "=== Lc0 self-play pipeline started ==="

mkdir -p "$PGN_DIR"

# 1. Get file listing (first 3000 files = enough for ~2M positions)
log "fetching file list..."
curl -sL "$LC0_URL/" | grep -oP 'href="[^"]+\.pgn"' | sed 's/href="//;s/"//' > "$PGN_DIR/file_list.txt"
TOTAL=$(wc -l < "$PGN_DIR/file_list.txt")
log "found $TOTAL PGN files — downloading first 3000"

# 2. Download PGNs (concurrent with xargs)
head -3000 "$PGN_DIR/file_list.txt" | xargs -P6 -I{} curl -sL -o "$PGN_DIR/{}" "$LC0_URL/{}"
COUNT=$(ls "$PGN_DIR"/*.pgn 2>/dev/null | wc -l)
log "downloaded $COUNT PGN files"

# 3. Concatenate all PGNs into one file
cat "$PGN_DIR"/*.pgn > "$COMBINED"
SIZE=$(stat -c%s "$COMBINED")
log "combined PGN: $SIZE bytes"

# 4. Shard (no Elo filter — Lc0 moves are superhuman quality)
log "sharding..."
$TOOLS/lo-data shard --in "$COMBINED" --out $DATA/shards/lc0_sp \
  --no-elo-filter --min-ply 4 --max-ply 200 \
  --max-positions 3000000 --sample 2 >> "$LOG" 2>&1 || fail "sharding failed"
log "sharding done"

# 5. Train (same architecture, 3 epochs on ~2-3M positions)
log "training Lc0-self-play model..."
/tmp/opencode/venv/bin/python /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
  --shard $DATA/shards/lc0_sp/bc.shard \
  --out $DATA/runs/lc0_sp \
  --epochs 3 --batch 512 >> "$LOG" 2>&1 || fail "training failed"
log "training done"

log "=== Lc0 SELF-PLAY PIPELINE COMPLETE ==="
