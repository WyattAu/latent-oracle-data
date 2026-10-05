#!/bin/bash
# BC-v1 pipeline: wait for downloads → shard all months → combine → train
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LO=$TOOLS/lo-data
LOG=$DATA/bc_v1.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }

log "=== BC-v1 pipeline started (pid $$) ==="

# 1. Wait for downloads
while pgrep -f "fetch_more" > /dev/null; do sleep 120; done
log "downloads complete"

# 2. Shard each month
for MONTH in 2026-07 2026-08 2026-09; do
  PGN=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  OUT=$DATA/shards/bc_${MONTH}.shard
  if [ -f "$OUT" ] && [ "$(stat -c%s "$OUT")" -gt 1000000 ]; then
    log "$MONTH already sharded ($(stat -c%s "$OUT") bytes) — skipping"
    continue
  fi
  if [ ! -f "$PGN" ]; then
    log "WARNING: $PGN not found — skipping $MONTH"
    continue
  fi
  log "sharding $MONTH..."
  $TOOLS/lo-data shard --in "$PGN" --out $DATA/shards/${MONTH}_tmp \
    --min-elo 2000 --min-ply 8 --max-ply 120 \
    --max-positions 25000000 >> "$LOG" 2>&1 || fail "shard $MONTH failed"
  log "$MONTH shard done: $(stat -c%s $DATA/shards/${MONTH}_tmp/bc.shard 2>/dev/null || echo 0) bytes"
done

# 3. Combine shards (header from first, records from all)
COMBINED=$DATA/shards/bc_v1_combined.shard
log "combining shards..."
python3 - <<PYEOF
import os, struct
DATA = "$DATA"
shard_files = []
for month in ["2026-07", "2026-08", "2026-09"]:
    p = os.path.join(DATA, "shards", f"bc_{month}.shard" if month != "2026-07" else "bc.shard")
    if month == "2026-07":
        p = os.path.join(DATA, "shards", "bc.shard")
    if os.path.exists(p):
        shard_files.append(p)

if not shard_files:
    exit(1)

HEADER = 16
REC = 64
total = 0
with open(os.path.join(DATA, "shards", "bc_v1_combined.shard"), "wb") as out:
    # header from first file
    with open(shard_files[0], "rb") as f:
        out.write(f.read(HEADER))
    for sf in shard_files:
        sz = os.path.getsize(sf)
        n_records = (sz - HEADER) // REC
        total += n_records
        with open(sf, "rb") as f:
            f.seek(HEADER)
            remaining = n_records * REC
            while remaining > 0:
                chunk = f.read(min(remaining, 1 << 20))
                out.write(chunk)
                remaining -= len(chunk)

# patch count
with open(os.path.join(DATA, "shards", "bc_v1_combined.shard"), "r+b") as f:
    f.seek(8)
    f.write(struct.pack("<Q", total))

print(f"combined: {total} records from {len(shard_files)} shards")
PYEOF
log "combined shard ready"

# 4. Train BC-v1 (3 epochs on ~60M positions = 180M samples)
log "training BC-v1..."
$PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
  --shard "$COMBINED" --out $DATA/runs/bc_v1 \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 3 --batch 512 --lr 3e-4 >> "$LOG" 2>&1 || fail "BC-v1 training failed"

log "BC-v1 training done"
log "=== BC-V1 PIPELINE COMPLETE ==="
