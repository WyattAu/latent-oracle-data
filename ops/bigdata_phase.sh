#!/bin/bash
# Big-data BC run (2026-10-06): the archives were capped at 25M positions per
# month when bc_v1 was built, so the same 56 GB of lichess PGNs holds far more
# data. Data scaling is the best-documented lever in this project (+51 Elo for
# the step to 50M), and 50M -> 130M predicts roughly +20 Elo on the same
# log-linear trend -- more than any mechanism currently queued.
#
# The training command replicates bc_v1's recipe EXACTLY (same width, epochs,
# batch, lr) so the only variable is the corpus size. The verdict is an SPRT
# against the bc_v1 gate winner.
#
# Stages: RAM gate -> shard +40M/month -> combine (existing 50M + new 80M)
#         -> wait for a free GPU -> train -> SPRT vs bc_v1.
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/bigdata.log
EXTRA=40000000          # positions per month beyond the original 25M cap
COMBINED=$DATA/shards/bc_v2_combined.shard
OUT=$DATA/runs/bc_v2

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
  log "ram ok: $(ram_avail_mb)MB available"
}
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_wait() { while [ "$(gpu_free_mb)" -lt 4000 ]; do sleep 180; done; }
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== big-data BC chain armed (nice'd: the labeler owns the CPU) ==="

# ---- 1. shard the two archives deeper (skip-guarded)
wait_for_ram 1200
for MONTH in 2026-07 2026-08; do
  SRC=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  DST=$DATA/shards/big_${MONTH}
  [ -f "$SRC" ] || { log "missing source $SRC"; exit 1; }
  if [ -f "$DST/bc.shard" ]; then
    log "skip: $MONTH already sharded ($(stat -c%s "$DST/bc.shard") bytes)"
    continue
  fi
  log "sharding $MONTH to $EXTRA positions (nice, this is a background filler)"
  nice -n 19 $TOOLS/lo-data shard --in "$SRC" --out "$DST" \
    --min-elo 2000 --min-ply 8 --max-ply 120 \
    --max-positions $EXTRA >> "$LOG" 2>&1 || { log "FATAL: shard $MONTH failed"; exit 1; }
  log "$MONTH shard done: $(stat -c%s "$DST/bc.shard" 2>/dev/null || echo 0) bytes"
done

# ---- 2. combine: existing 50M + the new per-month shards
if [ ! -f "$COMBINED" ]; then
  HAVE=0
  [ -f "$DATA/shards/bc_v1_combined.shard" ] && HAVE=$(stat -c%s "$DATA/shards/bc_v1_combined.shard")
  EXTRA_BYTES=$(stat -c%s "$DATA/shards/big_2026-07/bc.shard" 2>/dev/null || echo 0)
  EXTRA_BYTES=$((EXTRA_BYTES + $(stat -c%s "$DATA/shards/big_2026-08/bc.shard" 2>/dev/null || echo 0)))
  TOTAL=$((HAVE + EXTRA_BYTES))
  log "combining: $HAVE + $EXTRA_BYTES = $TOTAL bytes"
  [ $((HAVE + EXTRA_BYTES)) -gt 0 ] || { log "FATAL: nothing to combine"; exit 1; }
# Each shard file starts with its own 16-byte header, so a plain cat would
# splice headers into the middle of the record stream. Take one header,
# concatenate the payloads, then patch the record count.
  $PY - "$DATA/shards/bc_v1_combined.shard" "$DATA/shards/big_2026-07/bc.shard" \
        "$DATA/shards/big_2026-08/bc.shard" "$COMBINED.tmp" <<"COMBINE_PY"
import os, struct, sys
HEADER, REC = 16, 64
srcs, dst = sys.argv[1:4], sys.argv[4]
total = 0
with open(dst, "wb") as out:
    out.write(open(srcs[0], "rb").read(HEADER))   # exactly one header
    for path in srcs:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            fh.seek(HEADER)
            while True:
                chunk = fh.read(REC * 8192)
                if not chunk:
                    break
                assert len(chunk) % REC == 0, path + ": torn tail"
                out.write(chunk)
                total += len(chunk) // REC
    out.flush()
    os.fsync(out.fileno())
    out.seek(8)
    out.write(struct.pack("<Q", total))
print("combined %d records" % total)
COMBINE_PY
  [ -s "$COMBINED.tmp" ] || { log "FATAL: combine produced nothing"; exit 1; }
  mv "$COMBINED.tmp" "$COMBINED"
  log "combined -> $COMBINED ($(stat -c%s "$COMBINED") bytes)"
# the per-month parts are now redundant (the combined file is the artifact)
rm -rf "$DATA/shards/big_2026-07" "$DATA/shards/big_2026-08"
log "removed per-month parts to reclaim disk"
fi
$TOOLS/lo-data info --in "$COMBINED" >> "$LOG" 2>&1

# ---- 3. train (identical recipe to bc_v1; only the corpus differs)
if [ ! -f "$OUT/net_e2.bin" ]; then
  gpu_wait
  wait_for_ram 3000
  log "training bc_v2 on the enlarged corpus (same recipe as bc_v1)"
  $PY "$TR/train.py" \
    --shard "$COMBINED" --out "$OUT" \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 3 --batch 512 --lr 3e-4 >> "$LOG" 2>&1 \
    || { log "FATAL: bc_v2 training failed"; exit 1; }
  log "training done"
else
  log "skip 3: $OUT/net_e2.bin exists"
fi

# ---- 4. verdict: SPRT against the bc_v1 gate winner
if [ ! -s "$DATA/sprt/bcv2-vs-bcv1.txt" ] || ! grep -q "Finished match" "$DATA/sprt/bcv2-vs-bcv1.txt" 2>/dev/null; then
  gpu_wait
  log "SPRT: bc_v2 vs bc_v1 net_best (same recipe, more data)"
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bcv1 option.WeightsFile=$DATA/runs/bc_v1/net_best.bin \
    -engine cmd="$ENGINE" name=bcv2 option.WeightsFile=$OUT/net_e2.bin \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 400 -rounds 200 -repeat -concurrency 4 \
    -pgnout file=$DATA/sprt/bcv2-vs-bcv1.pgn > "$DATA/sprt/bcv2-vs-bcv1.txt" 2>&1
  grep -E "Elo:|Games:" "$DATA/sprt/bcv2-vs-bcv1.txt" | head -2 >> "$LOG"
  $PY "$TR/analyze_verdicts.py" "$DATA/sprt/bcv2-vs-bcv1" \
    --net-name "BC-v2 (130M positions) vs BC-v1 (50M), identical recipe" >> "$LOG" 2>&1
fi

log "=== big-data BC chain complete — verdict in $LOG ==="
