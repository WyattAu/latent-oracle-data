#!/bin/bash
# Phase-C: first SPRT/Elo measurements + the rest of the 1M/5M ladder.
#   1. BC net SPRT (light concurrency, runs during labeling)
#   2. wait for labeled_1m.shard -> distilled-1M training -> its SPRT
#   3. label to 5M (--resume) -> distilled-5M training -> its SPRT
# Every stage fails hard. flock prevents concurrent instances.

set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LOG=$DATA/phase_c.log
PGN_DIR=$DATA/sprt
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
PY=/tmp/opencode/venv/bin/python
LO_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data

mkdir -p "$PGN_DIR"
exec 9>"$DATA/phase_c.lock"
flock -n 9 || { echo "[$(date '+%m-%d %H:%M:%S')] phase_c already running" >> "$LOG"; exit 1; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }

sprt_match() { # $1 net  $2 name  $3 plies  $4 games  $5 conc
  log "SPRT match: $2 vs sf16 plies=$3 ($4 games, conc $5)..."
  "$TOOLS/fastchess" \
    -engine cmd="$ENGINE" name="$2" option.WeightsFile="$1" \
    -engine cmd="$SF" name="sf16-p$3" option.Threads=1 option.Hash=16 \
    -each proto=uci tc=60+6 plies=$3 \
    -openings file="$TOOLS/openings.epd" format=epd order=random \
    -games "$4" -rounds $(( $4 / 2 )) -repeat -concurrency "$5" \
    -pgnout file="$PGN_DIR/$2.pgn" > "$PGN_DIR/$2.txt" 2>&1
  grep -E "Elo:|LOS:|Games:|Finished match" "$PGN_DIR/$2.txt" | head -4 >> "$LOG"
  grep -q "Finished match" "$PGN_DIR/$2.txt" || fail "SPRT $2 did not finish (see $PGN_DIR/$2.txt)"
  log "SPRT $2 done"
}

log "=== Phase-C started (pid $$) ==="

# 1. BC net measurement — light concurrency, coexists with labeling
[ -f "$DATA/runs/bc/net_e1.bin" ] || fail "BC net missing"
sprt_match "$DATA/runs/bc/net_e1.bin" "bc-v0" 1 200 2
sprt_match "$DATA/runs/bc/net_e1.bin" "bc-v0" 2 200 2
log "BC measurements complete"

# 2. wait for the 1M labeling checkpoint
while [ ! -f "$DATA/shards/labeled_1m.shard" ]; do sleep 120; done
sleep 30  # let the labeler finish writing
log "labeled_1m present — training distilled-1M"

# 3. distilled-1M training
$PY $LO_REPO/trainer/train.py --shard $DATA/shards/labeled_1m.shard \
  --out $DATA/runs/dist1m --epochs 2 --batch 512 >> "$LOG" 2>&1 \
  || fail "distilled-1M training failed"
[ -f "$DATA/runs/dist1m/net_e1.bin" ] || fail "distilled-1M net missing"
log "distilled-1M training done"

# 4. distilled-1M measurement
sprt_match "$DATA/runs/dist1m/net_e1.bin" "dist1m-v0" 1 200 5
sprt_match "$DATA/runs/dist1m/net_e1.bin" "dist1m-v0" 2 200 5
log "distilled-1M measurements complete"

# 5. label to 5M (resume skips the first 1M)
log "labeling to 5M..."
$TOOLS/lo-data label --in $DATA/shards/bc.shard --out $DATA/shards/labeled_5m.shard \
  --sf "$SF" --depth 16 --multipv 3 --threads 5 --max-records 5000000 --resume >> "$LOG" 2>&1 \
  || fail "5M labeling failed"
[ -f "$DATA/shards/labeled_5m.shard" ] || fail "labeled_5m.shard missing"
log "label 5M done"

# 6. distilled-5M training + measurement
$PY $LO_REPO/trainer/train.py --shard $DATA/shards/labeled_5m.shard \
  --out $DATA/runs/dist5m --epochs 2 --batch 512 >> "$LOG" 2>&1 \
  || fail "distilled-5M training failed"
[ -f "$DATA/runs/dist5m/net_e1.bin" ] || fail "distilled-5M net missing"
sprt_match "$DATA/runs/dist5m/net_e1.bin" "dist5m-v0" 1 200 5
sprt_match "$DATA/runs/dist5m/net_e1.bin" "dist5m-v0" 2 200 5

log "=== PHASE C COMPLETE — H1 table ready (see $PGN_DIR/*.txt) ==="
