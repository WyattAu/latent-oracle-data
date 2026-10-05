#!/bin/bash
# Phase-C2: distilled-1M -> SPRTs -> BC rerun (same binary) ->
#           label to 5M at DEPTH 14 -> distilled-5M -> SPRTs.
# Replaces phase_c (killed while waiting on the labeler). Fails hard.

set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LOG=$DATA/phase_c2.log
PGN_DIR=$DATA/sprt
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
PY=/tmp/opencode/venv/bin/python
LO=$TOOLS/lo-data

mkdir -p "$PGN_DIR"
exec 9>"$DATA/phase_c2.lock"
flock -n 9 || { echo "already running" >> "$LOG"; exit 1; }

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
    -pgnout file="$PGN_DIR/$2-p$3.pgn" > "$PGN_DIR/$2-p$3.txt" 2>&1
  grep -q "Finished match" "$PGN_DIR/$2-p$3.txt" || fail "SPRT $2-p$3 did not finish"
  grep -E "Elo:|Games:" "$PGN_DIR/$2-p$3.txt" | head -2 >> "$LOG"
  log "SPRT $2-p$3 done"
}

log "=== Phase-C2 started (pid $$) ==="

# 1. wait for the 1M labeling checkpoint
while pgrep -f "lo-data label" > /dev/null; do sleep 120; done
[ -f "$DATA/shards/labeled_1m.shard" ] || fail "labeled_1m.shard missing after labeler exit"
sleep 30
log "labeled_1m present — training distilled-1M"

# 2. distilled-1M training
$PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py --shard $DATA/shards/labeled_1m.shard \
  --out $DATA/runs/dist1m --epochs 2 --batch 512 >> "$LOG" 2>&1 \
  || fail "distilled-1M training failed"
[ -f "$DATA/runs/dist1m/net_e1.bin" ] || fail "distilled-1M net missing"
log "distilled-1M training done"

# 3. measurements: distilled-1M and the BC rerun — same binary, same conditions
sprt_match "$DATA/runs/dist1m/net_e1.bin" "dist1m-v0" 1 200 5
sprt_match "$DATA/runs/dist1m/net_e1.bin" "dist1m-v0" 2 200 5
NET_BC=$DATA/runs/bc/net_e1.bin
[ -f "$NET_BC" ] || fail "BC net missing"
sprt_match "$NET_BC" "bc-v0-rerun" 1 200 5
sprt_match "$NET_BC" "bc-v0-rerun" 2 200 5
log "distilled-1M + BC rerun measurements complete — H1a table ready"

# 4. label the remainder to 5M at depth 14 (resume skips the first 1M)
log "labeling to 5M at depth 14..."
$LO label --in $DATA/shards/bc.shard --out $DATA/shards/labeled_5m.shard \
  --sf "$SF" --depth 14 --multipv 3 --threads 5 --max-records 5000000 --resume >> "$LOG" 2>&1 \
  || fail "5M labeling failed"
[ -f "$DATA/shards/labeled_5m.shard" ] || fail "labeled_5m.shard missing"
log "label 5M done"

# 5. distilled-5M training + measurements
$PY $LO/trainer/train.py --shard $DATA/shards/labeled_5m.shard \
  --out $DATA/runs/dist5m --epochs 2 --batch 512 >> "$LOG" 2>&1 \
  || fail "distilled-5M training failed"
[ -f "$DATA/runs/dist5m/net_e1.bin" ] || fail "distilled-5M net missing"
sprt_match "$DATA/runs/dist5m/net_e1.bin" "dist5m-v0" 1 200 5
sprt_match "$DATA/runs/dist5m/net_e1.bin" "dist5m-v0" 2 200 5

log "=== PHASE C2 COMPLETE — H1 table ready (see $PGN_DIR) ==="
