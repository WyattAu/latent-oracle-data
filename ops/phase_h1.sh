#!/bin/bash
# Post-training chain: wait for dist1m_ext → SPRT → H1 verdict → branch.
# Fully detached, logged, fail-hard per stage.

DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
LOG=$DATA/phase_h1.log
PGN_DIR=$DATA/sprt
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LO=$TOOLS/lo-data
PY=/tmp/opencode/venv/bin/python
NET_DIR=$DATA/runs/dist1m_ext
NET_BC=$DATA/runs/bc/net_e1.bin

mkdir -p "$PGN_DIR"
exec 9>"$DATA/phase_h1.lock"
flock -n 9 || { echo "already running" >> "$LOG"; exit 1; }

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }

log "=== Phase-H1 chain started (pid $$) ==="

# 1. Wait for training completion (net_e19.bin = last of 20 epochs)
while [ ! -f "$NET_DIR/net_e19.bin" ]; do sleep 120; done
sleep 30  # let the final export settle
log "training complete — net_e19.bin found"

# 2. SPRT the final distilled net
sprt() { # $1 net  $2 name  $3 plies  $4 games  $5 conc
  local txt="$PGN_DIR/$2-p$3.txt"
  log "SPRT: $2 vs sf16 plies=$3 ($4 games, conc $5)..."
  "$TOOLS/fastchess" \
    -engine cmd="$ENGINE" name="$2" option.WeightsFile="$1" \
    -engine cmd="$SF" name="sf16-p$3" option.Threads=1 option.Hash=16 \
    -each proto=uci tc=60+6 plies=$3 \
    -openings file="$TOOLS/openings.epd" format=epd order=random \
    -games "$4" -rounds $(( $4 / 2 )) -repeat -concurrency "$5" \
    -pgnout file="$PGN_DIR/$2-p$3.pgn" > "$txt" 2>&1
  grep -q "Finished match" "$txt" || fail "SPRT $2-p$3 did not finish"
  grep -E "Elo:|Games:" "$txt" | head -2 >> "$LOG"
  log "SPRT $2-p$3 done"
}

sprt "$NET_DIR/net_e19.bin" "dist1mext-v0" 1 200 5
sprt "$NET_DIR/net_e19.bin" "dist1mext-v0" 2 200 5
log "dist1m_ext SPRTs complete"

# 3. Read the Elo estimates and compare with the BC rerun
# BC rerun: -83 (p1), -72 (p2) from phase_c2
# Extract the Elo means from the SPRT outputs
ELO_P1=$(grep -oP 'Elo: \K-?\d+' "$PGN_DIR/dist1mext-v0-p1.txt" | head -1)
ELO_P2=$(grep -oP 'Elo: \K-?\d+' "$PGN_DIR/dist1mext-v0-p2.txt" | head -1)
BC_P1=-83
BC_P2=-72

DIFF_P1=$((ELO_P1 - BC_P1))
DIFF_P2=$((ELO_P2 - BC_P2))
AVG_DIFF=$(((DIFF_P1 + DIFF_P2) / 2))

log "H1 VERDICT: dist1m_ext p1=${ELO_P1} p2=${ELO_P2}"
log "H1 VERDICT: BC rerun  p1=${BC_P1} p2=${BC_P2}"
log "H1 VERDICT: avg diff = ${AVG_DIFF} Elo"

# 4. Branch
if [ "$AVG_DIFF" -ge 50 ]; then
  log "H1 POSITIVE: SF distillation works (≥+50 Elo) — launching 5M labeling at depth 14"
  # 5M labeling at depth 14 with resume
  $LO label --in $DATA/shards/bc.shard --out $DATA/shards/labeled_5m.shard \
    --sf "$SF" --depth 14 --multipv 3 --threads 5 --max-records 5000000 --resume >> "$LOG" 2>&1 \
    || fail "5M labeling failed"
  [ -f "$DATA/shards/labeled_5m.shard" ] || fail "labeled_5m.shard missing"
  log "5M labeling done — training distilled-5M"

  $PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
    --shard $DATA/shards/labeled_5m.shard --out $DATA/runs/dist5m \
    --epochs 20 --batch 512 >> "$LOG" 2>&1 || fail "dist5m training failed"
  [ -f "$DATA/runs/dist5m/net_e19.bin" ] || fail "dist5m net missing"
  log "distilled-5M training done"

  sprt "$DATA/runs/dist5m/net_e19.bin" "dist5m-v0" 1 200 5
  sprt "$DATA/runs/dist5m/net_e19.bin" "dist5m-v0" 2 200 5
  log "distilled-5M measurements complete"
else
  log "H1 NEGATIVE/INCONCLUSIVE: SF distillation gain < 50 Elo — skipping 5M, focus on M2 + BC scaling"
fi

log "=== PHASE H1 COMPLETE — results in $PGN_DIR and $LOG ==="
