#!/bin/bash
# Post-BC-v1 pipeline:
# 1. SPRT BC-v1 (fires automatically via phase_bc_v1_sprt_v2.sh)
# 2. Train BC-v1f: quality-filtered on 1M SF-labeled positions
# 3. SPRT BC-v1f
# 4. Compare all results → research map update
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LO=$TOOLS/lo-data
PY=/tmp/opencode/venv/bin/python
TRAINER=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py
LOG=$DATA/phase_post.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }

log "=== Phase-post started: waiting for BC-v1 training ==="

# 1. Wait for BC-v1 training to complete
while pgrep -f "train.py.*bc_v1_combined" > /dev/null; do sleep 300; done

NET=$DATA/runs/bc_v1/net_e2.bin
[ -f "$NET" ] || fail "BC-v1 net_e2.bin missing after training"
log "BC-v1 training complete — waiting for >= 4GB MemAvailable (shared-box guard)"
while [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -lt 4194304 ]; do sleep 120; done
log "memory OK — starting SPRT"

# 2. SPRT BC-v1 (200 games each at p1/p2)
for PLIES in 1 2; do
  TXT=$DATA/sprt/bcv1-final-p$PLIES.txt
  log "SPRT: bc-v1-final vs sf16-p$PLIES..."
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bc-v1-final option.WeightsFile=$NET \
    -engine cmd="$SF" name=sf16-p$PLIES option.Threads=1 option.Hash=16 \
    -each proto=uci tc=60+6 plies=$PLIES \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 5 \
    -pgnout file=$DATA/sprt/bcv1-final-p$PLIES.pgn > "$TXT" 2>&1
  grep -q "Finished match" "$TXT" || fail "SPRT bc-v1-p$PLIES did not finish"
  grep -E "Elo:|Games:" "$TXT" | head -2 >> "$LOG"
  log "SPRT bc-v1-p$PLIES done"
done

# 3. Quality-filtered training: BC-v0 checkpoint → fine-tune on quality-filtered labeled data
log "generating quality-filtered training data..."
# The labeled_1m.shard already has SF eval labels. The --quality-filter flag
# in the trainer skips records where |eval| > 300 (decisive positions).
log "training BC-v1f: fine-tune BC-v0 on quality-filtered Lc0 self-play"
$PY "$TRAINER" \
  --shard $DATA/shards/lc0_sp/bc.shard \
  --out $DATA/runs/bc_v1f \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 3 --batch 512 --lr 1e-4 \
  --init-from $DATA/runs/bc/net_e1.pt \
  --quality-filter >> "$LOG" 2>&1 || fail "BC-v1f training failed"

NET_F=$DATA/runs/bc_v1f/net_e2.bin
[ -f "$NET_F" ] || fail "BC-v1f net missing"

# 4. SPRT BC-v1f
for PLIES in 1 2; do
  TXT=$DATA/sprt/bcv1f-p$PLIES.txt
  log "SPRT: bc-v1f vs sf16-p$PLIES..."
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bc-v1f option.WeightsFile=$NET_F \
    -engine cmd="$SF" name=sf16-p$PLIES option.Threads=1 option.Hash=16 \
    -each proto=uci tc=60+6 plies=$PLIES \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 5 \
    -pgnout file=$DATA/sprt/bcv1f-p$PLIES.pgn > "$TXT" 2>&1
  grep -q "Finished match" "$TXT" || fail "SPRT bc-v1f-p$PLIES did not finish"
  grep -E "Elo:|Games:" "$TXT" | head -2 >> "$LOG"
  log "SPRT bc-v1f-p$PLIES done"
done

log "=== ALL POST-TRAINING COMPLETE — full ablation table ready ==="
