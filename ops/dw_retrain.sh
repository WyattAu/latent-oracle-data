#!/bin/bash
# Decisiveness-weighted distilled retrain: tests the fix for the H1 failure
# mode. H1 showed SF-label distillation with UNIFORM loss degrades play
# (~309 Elo vs BC). Hypothesis: equal positions poisoned the policy. This
# run retrains on the same labeled_1m.shard with decisiveness weighting.
# Queues after the running fine-tune exits (GPU contention).
set -u
DATA=/home/wyatt/data
LOG=$DATA/dw_train.log
PY=/tmp/opencode/venv/bin/python
TRAINER=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "=== decisive-weighted distilled retrain queued ==="

# Wait for the fine-tune (finetune_bc0_lc0) to finish and free GPU memory
while pgrep -f "train.py.*finetune_bc0_lc0|train.py.*lc0_sp" > /dev/null 2>&1 || \
      pgrep -f "init-from.*runs/bc/net_e1.pt" > /dev/null 2>&1; do
  sleep 300
done

log "GPU free — starting decisive-weighted retrain on labeled_1m.shard"

$PY "$TRAINER" \
  --shard $DATA/shards/labeled_1m.shard \
  --out $DATA/runs/dist1m_dw \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 3 --batch 512 --lr 3e-4 \
  --decisive-weighting >> "$LOG" 2>&1

NET=$DATA/runs/dist1m_dw/net_e2.bin
if [ ! -f "$NET" ]; then log "FATAL: training failed"; exit 1; fi
log "training complete — starting SPRT vs SF16-p1"

TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2

$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=dist1m-dw option.WeightsFile=$NET \
  -engine cmd="$SF" name=sf16-p1 option.Threads=1 option.Hash=16 \
  -each proto=uci tc=60+6 plies=1 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 200 -rounds 100 -repeat -concurrency 5 \
  -pgnout file=$DATA/sprt/dist1m-dw-p1.pgn > $DATA/sprt/dist1m-dw-p1.txt 2>&1

grep -q "Finished match" $DATA/sprt/dist1m-dw-p1.txt || { log "FATAL: SPRT failed"; exit 1; }
grep -E "Elo:|Games:" $DATA/sprt/dist1m-dw-p1.txt | head -2 >> "$LOG"
log "=== H2 COMPLETE: uniform (-436) vs decisive-weighted (above) ==="
