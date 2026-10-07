#!/bin/bash
# Stop bc_v2 after epoch 0 and take the verdict.
#
# Why: at ~2.1 steps/s on this contended box a full 3-epoch pass over 130M
# positions is ~100 h. Epoch 0 alone already answers the question, and it does
# so at matched compute: 1 epoch over 130M unique positions shows 130M
# positions, against bc_v1's 3 epochs over 50M showing 150M. The difference
# between the two runs is therefore data DIVERSITY, not compute -- which is
# exactly the lever the data-scaling result (+51 Elo for 10x) points at.
#
# If epoch 0 turns out to be promising, the remaining epochs can simply be run
# later (the corpus and recipe are unchanged).
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
OUT=$DATA/runs/bc_v2
LOG=$DATA/bigdata_epoch0.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "waiting for epoch 0 of bc_v2 to finish"
while [ ! -f "$OUT/net_e0.bin" ]; do sleep 300; done
sleep 20
# stop before epoch 1 burns another 30 h
pkill -f "train.py --shard $DATA/shards/bc_v2_combined.shard"
log "epoch 0 complete; stopped the run before epoch 1"
sleep 5

if [ ! -s "$DATA/sprt/bcv2-vs-bcv1.txt" ] || ! grep -q "Finished match" "$DATA/sprt/bcv2-vs-bcv1.txt" 2>/dev/null; then
  log "SPRT: BC-v2 (130M positions, 1 epoch) vs BC-v1 gate winner (50M, 3 epochs)"
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bcv1 option.WeightsFile=$DATA/runs/bc_v1/net_best.bin \
    -engine cmd="$ENGINE" name=bcv2 option.WeightsFile=$OUT/net_e0.bin \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 400 -rounds 200 -repeat -concurrency 4 \
    -pgnout file=$DATA/sprt/bcv2-vs-bcv1.pgn > "$DATA/sprt/bcv2-vs-bcv1.txt" 2>&1
  grep -E "Elo:|Games:" "$DATA/sprt/bcv2-vs-bcv1.txt" | head -2 >> "$LOG"
  $PY "$TR/analyze_verdicts.py" "$DATA/sprt/bcv2-vs-bcv1" \
    --net-name "BC-v2 (130M unique, 1 epoch) vs BC-v1 (50M, 3 epochs)" >> "$LOG" 2>&1
fi
log "=== bc_v2 epoch-0 verdict complete ==="
