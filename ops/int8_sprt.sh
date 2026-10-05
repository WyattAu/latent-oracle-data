#!/bin/bash
# INT8-vs-FP32 quantization-cost SPRT: same BC-v0 weights, LOQW blob vs
# LONW blob, head-to-head at fast triage time control. Runs after the
# loopcd_triage chain drains (CPU-free window).
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/int8_sprt.log
FP=$DATA/runs/bc/net_e1.bin
Q=$DATA/runs/bc/net_q_v2.bin

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
log "=== int8-vs-fp32 chain armed (waits for loopcd_triage) ==="
while pgrep -f "loopcd_triage.sh" > /dev/null; do sleep 600; done

match() {  # $1 tag  $2 white-engine-blob  $3 black-engine-blob(unused label) ...
  local txt=$DATA/sprt/int8-$1.txt
  [ -s "$txt" ] && grep -q "Finished match" "$txt" && return 0
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=fp32 option.WeightsFile=$FP option.RecyclePasses=1 \
    -engine cmd="$ENGINE" name=int8 option.WeightsFile=$Q option.RecyclePasses=1 \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 400 -rounds 200 -repeat -concurrency 6 \
    -pgnout file=$DATA/sprt/int8-$1.pgn > "$txt" 2>&1
  grep -E "Elo:|Games:" "$txt" | head -2 >> "$LOG"
  log "head-to-head $1 done"
}

match h2h
log "=== int8-vs-fp32 complete (log above = quantization cost in Elo) ==="
