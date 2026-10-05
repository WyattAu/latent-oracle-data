#!/bin/bash
# SPRT BC-v1 when training completes. Runs 200 games at plies 1 and 2.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
NET=$DATA/runs/bc_v1/net_e2.bin
LOG=$DATA/phase_bc_v1_sprt.log

# Wait for training process to exit
while pgrep -f "train.py.*bc_v1_combined" > /dev/null; do sleep 300; done

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

[ -f "$NET" ] || { log "FATAL: $NET not found after training"; exit 1; }
log "BC-v1 training complete — net_e2.bin found. Starting SPRT..."

for PLIES in 1 2; do
  TXT="$DATA/sprt/bc-v1-p${PLIES}.txt"
  log "SPRT: bc-v1 vs sf16-p$PLIES (200 games)..."
  "$TOOLS/fastchess" \
    -engine cmd="$ENGINE" name="bc-v1" option.WeightsFile="$NET" \
    -engine cmd="$SF" name="sf16-p$PLIES" option.Threads=1 option.Hash=16 \
    -each proto=uci tc=60+6 plies=$PLIES \
    -openings file="$TOOLS/openings.epd" format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 5 \
    -pgnout file="$DATA/sprt/bc-v1-p${PLIES}.pgn" > "$txt" 2>&1
  grep -q "Finished match" "$txt" || { log "FATAL: SPRT bc-v1-p$PLIES did not finish"; exit 1; }
  grep -E "Elo:|Games:" "$txt" | head -2 >> "$LOG"
  log "SPRT bc-v1-p$PLIES done"
done

log "=== BC-v1 SPRT COMPLETE ==="
log "Compare with BC-v0 rerun: p1=-83, p2=-72"
log "If BC-v1 > BC-v0 → tag v0.2 and update README"
