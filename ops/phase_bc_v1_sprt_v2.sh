#!/bin/bash
# SPRT BC-v1 when the net_e2.bin checkpoint file appears.
# Robust: watches for the FILE, not the process. Survives training restarts.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
NET=$DATA/runs/bc_v1/net_e2.bin
LOG=$DATA/phase_bc_v1_sprt.log

# Clean up any stale fastchess/engine processes from a previous run
pkill -x fastchess 2>/dev/null
pkill -x latent-oracle 2>/dev/null
sleep 2

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
fail() { log "FATAL: $*"; exit 1; }

# 1. Wait for the training to produce the final checkpoint
log "waiting for $NET to appear..."
while [ ! -f "$NET" ]; do sleep 300; done

# Ensure the file is fully written (not still being flushed)
sleep 30
ACTUAL=$(stat -c%s "$NET" 2>/dev/null || echo 0)
log "checkpoint found: $ACTUAL bytes"

# 2. Run SPRT
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
  grep -q "Finished match" "$txt" || fail "SPRT bc-v1-p$PLIES did not finish"
  grep -E "Elo:|Games:" "$txt" | head -2 >> "$LOG"
  log "SPRT bc-v1-p$PLIES done"
done

log "=== BC-v1 SPRT COMPLETE ==="
log "Compare with BC-v0 rerun: p1=-83, p2=-72"
