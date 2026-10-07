#!/bin/bash
# MirrorAvg game-level SPRT: the paired position-level test measured +23.4cp
# (95% CI [+5.3, +44.6], 180 changed positions) and +1.3pp policy accuracy on a
# v3 net, but Elo is what a release ships with. This measures it directly.
#
# Runs AFTER verdict A so it cannot confound the AV comparison: both sides are
# the same net, so the only variable is the UCI MirrorAvg option.
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
NET=$DATA/runs/bc_v1/net_best.bin
LOG=$DATA/mirror_sprt.log
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== MirrorAvg SPRT armed (waits for verdict A) ==="
# verdict A is done when its analysis block exists in the AV log
until grep -q "VERDICT A complete" "$DATA/av_phase.log" 2>/dev/null; do sleep 600; done
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 3000 ]; do sleep 300; done

if [ -s "$DATA/sprt/mirror-h2h.txt" ] && grep -q "Finished match" "$DATA/sprt/mirror-h2h.txt" 2>/dev/null; then
  log "skip: mirror-h2h already finished"
else
  log "400 games: bc-best vs bc-best+MirrorAvg (same weights)"
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=plain option.WeightsFile=$NET \
    -engine cmd="$ENGINE" name=mirror option.WeightsFile=$NET option.MirrorAvg=true \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 400 -rounds 200 -repeat -concurrency 4 \
    -pgnout file=$DATA/sprt/mirror-h2h.pgn > "$DATA/sprt/mirror-h2h.txt" 2>&1
  grep -E "Elo:|Games:" "$DATA/sprt/mirror-h2h.txt" | head -2 >> "$LOG"
  $PY "$TR/analyze_verdicts.py" "$DATA/sprt/mirror-h2h" \
    --net-name "MirrorAvg on vs off (same net)" >> "$LOG" 2>&1 || true
  log "=== MirrorAvg game verdict done — flip the default only if the Elo is clearly positive ==="
fi
