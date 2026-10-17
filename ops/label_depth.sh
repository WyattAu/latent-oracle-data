#!/bin/bash
# Clean label-depth test: does deeper-oracle supervision help through PLAIN BC?
#
# Verdict-B confounded two changes at once: the d16 shard (SF depth-16 labels)
# AND the RCT/QAT recipe. The AV checkpoint also confounds (weak trunk). This
# test isolates label depth -- fine-tune the stage-1 champion on d16 labels
# with no RCT/QAT/recycle/sidecar weighting at all, then SPRT vs the champion.
#
# H0: label depth is a null lever (d16 ~= d10 through plain BC).
# 300 games, tc=15+0.2, resolution ~ +/- 25 Elo (pair model, by engine name).
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle

NEW=$DATA/runs/av_v3_d16bc/net_e1_ema.bin
OLD=$DATA/runs/av_v3/net_e1.bin
PGN=$DATA/sprt/av-d16bc-vs-s1.pgn
TXT=$DATA/sprt/av-d16bc-vs-s1.txt
LOG=$DATA/label_depth.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
load1() { awk '{print int($1)}' /proc/loadavg; }
wait_for_idle_box() {
  local need_ram=${1:-3000} need_load=${2:-6} waited=0
  while :; do
    local a l
    a=$(ram_avail_mb); l=$(load1)
    if [ "$a" -ge "$need_ram" ] && [ "$l" -le "$need_load" ]; then
      log "box ok: ram ${a}MB, load ${l}"
      return 0
    fi
    if [ $((waited % 1800)) -eq 0 ]; then
      log "box busy: ram ${a}MB (need ${need_ram}), load ${l} (need <=${need_load}); waiting"
    fi
    sleep 300; waited=$((waited+300))
  done
}

[ -f "$NEW" ] || { log "FATAL: d16bc net missing: $NEW"; exit 1; }
[ -f "$OLD" ] || { log "FATAL: champion missing: $OLD"; exit 1; }
[ -x "$ENGINE" ] || { log "FATAL: engine binary missing: $ENGINE"; exit 1; }

if [ -f "$TXT" ] && grep -q "Finished match" "$TXT" 2>/dev/null; then
  log "label-depth match already complete; nothing to do"
  exit 0
fi

wait_for_idle_box 3000 6

log "running 300-game label-depth test: d16bc-ema vs stage1, tc=15+0.2"
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=av-d16bc option.WeightsFile=$NEW \
  -engine cmd="$ENGINE" name=av-stage1 option.WeightsFile=$OLD \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 300 -rounds 150 -repeat -concurrency 6 \
  -pgnout file=$PGN > "$TXT" 2>&1

log "--- fastchess final (d16bc perspective):"
grep -E "Elo:|Games:" "$TXT" | tail -2 >> "$LOG"
log "--- pair-model recount by engine name (authoritative):"
$PY $TR/sprt_pentanomial.py "$PGN" >> "$LOG" 2>&1
log "=== label-depth test complete ==="
