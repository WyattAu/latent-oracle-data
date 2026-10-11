#!/bin/bash
# Verdict-B confirmation: sharpen av_v3_s2 vs av_v3 (stage2 vs stage1) from
# 300 games (+18.55 +/- 24.16, ~0.77sigma) to 800 games.
# Serialized BEHIND grpo_phase.sh: the RL run owns the first idle window; this
# takes the next one. fastchess txt Elo follows CLI order (NOT PGN White
# order), so stage2 is listed FIRST here and the recount by engine name from
# the PGN is the authoritative number (see RESULTS 2026-10-09).
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle

S2=$DATA/runs/av_v3_s2/av_e1.bin
S1=$DATA/runs/av_v3/av_e1.bin
PGN=$DATA/sprt/av-s2-vs-s1-confirm.pgn
TXT=$DATA/sprt/av-s2-vs-s1-confirm.txt
LOG=$DATA/b_confirm.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
load1() { awk '{print int($1)}' /proc/loadavg; }
box_thrash_kbs() { vmstat 1 2 | tail -1 | awk '{print $7+$8}'; }
wait_for_idle_box() {
  local need_ram=${1:-3000} max_thrash=${2:-50} need_load=${3:-6} waited=0
  while :; do
    local a t l
    a=$(ram_avail_mb); t=$(box_thrash_kbs); l=$(load1)
    if [ "$a" -ge "$need_ram" ] && [ "$t" -le "$max_thrash" ] && [ "$l" -le "$need_load" ]; then
      log "box ok: ram ${a}MB, thrash ${t}KB/s, load ${l}"
      return 0
    fi
    if [ $((waited % 1800)) -eq 0 ]; then
      log "box busy: ram ${a}MB (need ${need_ram}), thrash ${t}KB/s (need <=${max_thrash}), load ${l} (need <=${need_load}); waiting"
    fi
    sleep 300; waited=$((waited+300))
  done
}

[ -f "$S2" ] || { log "FATAL: stage2 bin missing: $S2"; exit 1; }
[ -f "$S1" ] || { log "FATAL: stage1 bin missing: $S1"; exit 1; }
[ -x "$ENGINE" ] || { log "FATAL: engine binary missing: $ENGINE"; exit 1; }

# already done?
if [ -f "$TXT" ] && grep -q "Finished match" "$TXT" 2>/dev/null; then
  log "confirmation already complete ($(grep -c '^\[Event' "$PGN" 2>/dev/null) games); nothing to do"
  exit 0
fi

log "=== B-confirm armed (waits grpo_phase exit) ==="
while pgrep -f "grpo_phase.sh" > /dev/null; do sleep 600; done
log "grpo_phase gone; waiting for idle box"
# 2026-10-10: relaxed per user decision (see grpo_phase.sh) -- fastchess
# time-controls degrade both engines equally under load, so the verdict
# stays fair, just slower and noisier.
wait_for_idle_box 3000 2048 24

# 400 rounds x 2 colors = 800 games (~1.5-3h idle). stage2 FIRST: txt Elo then
# reads from stage2's perspective.
log "running 800-game confirmation: stage2 vs stage1, tc=15+0.2"
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=av-stage2 option.WeightsFile=$S2 \
  -engine cmd="$ENGINE" name=av-stage1 option.WeightsFile=$S1 \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 800 -rounds 400 -repeat -concurrency 6 \
  -pgnout file=$PGN > "$TXT" 2>&1

log "--- fastchess final (stage2 perspective):"
grep -E "Elo:|Games:" "$TXT" | tail -2 >> "$LOG"
log "--- pair-model recount by engine name (authoritative):"
$PY $TR/sprt_pentanomial.py "$PGN" >> "$LOG" 2>&1
log "=== B-confirm complete ==="
