#!/bin/bash
# AV stage-3 phase: re-run the stage-2 recipe (d16 RCT/QAT fine-tune from the
# stage-1 net) with the FULL d16 TB sidecar (3,728 labels, complete 500k
# shard) instead of the 2,653-label 335k-prefix sidecar that trained
# av_v3_s2. Same hyperparameters as av_phase.sh phase B throughout, so the
# s3-vs-s2 verdict isolates exactly one variable: sidecar completeness.
# Chained behind grpo_phase.sh and b_confirm.sh — the RL run and the verdict-B
# confirmation own the first idle windows; this takes the next one.
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/av_s3.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_foreign_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }
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

[ -f "$DATA/shards/labeled_d16_1m.shard" ] || { log "FATAL: d16 shard missing"; exit 1; }
[ -f "$DATA/shards/labeled_d16_1m.shard.tb.jsonl" ] || { log "FATAL: full TB sidecar missing"; exit 1; }
[ -f "$DATA/runs/av_v3/net_e1.pt" ] || { log "FATAL: stage-1 net missing"; exit 1; }
[ -x "$ENGINE" ] || { log "FATAL: engine binary missing"; exit 1; }

log "=== AV s3 phase armed (waits grpo_phase + b_confirm exit) ==="
while pgrep -f "grpo_phase.sh" > /dev/null; do sleep 600; done
while pgrep -f "b_confirm.sh" > /dev/null; do sleep 600; done
while [ "$(gpu_free_mb)" -lt 3600 ]; do
  sleep 300
done
# 2026-10-10: relaxed per user decision (see grpo_phase.sh).
wait_for_idle_box 3000 2048 24

log "AV s3: d16 RCT/QAT fine-tune from stage-1, FULL TB sidecar (3,728 labels)"
cd "$TR" || exit 1
$PY train.py \
  --shard $DATA/shards/labeled_d16_1m.shard \
  --out $DATA/runs/av_v3_s3 \
  --epochs 2 --batch 256 --optimizer muon \
  --gab --v3 --init-from $DATA/runs/av_v3/net_e1.pt \
  --mirror --ema-decay 0.999 \
  --decisive-weighting --tb-sidecar $DATA/shards/labeled_d16_1m.shard.tb.jsonl \
  --opening-weight 1.5 \
  --recycle 2 --rct-lambda 0.5 --qat \
  >> "$LOG" 2>&1
[ -f "$DATA/runs/av_v3_s3/net_e1.pt" ] || { log "FATAL: AV s3 training failed"; exit 1; }

$PY - << 'PYEXPORT' >> "$LOG" 2>&1
import sys, torch
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet
m = ChessNet(v3=True)
m.load_state_dict(torch.load("/home/wyatt/data/runs/av_v3_s3/net_e1.pt",
                             map_location="cpu", weights_only=True))
m.export_blob("/home/wyatt/data/runs/av_v3_s3/av_e1.bin")
print("s3 blob exported")
PYEXPORT

# Sidecar A/B: s3 (full sidecar) vs s2 (prefix sidecar) — same recipe, one
# variable. stage2 listed second: txt Elo then reads from stage3's perspective;
# the engine-name PGN recount stays authoritative.
log "SPRT: av-s3 vs av-s2 (sidecar A/B, 300 games)"
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=av-stage3 option.WeightsFile=$DATA/runs/av_v3_s3/av_e1.bin \
  -engine cmd="$ENGINE" name=av-stage2 option.WeightsFile=$DATA/runs/av_v3_s2/av_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 300 -rounds 150 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/av-s3-vs-s2.pgn > $DATA/sprt/av-s3-vs-s2.txt 2>&1
log "--- fastchess final (stage3 perspective):"
grep -E "Elo:|Games:" $DATA/sprt/av-s3-vs-s2.txt | tail -2 >> "$LOG"
log "--- pair-model recount by engine name (authoritative):"
$PY $TR/sprt_pentanomial.py $DATA/sprt/av-s3-vs-s2.pgn >> "$LOG" 2>&1
log "=== AV s3 phase complete — sidecar A/B verdict in $LOG ==="
