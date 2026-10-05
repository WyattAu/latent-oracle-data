#!/bin/bash
# GRPO phase (Phase 3): RL fine-tuning of the AV net.
# Waits for av_phase.sh exit (GPU free), runs GRPO with SF-reward groups
# (Gumbel-top-K sampling), then verdicts:
#   - fast SPRT A/B vs the AV base (promotion gate)
#   - fast SPRT vs SF16-p1 (absolute)
#   - E4 conversion metrics on the A/B PGNs
# Promotion rule: beat base AND don't regress vs pool. Kill: no base gain
# after 3 exports (handled by reading the A/B logs).
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/grpo.log
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- memory preflight (2026-10-05: OOM kills took out the d10 labeler at
# 3.9M/4M records and the DiffuSearch sample build; another session on this
# box routinely eats 25+ GB). Wait for real headroom instead of dying.
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
ram_total_mb() { awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo; }
wait_for_ram() {
  local need=${1:-3000} waited=0
  while [ "$(ram_avail_mb)" -lt "$need" ]; do
    if [ $((waited % 1800)) -eq 0 ]; then
      log "ram wait: need ${need}MB, have $(ram_avail_mb)MB of $(ram_total_mb)MB"
    fi
    sleep 300; waited=$((waited+300))
  done
  log "ram ok: $(ram_avail_mb)MB available (needed ${need}MB)"
}

log "=== GRPO phase armed (waits av_phase) ==="
while pgrep -f "av_phase.sh" > /dev/null; do sleep 600; done
while [ "$(gpu_free_mb)" -lt 3600 ]; do sleep 300; done
cd "$TR" || exit 1

BASE_PT=$DATA/runs/av_v3_s2/net_e1.pt
BASE_BIN=$DATA/runs/av_v3_s2/av_e1.bin
[ -f "$BASE_PT" ] || { log "FATAL: AV net missing"; exit 1; }

log "GRPO training (1500 steps, groups=128, K=16, SF depth 12)"
$PY grpo_train.py \
  --net $BASE_PT \
  --shard $DATA/shards/labeled_d10_4m.shard \
  --out $DATA/runs/grpo_v1 \
  --v3 \
  --groups 128 --k 16 --steps 1500 --depth 12 --reward-clip 300 \
  --lr 1e-5 --kl-beta 0.03 --clip 0.2 --export-every 500 \
  --sf-pool 4 >> "$LOG" 2>&1
[ -f "$DATA/runs/grpo_v1/grpo_s1500.pt" ] || FINAL=$(ls -t $DATA/runs/grpo_v1/grpo_s*.pt 2>/dev/null | head -1)
FINAL=${FINAL:-$DATA/runs/grpo_v1/grpo_s1500.pt}
[ -f "$FINAL" ] || { log "FATAL: GRPO produced no checkpoint"; exit 1; }
FINAL_BIN=${FINAL%.pt}.bin
log "GRPO complete: $FINAL — SPRT A/B vs base, then vs pool"

# ---- A/B vs base
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=av-base option.WeightsFile=$BASE_BIN \
  -engine cmd="$ENGINE" name=grpo option.WeightsFile=$FINAL_BIN \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 400 -rounds 200 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/grpo-vs-av.pgn > $DATA/sprt/grpo-vs-av.txt 2>&1
grep -E "Elo:|Games:" $DATA/sprt/grpo-vs-av.txt | head -2 >> "$LOG"

# ---- vs pool
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=grpo option.WeightsFile=$FINAL_BIN \
  -engine cmd="$SF" name=sf16-p1 option.Threads=1 option.Hash=16 \
  -each proto=uci tc=15+0.2 plies=1 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 400 -rounds 200 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/grpo-vs-pool.pgn > $DATA/sprt/grpo-vs-pool.txt 2>&1
grep -E "Elo:|Games:" $DATA/sprt/grpo-vs-pool.txt | head -2 >> "$LOG"

# ---- E4 conversion metrics on both PGNs
$PY conversion_metrics.py $DATA/sprt/grpo-vs-av.pgn >> "$LOG" 2>&1
$PY conversion_metrics.py $DATA/sprt/grpo-vs-pool.pgn >> "$LOG" 2>&1
log "=== GRPO phase complete — promotion rule: beat base AND no pool regression ==="
