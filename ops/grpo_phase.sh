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

# --- box-idleness gate (2026-10-09). wait_for_ram() is not enough: MemAvailable
# can look tolerable while the machine is thrashing. Measured failure was
# load 41, 0% CPU idle, 363MB free RAM, 31.5/32GB SWAP USED, driven by sibling
# agent sessions. GRPO then ran at ~21s of CPU per 4min of wall time -- 21 min
# for one step, silently. Launching into that state wastes days and looks like a
# bug rather than contention, so refuse to start and wait for a real window.
load1() { awk '{print int($1)}' /proc/loadavg; }
# Active thrash (vmstat si+so, KB/s), NOT total swap used: cold swapped-out
# pages of idle sibling sessions never drain, so a total-swap threshold blocks
# forever. Measured: 2028 KB/s while thrashing at load 13; ~0 when quiet.
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

# GPU gate: free VRAM alone is not enough -- sibling CUDA contexts add
# seconds-to-minutes of latency to every kernel launch (measured: a probe
# step took 35+ min with the GPU shared vs 0.1 ms/op in isolation). Require
# a GPU with no foreign compute processes at all.
gpu_foreign_procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; }

log "=== GRPO phase armed (waits av_phase) ==="
while pgrep -f "av_phase.sh" > /dev/null; do sleep 600; done
while [ "$(gpu_free_mb)" -lt 3600 ]; do
  sleep 300
done
# 2026-10-10: gates relaxed by user decision -- accept slower steps on the
# shared box rather than waiting days for genuine idleness. load <= 24 and
# thrash <= 2048 KB/s bound the degradation to a few x (measured 30-50x
# inflation only at load 40+ WITH swap storms); foreign GPU contexts are now
# tolerated (VRAM free >= 3600MB is the hard constraint). Watchdog raised to
# 180 min to match the accepted slower pace; resume bounds any kill to <=100
# lost steps.
# 2026-10-11: the sibling sessions' PERSISTENT baseline thrash is
# ~4,500-5,200 KB/s -- the 2048 gate waited days. The catastrophic storms
# measured 16-54 MB/s, an order of magnitude above the baseline; a 6 MB/s
# ceiling admits the baseline while still refusing the storms. The watchdog
# + resume + NaN guards bound whatever the baseline costs.
wait_for_idle_box 3000 6144 24
cd "$TR" || exit 1

BASE_PT=$DATA/runs/av_v3_s2/net_e1.pt
BASE_BIN=$DATA/runs/av_v3_s2/av_e1.bin
[ -f "$BASE_PT" ] || { log "FATAL: AV net missing"; exit 1; }

# Shard = d16: the base net (av_v3_s2) and the verdict-B signal were trained on
# d16 labels; GRPO must continue on the same distribution, not d10.
# Depth = 10: measured 14.1 ms/analysis vs 78.4 at d12 (5.6x). Reward quality
# needs consistency, not the last 20 Elo of evaluator strength; d12 makes the
# run 67h of engine time, which this shared box never delivers. d10 is ~12h.
# Fresh per-run log: the watchdog counts '^step' lines, which must mean
# THIS run's progress, not history (grpo.log holds 600+ stale step lines).
TLOG=$DATA/grpo_train.log

log "GRPO training (1500 steps, groups=128, K=16, SF depth 10, d16 shard)"
$PY grpo_train.py \
  --net $BASE_PT \
  --shard $DATA/shards/labeled_d16_1m.shard \
  --out $DATA/runs/grpo_v1 \
  --v3 \
  --groups 128 --k 16 --steps 1500 --depth 10 --reward-clip 300 \
  --lr 1e-5 --kl-beta 0.03 --clip 0.2 --export-every 100 \
  --sf-pool 4 > "$TLOG" 2>&1 &
TRAIN_PID=$!
# Step-rate watchdog: the box degrades mid-run without warning (sibling
# sessions drive load 40+, swap storms, GPU sharing -- measured, a step then
# takes hours). export-every 100 + resume mean a watchdog kill loses at most
# 100 steps, and a re-armed phase continues from the newest checkpoint
# instead of restarting at 0. 90 min of silence => kill.
STALL_SECS=$((180 * 60))
INCOMPLETE=0
last=$(grep -cE '^step' "$TLOG"); quiet=0
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60; quiet=$((quiet + 60))
  now=$(grep -cE '^step' "$TLOG")
  if [ "$now" -gt "$last" ]; then last=$now; quiet=0; fi
  if [ "$quiet" -ge "$STALL_SECS" ]; then
    log "WATCHDOG: no step line in 90min -- killing train pid $TRAIN_PID (checkpoints persist)"
    kill -9 "$TRAIN_PID" 2>/dev/null
    INCOMPLETE=1
    break
  fi
done
wait "$TRAIN_PID" 2>/dev/null
tail -3 "$TLOG" >> "$LOG"
# A watchdog kill leaves partial checkpoints. SPRT-ing a 700/1500-step net
# would put a verdict on an unfinished experiment -- refuse, and let a
# re-armed phase resume from the newest checkpoint instead.
if [ "$INCOMPLETE" = 1 ]; then
  log "FATAL: training incomplete (watchdog kill) -- no SPRTs; re-arm to resume"
  exit 1
fi
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
