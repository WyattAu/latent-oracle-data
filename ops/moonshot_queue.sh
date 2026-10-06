#!/bin/bash
# Moonshot queue v2 (2026-10-04): skip-guarded stages + pre-scripted
# double-down extensions (user decision: re-invest GPU in winners).
#   1. DiffuSearch h=2 training (skip if diffu_e1.pt exists)
#   2. DiffuSearch eval: a0-match T=16, 300 samples
#   3. AMZ pilot (skip if amz_e0.pt exists) + puzzle gates
#   4. AMZ blob export + fast SPRT A/B vs BC-v1 e2
#   5. DOUBLE-DOWN DiffuSearch: h=4 run if a0 >= 0.40; h=2 2x-epoch
#      extension if 0.25 <= a0 < 0.40; none below 0.25
#   6. DOUBLE-DOWN AMZ: bootstrap iterations 2+3 if puzzle gate passes
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/moonshot.log
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

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
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== moonshot queue v2 armed (waits recovery_seq + GPU) ==="
while pgrep -f "recovery_seq.sh" > /dev/null; do sleep 600; done
gpu_wait() { while [ "$(gpu_free_mb)" -lt 3600 ]; do sleep 180; done; }
gpu_wait
cd "$TR"

# ---- 0. DiffuSearch strength verdict (2026-10-06): REJECTED.
# A position-level strength test (trainer/diffusion_playout.py, diffusion
# policy vs greedy play of the SAME BC weights) scored 0W 0D 12L with the
# inference path verified against infer_diffusion (27% vs 33% a0 agreement).
# The a0-match gate below measures next-move predictability, NOT strength: it
# read 0.33 for a policy that loses every game, and 0.97 when half the future
# was left visible. Do not re-run DiffuSearch training on the a0 signal.
DIFFU_VERDICT="REJECTED (playout 0-12 to greedy BC; a0 gate invalid as strength proxy)"
DIFFU_REJECTED=1
log "DiffuSearch: $DIFFU_VERDICT"

# ---- 1. DiffuSearch h=2
if [ "$DIFFU_REJECTED" = "1" ]; then
  log "skip 1: DiffuSearch rejected by strength verdict"
elif [ ! -f "$DATA/runs/diffu_v1/diffu_e1.pt" ]; then
  gpu_wait
  wait_for_ram 3000
  log "moonshot 1: DiffuSearch h=2 training"
  $PY train_diffusion.py \
    --shard $DATA/shards/bc_v1_combined.shard \
    --out $DATA/runs/diffu_v1 \
    --horizon 2 --max-samples 2000000 \
    --epochs 2 --batch 64 --d 256 --layers 8 --heads 8 --dff 1024 \
    --T 64 --lr 3e-4 --holdout 2000 \
    >> "$LOG" 2>&1
  [ -f "$DATA/runs/diffu_v1/diffu_e1.pt" ] || { log "FATAL: DiffuSearch training failed"; exit 1; }
else
  log "skip 1: diffu_e1.pt exists"
fi

# ---- 2. DiffuSearch eval
if [ "$DIFFU_REJECTED" = "1" ]; then
  AMATCH=0.0
  log "skip 2: DiffuSearch eval (rejected)"
else
$PY infer_diffusion.py --run $DATA/runs/diffu_v1 --ckpt diffu_e1.pt \
  --T 16 --limit 300 >> "$LOG" 2>&1
AMATCH=$(grep -oE "a0 match [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
log "DiffuSearch a0-match = ${AMATCH:-?} (kill < 0.25; double-down >= 0.40)"
fi

# ---- 3. AMZ pilot
if [ ! -f "$DATA/runs/amz_pilot/amz_e0.pt" ]; then
  gpu_wait
  log "moonshot 2: AMZ pilot (50k positions)"
  $PY amz_pilot.py \
    --net $DATA/runs/bc_v1/net_e2.pt \
    --shard $DATA/shards/bc_v1_combined.shard \
    --out $DATA/runs/amz_pilot \
    --positions 50000 --epochs 1 --batch 256 --lr 1e-4 \
    >> "$LOG" 2>&1
  [ -f "$DATA/runs/amz_pilot/amz_e0.pt" ] || { log "FATAL: AMZ pilot failed"; exit 1; }
else
  log "skip 3: amz_e0.pt exists"
fi

BASE_MATCH=$($PY score_puzzles.py --net $DATA/runs/bc_v1/net_e2.pt \
  --puzzles $DATA/puzzles_500.jsonl 2>/dev/null | grep -oE "= [0-9.]+" | head -1 | tr -d "= ")
AMZ_MATCH=$($PY score_puzzles.py --net $DATA/runs/amz_pilot/amz_e0.pt \
  --puzzles $DATA/puzzles_500.jsonl 2>/dev/null | grep -oE "= [0-9.]+" | head -1 | tr -d "= ")
log "puzzles: base=$BASE_MATCH amz=$AMZ_MATCH (gate: amz >= base + 1.5 points i.e. +0.015)"

# ---- 4. AMZ blob + fast SPRT A/B
if [ ! -f "$DATA/runs/amz_pilot/amz_e0.bin" ]; then
  $PY - << 'PYEOF' >> "$LOG" 2>&1
import sys, torch
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet
m = ChessNet()
m.load_state_dict(torch.load("/home/wyatt/data/runs/amz_pilot/amz_e0.pt",
                             map_location="cpu", weights_only=True))
m.export_blob("/home/wyatt/data/runs/amz_pilot/amz_e0.bin")
print("AMZ blob exported")
PYEOF
fi
if [ ! -s "$DATA/sprt/amz-vs-bc.txt" ] || ! grep -q "Finished match" "$DATA/sprt/amz-vs-bc.txt" 2>/dev/null; then
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bc-v1-e2 option.WeightsFile=$DATA/runs/bc_v1/net_e2.bin \
    -engine cmd="$ENGINE" name=amz-e0 option.WeightsFile=$DATA/runs/amz_pilot/amz_e0.bin \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 400 -rounds 200 -repeat -concurrency 6 \
    -pgnout file=$DATA/sprt/amz-vs-bc.pgn > $DATA/sprt/amz-vs-bc.txt 2>&1
  grep -E "Elo:|Games:" $DATA/sprt/amz-vs-bc.txt | head -2 >> "$LOG"
fi

# ---- 5. DiffuSearch double-down (pre-scripted, user decision)
DOUBLE=0
if python3 -c "import sys; sys.exit(0 if float('${AMATCH:-0}') >= 0.40 else 1)" 2>/dev/null; then
  log "double-down: a0 >= 0.40 — DiffuSearch h=4 (paper config)"
  H=4; EPOCHS=2
elif python3 -c "import sys; sys.exit(0 if float('${AMATCH:-0}') >= 0.25 else 1)" 2>/dev/null; then
  log "double-down: 0.25 <= a0 < 0.40 — DiffuSearch h=2 extended training"
  H=2; EPOCHS=4
else
  log "no DiffuSearch double-down (a0 < 0.25)"
  H=0
fi
if [ "$DIFFU_REJECTED" = "1" ]; then
  log "skip 5: no DiffuSearch double-down (rejected by strength verdict)"
elif [ "$H" != "0" ]; then
  gpu_wait
  $PY train_diffusion.py \
    --shard $DATA/shards/bc_v1_combined.shard \
    --out $DATA/runs/diffu_dd \
    --horizon $H --max-samples 2000000 \
    --epochs $EPOCHS --batch 64 --d 256 --layers 8 --heads 8 --dff 1024 \
    --T 64 --lr 3e-4 --holdout 2000 \
    >> "$LOG" 2>&1
  [ -f "$DATA/runs/diffu_dd/diffu_e1.pt" ] && {
    $PY infer_diffusion.py --run $DATA/runs/diffu_dd --ckpt diffu_e1.pt \
      --T 16 --limit 300 >> "$LOG" 2>&1
    AMATCH2=$(grep -oE "a0 match [0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+$")
    log "double-down a0-match = ${AMATCH2:-?} (was $AMATCH)"
  }
fi

# ---- 6. AMZ ladder (pre-scripted): iterations 2+3 on gate pass
PASS=$(python3 -c "
import sys
try: sys.exit(0 if float('${AMZ_MATCH:-0}') >= float('${BASE_MATCH:-1}') + 0.015 else 1)
except Exception: sys.exit(1)" 2>/dev/null && echo yes || echo no)
if [ "$PASS" = "yes" ]; then
  for IT in 2 3; do
    gpu_wait
    log "AMZ ladder iteration $IT (targets from iteration $((IT-1)) net)"
    PREV=$DATA/runs/amz_pilot/amz_e$((IT-2)).pt
    OUTD=$DATA/runs/amz_pilot/it$IT
    $PY amz_pilot.py \
      --net $PREV \
      --shard $DATA/shards/bc_v1_combined.shard \
      --out $OUTD \
      --positions 50000 --epochs 1 --batch 256 --lr 1e-4 \
      >> "$LOG" 2>&1
    $PY score_puzzles.py --net $OUTD/amz_e0.pt \
      --puzzles $DATA/puzzles_500.jsonl >> "$LOG" 2>&1
  done
  log "AMZ ladder complete — bootstrap check: monotone puzzle match?"
else
  log "no AMZ ladder (gate failed)"
fi

log "=== moonshots v2 complete — read $LOG ==="
