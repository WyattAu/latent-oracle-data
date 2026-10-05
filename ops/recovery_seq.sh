#!/bin/bash
# Recovery sequencer v3: strictly sequential on the GPU.
#   wait BC-v1 exits -> fine-tune resume -> H2 retrain -> H2 SPRT
# The v2 attempt co-ran the fine-tune next to BC-v1 and OOM'd twice; a free
# GPU makes each stage fast and safe.
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TRAINER=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/recovery_seq.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
wait_gpu() { while [ "$(gpu_free_mb)" -lt "$1" ]; do sleep 120; done; }

log "=== recovery sequencer v3 armed (strictly sequential) ==="

# ---- Wait for BC-v1 to free the GPU
log "waiting for BC-v1 training to exit..."
while pgrep -f "train.py.*bc_v1_combined" > /dev/null; do sleep 300; done
# phase_post fires on the same trigger and runs its own GPU fine-tune
# (bc_v1f); our GPU stages must queue behind it.
log "waiting for phase_post chain (incl. its fine-tune) to fully exit..."
while pgrep -f "phase_post.sh" > /dev/null; do sleep 300; done
log "BC-v1 gone; waiting for >=3600MB GPU free..."
wait_gpu 3600

# ---- Stage 1: fine-tune resume (full GPU available: batch 256)
log "resuming Lc0 fine-tune from epoch-0 checkpoint (2 epochs, batch 256)"
$PY "$TRAINER" \
  --shard $DATA/shards/lc0_sp/bc.shard \
  --out $DATA/runs/finetune_bc0_pt2 \
  --epochs 2 --batch 256 --lr-ft 1e-4 \
  --init-from $DATA/runs/finetune_bc0_lc0/net_e0.pt \
  --mirror --ema-decay 0.999 >> "$LOG" 2>&1
[ -f "$DATA/runs/finetune_bc0_pt2/net_e1.bin" ] || { log "FATAL: fine-tune resume failed"; exit 1; }
log "fine-tune resume complete"

# ---- Stage 2: H2 decisiveness-weighted distilled retrain
wait_gpu 3600
log "H2: decisiveness-weighted retrain on labeled_1m"
$PY "$TRAINER" \
  --shard $DATA/shards/labeled_1m.shard \
  --out $DATA/runs/dist1m_dw \
  --epochs 3 --batch 512 --lr 3e-4 \
  --decisive-weighting \
  --mirror --ema-decay 0.999 >> "$LOG" 2>&1
[ -f "$DATA/runs/dist1m_dw/net_e2.bin" ] || { log "FATAL: H2 training failed"; exit 1; }
log "H2 training complete — SPRT vs SF16-p1"

# ---- Stage 3: H2 SPRT
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=dist1m-dw option.WeightsFile=$DATA/runs/dist1m_dw/net_e2.bin \
  -engine cmd="$SF" name=sf16-p1 option.Threads=1 option.Hash=16 \
  -each proto=uci tc=60+6 plies=1 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 200 -rounds 100 -repeat -concurrency 5 \
  -pgnout file=$DATA/sprt/dist1m-dw-p1.pgn > $DATA/sprt/dist1m-dw-p1.txt 2>&1
grep -q "Finished match" $DATA/sprt/dist1m-dw-p1.txt || { log "FATAL: H2 SPRT failed"; exit 1; }
grep -E "Elo:|Games:" $DATA/sprt/dist1m-dw-p1.txt | head -2 >> "$LOG"
log "=== H2 COMPLETE: uniform baseline -436 vs decisive-weighted above ==="
