#!/bin/bash
# AV phase (Phase 2): two-stage action-value training on mixed-depth labels.
#   stage 1: d10 4M pretrain  — v3 arch, Muon, WSD, mirror, EMA, TB sidecar,
#                              decisive weighting, opening upweight. No RCT.
#   verdict A: the stage-1 d10 bundle vs the BC chain (fires as soon as the
#     d10 shard exists -- no need to wait for d16)
#   phase B: d16 fine-tune + RCT (recycle 2) + QAT, then verdict B
#                              (RCT doubles trunk activations).
# Then: SPRT A/B vs the BC chain's net_best + conversion metrics (E4).
set -u
DATA=/home/wyatt/data
PY=/tmp/opencode/venv/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/av_phase.log
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

log "=== AV phase armed (two verdicts: d10 bundle, then +RCT/QAT) ==="
# Split into two verdicts. Waiting for the whole labeler put ~20 h of d16
# labeling between the first AV signal and any verdict; stage 1 only needs the
# d10 shard. It also isolates the two halves of the mechanism: verdict A is
# "does the AV bundle help at all", verdict B is "does the RCT/QAT refinement
# on d16 add anything".
while pgrep -f "moonshot_queue.sh" > /dev/null; do sleep 600; done
log "waiting for the d10 shard"
while [ ! -f "$DATA/shards/labeled_d10_4m.shard" ]; do sleep 600; done
# keep_best starts the moment labeling ends and publishes net_best.pt once the
# bc_v1 epochs are ranked. Wait for the ARTIFACT rather than for the script to
# exit: the gate also measures bc_v1f and dist1m_dw for the record, and waiting
# on those would put hours of pure information between labeling and the AV run.
while [ ! -f "$DATA/runs/bc_v1/net_best.pt" ]; do sleep 600; done
log "gate winner available: net_best.pt"
while [ "$(gpu_free_mb)" -lt 3600 ]; do sleep 300; done
cd "$TR"

# ---- Syzygy sidecars for both labeled shards (exact endgame labels, E1)
if [ ! -f "$DATA/shards/labeled_d10_4m.shard.tb.jsonl" ]; then
  log "E1: syzygy-rescoring d10 shard"
  $PY make_tb_labels.py --shard $DATA/shards/labeled_d10_4m.shard \
    --out $DATA/shards/labeled_d10_4m.shard.tb.jsonl >> "$LOG" 2>&1
fi

# keep_best.sh now writes the gate winner's checkpoint as net_best.pt
# (it used to look for net_best.bin.pt, which never existed, so the phase
# silently fell back to net_e2.pt and ignored the gate ranking).
BASE=$DATA/runs/bc_v1/net_best.pt
[ -f "$BASE" ] || BASE=$DATA/runs/bc_v1/net_e2.pt

# ---- Stage 1: d10 pretrain (batch 512, no RCT)
log "AV stage 1: d10 4M pretrain (v3, muon, wsd)"
$PY train.py \
  --shard $DATA/shards/labeled_d10_4m.shard \
  --out $DATA/runs/av_v3 \
  --epochs 2 --batch 512 --optimizer muon --sched wsd \
  --gab --v3 --init-from $BASE \
  --mirror --ema-decay 0.999 \
  --decisive-weighting --tb-sidecar $DATA/shards/labeled_d10_4m.shard.tb.jsonl \
  --opening-weight 1.5 \
  >> "$LOG" 2>&1
[ -f "$DATA/runs/av_v3/net_e1.pt" ] || { log "FATAL: AV stage 1 failed"; exit 1; }
log "AV stage 1 complete"

# ---- Export the STAGE-1 v3 blobs for verdict A (stage 2 lives in phase B)
$PY - << 'PYEOF' >> "$LOG" 2>&1
import sys, torch
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet
m = ChessNet(v3=True)
m.load_state_dict(torch.load("/home/wyatt/data/runs/av_v3/net_e1.pt",
                             map_location="cpu", weights_only=True))
m.export_blob("/home/wyatt/data/runs/av_v3/av_e1.bin")
print("AV v3 stage-1 blob exported")
PYEOF
$PY export_int8.py --net $DATA/runs/av_v3/net_e1.pt --v3 \
  --shard $DATA/shards/labeled_d10_4m.shard --calib 2048 \
  --out $DATA/runs/av_v3/av_e1_q.bin >> "$LOG" 2>&1

# ---- VERDICT A: the AV bundle (d10 fine-tune) vs the BC chain best
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=bc-best option.WeightsFile=$DATA/runs/bc_v1/net_best.bin \
  -engine cmd="$ENGINE" name=av-v3 option.WeightsFile=$DATA/runs/av_v3/av_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 400 -rounds 200 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/av-vs-bc.pgn > $DATA/sprt/av-vs-bc.txt 2>&1
grep -E "Elo:|Games:" $DATA/sprt/av-vs-bc.txt | head -2 >> "$LOG"
$PY conversion_metrics.py $DATA/sprt/av-vs-bc.pgn >> "$LOG" 2>&1
# one-command verdict record: final game-level Elo + pentanomial pair model +
# conversion, so the ledger entry never has to be reconstructed by hand
log "--- AV verdict summary ---"
$PY analyze_verdicts.py $DATA/sprt/av-vs-bc \
  --net-name "AV-v3 stage 1 (d10 bundle) vs BC gate winner" >> "$LOG" 2>&1
log "=== VERDICT A complete (AV d10 bundle vs BC) ==="

# ---- Phase B: the d16 refinement, once its labels exist
log "waiting for the d16 shard (labeling continues in the background)"
while [ ! -f "$DATA/shards/labeled_d16_1m.shard" ]; do sleep 600; done
if [ ! -f "$DATA/shards/labeled_d16_1m.shard.tb.jsonl" ]; then
  log "E1: syzygy-rescoring d16 shard"
  $PY make_tb_labels.py --shard $DATA/shards/labeled_d16_1m.shard \
    --out $DATA/shards/labeled_d16_1m.shard.tb.jsonl >> "$LOG" 2>&1
fi
# give the labeler the CPU back while it finishes d16
while pgrep -f "label_mixed.sh" > /dev/null; do sleep 600; done
while [ "$(gpu_free_mb)" -lt 3600 ]; do sleep 300; done

log "AV phase B: d16 fine-tune (RCT, QAT)"
$PY train.py \
  --shard $DATA/shards/labeled_d16_1m.shard \
  --out $DATA/runs/av_v3_s2 \
  --epochs 2 --batch 256 --optimizer muon \
  --gab --v3 --init-from $DATA/runs/av_v3/net_e1.pt \
  --mirror --ema-decay 0.999 \
  --decisive-weighting --tb-sidecar $DATA/shards/labeled_d16_1m.shard.tb.jsonl \
  --opening-weight 1.5 \
  --recycle 2 --rct-lambda 0.5 --qat \
  >> "$LOG" 2>&1
[ -f "$DATA/runs/av_v3_s2/net_e1.pt" ] || { log "FATAL: AV phase B failed"; exit 1; }

$PY - << 'PYEXPORT' >> "$LOG" 2>&1
import sys, torch
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet
m = ChessNet(v3=True)
m.load_state_dict(torch.load("/home/wyatt/data/runs/av_v3_s2/net_e1.pt",
                             map_location="cpu", weights_only=True))
m.export_blob("/home/wyatt/data/runs/av_v3_s2/av_e1.bin")
print("phase B blob exported")
PYEXPORT

$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=av-stage1 option.WeightsFile=$DATA/runs/av_v3/av_e1.bin \
  -engine cmd="$ENGINE" name=av-stage2 option.WeightsFile=$DATA/runs/av_v3_s2/av_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 300 -rounds 150 -repeat -concurrency 4 \
  -pgnout file=$DATA/sprt/av-s2-vs-s1.pgn > $DATA/sprt/av-s2-vs-s1.txt 2>&1
grep -E "Elo:|Games:" $DATA/sprt/av-s2-vs-s1.txt | head -2 >> "$LOG"
$PY analyze_verdicts.py $DATA/sprt/av-s2-vs-s1 \
  --net-name "AV-v3 +d16 RCT/QAT vs AV-v3 stage 1" >> "$LOG" 2>&1
log "=== AV phase complete — verdict A (bundle) and B (refinement) in $LOG ==="
