#!/bin/bash
# Chain 4: the label-paradigm / encoding / INT8 experiment battery.
# Fires when chain3 (BC-at-40M control) exits.
#  GPU: AV-on-shard (99k steps) then GAB-v3 BC (1 epoch)
#  CPU (concurrent): INT8 replication test of the BC control; ladder matches
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SHARD=$DATA/shards/chessbench_bc40m.shard
LOG=$DATA/chain4.log

log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

# wait for the BC control (chain3) to finish
while pgrep -f "bc_avlabels_chain3" > /dev/null || pgrep -f "runs/bc_avlabels40m" > /dev/null; do
  sleep 300
done
log "BC control finished; battery starts"

# ---- CPU thread 1: INT8 replication test on the BC control ----
(
  mkdir -p $DATA/runs/bc_avlabels40m
  if [ -f $DATA/runs/bc_avlabels40m/net_e0.pt ]; then
    log "INT8: exporting bc40m via export_int8 (calibrated PTQ)"
    $PY $TR/export_int8.py --net $DATA/runs/bc_avlabels40m/net_e0.pt \
      --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
      --shard $SHARD --out $DATA/runs/bc_avlabels40m/net_e0_q.bin >> $LOG 2>&1
    if [ -f $DATA/runs/bc_avlabels40m/net_e0_q.bin ]; then
      log "INT8: parity gate"
      NETQ_BLOB=$DATA/runs/bc_avlabels40m/net_e0_q.bin \
        $PY $TR/../ops/netq_parity.py >> $LOG 2>&1
      log "INT8: FP32-vs-INT8 match of the SAME net (200 games)"
      $TOOLS/fastchess \
        -engine cmd=$ENGINE name=bc40m-fp32 option.WeightsFile=$DATA/runs/bc_avlabels40m/net_e0.bin \
        -engine cmd=$ENGINE name=bc40m-int8 option.WeightsFile=$DATA/runs/bc_avlabels40m/net_e0_q.bin \
        -each proto=uci tc=15+0.2 \
        -openings file=$TOOLS/openings.epd format=epd order=random \
        -games 200 -rounds 100 -repeat -concurrency 6 \
        -pgnout file=$DATA/sprt/bc40m-fp32-vs-int8.pgn > $DATA/sprt/bc40m-fp32-vs-int8.txt 2>&1
      log "INT8: match done"
    fi
  fi
) &

# ---- GPU thread: AV-on-shard (label paradigm A/B vs the BC control) ----
log "starting AV-on-shard training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/av_shard_trainer.py \
  --shard $SHARD --out $DATA/runs/av_shard40m \
  --steps 0 --batch 384 --lr 3e-4 \
  --export-every 25000 >> $LOG 2>&1
log "AV-on-shard training done"

# ---- GPU thread: GAB + rule-state BC (v3-lite, ChessFormer-aligned inputs) ----
mkdir -p $DATA/runs/bc_v3gab40m
log "starting GAB-v3 BC training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $SHARD --out $DATA/runs/bc_v3gab40m \
  --v3 --gab \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --batch 384 --lr 3e-4 --fp32 >> $LOG 2>&1
log "GAB-v3 BC training done"

# ---- ladder matches (CPU): new nets vs the champion ----
match() { # $1 tag, $2 new-bin, $3 extra engine args for the new net (or empty)
  local tag=$1 bin=$2; shift 2
  local txt=$DATA/sprt/$tag.txt
  [ -s "$txt" ] && grep -q "Finished match" "$txt" && return 0
  $TOOLS/fastchess \
    -engine cmd=$ENGINE name=new option.WeightsFile=$bin "$@" \
    -engine cmd=$ENGINE name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 300 -rounds 150 -repeat -concurrency 6 \
    -pgnout file=$DATA/sprt/$tag.pgn > "$txt" 2>&1
  log "match $tag done"
}

match bc40m-vs-champ $DATA/runs/bc_avlabels40m/net_e0_ema.bin
match bc-v3gab-vs-champ $DATA/runs/bc_v3gab40m/net_e0_ema.bin
log "=== chain4 complete ==="
