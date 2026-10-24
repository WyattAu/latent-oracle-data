#!/bin/bash
# Chain 6: matched-exposure data-pool control.
# bc_v2_combined (old pool, ~122M records) trained for the SAME 98,750 steps
# x 384 batch as bc40m — isolates DATA POOL from EXPOSURE.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/chain6.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
OUT=$D/runs/bc_v2pool_1ep
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

mkdir -p $OUT
log "starting old-pool matched-exposure control (98750 steps, batch 384, fp32)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $D/shards/bc_v2_combined.shard --out $OUT \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --steps 98750 --batch 384 --lr 3e-4 --fp32 >> $LOG 2>&1
log "training done"
# ladder: bc40m vs new control; champ vs new control
run() { $T/fastchess \
    -engine cmd=$ENG name=$2 option.WeightsFile=$3 \
    -engine cmd=$ENG name=$4 option.WeightsFile=$5 \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 150 -rounds 75 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/$1.pgn > $D/sprt/$1.txt 2>&1
  log "match $1 done"; }
run bc40m-vs-v2pool bc40m $D/runs/bc_avlabels40m/net_e0_ema.bin v2pool $OUT/net_e0_ema.bin
run v2pool-vs-champ v2pool $OUT/net_e0_ema.bin champ $D/runs/bc/net_q_v2.bin
log "=== chain6 complete ==="
