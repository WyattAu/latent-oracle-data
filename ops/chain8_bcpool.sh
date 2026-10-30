#!/bin/bash
# Chain 8: BC-pool matched-exposure control + ladder.
# Fires when the bc-bag conversion completes; trains 98,750 steps x 384 fp32
# (identical exposure to bc40m/v2pool) on the ~450M-record human-move shard,
# then ladders vs v2pool (old pool, same exposure) and champ.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/chain8.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
SHARD=$D/shards/bc_sub/slice0.full.shard
OUT=$D/runs/bcpool_1ep
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while ! grep -q "BC fast conversion complete" $D/convert_bc_bag.log 2>/dev/null; do sleep 300; done
while ! grep -q "chain7 service complete" $D/chain7.log 2>/dev/null; do sleep 300; done
log "shard ready + GPU free (chain7 done); training matched-exposure control"
mkdir -p $OUT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $SHARD --out $OUT \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --steps 98750 --batch 384 --lr 3e-4 --fp32 >> $LOG 2>&1
log "training done"
NET=$OUT/net_e0_ema.bin
run() { $T/fastchess \
    -engine cmd=$ENG name=$2 option.WeightsFile=$3 \
    -engine cmd=$ENG name=$4 option.WeightsFile=$5 \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 150 -rounds 75 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/$1.pgn > $D/sprt/$1.txt 2>&1
  log "match $1 done"; }
run bcpool-vs-v2pool bcpool $NET v2pool $D/runs/bc_v2pool_1ep/net_e0_ema.bin
run bcpool-vs-champ bcpool $NET champ $D/runs/bc/net_q_v2.bin
log "=== chain8 complete ==="
