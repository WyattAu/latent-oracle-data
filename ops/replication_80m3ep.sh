#!/bin/bash
# THE REPLICATION: the champ recipe on the restored pool.
# bc_v2_combined (80M real-result records) x 3 epochs + Loop B distill
# = 625,000 steps at batch 384 (53% of the champ's 450M-sample budget).
# Then: ladders vs champ and vs v2pool.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/replication.log
R=$DATA/runs/replication_80m3ep
COMBINED=$DATA/shards/bc_v2_combined.shard
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

mkdir -p $R
if [ ! -f "$R/net_e2_ema.bin" ]; then
  log "training: 80M x 3 epochs + distill (625,000 steps batch 384 fp32)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
    $DATA_REPO/trainer/train.py \
    --shard $COMBINED --out $R \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 3 --batch 384 --lr 3e-4 --fp32 \
    --distill1ply 16 --distill-weight 0.1 >> "$LOG" 2>&1 || { log "training FAILED"; exit 1; }
  log "training done"
fi

NET=$R/net_e2_ema.bin
log "ladder: replication vs champ"
$TOOLS/fastchess -engine cmd=$ENG name=replication option.WeightsFile=$NET \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/replication-vs-champ.pgn > $DATA/sprt/replication-vs-champ.txt 2>&1
log "vs champ done: $(grep -c 'Finished game' $DATA/sprt/replication-vs-champ.txt) games"
log "ladder: replication vs v2pool"
$TOOLS/fastchess -engine cmd=$ENG name=replication option.WeightsFile=$NET \
  -engine cmd=$ENG name=v2pool option.WeightsFile=$DATA/runs/bc_v2pool_1ep/net_e0_ema.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/replication-vs-v2pool.pgn > $DATA/sprt/replication-vs-v2pool.txt 2>&1
log "vs v2pool done: $(grep -c 'Finished game' $DATA/sprt/replication-vs-v2pool.txt) games"
log "=== replication complete ==="
