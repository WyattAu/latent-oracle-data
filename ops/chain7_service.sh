#!/bin/bash
# Exposure test as a service: bc40m + 2 epochs from net_e0.pt, then ladder.
# (No wdl patch: A/B proved it irrelevant to policy; engine ignores value.)
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/chain7.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
OUT=$D/runs/bc40m_e3
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
[ -f "$D/sprt/bc40me3-vs-champ.pgn" ] && { log "chain7 already complete"; exit 0; }
mkdir -p $OUT
if [ ! -f $OUT/net_e1_ema.bin ] && [ ! -f $OUT/net_e0_ema.bin ]; then
  log "service: training 2 epochs from net_e0.pt"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
    --shard $D/shards/chessbench_bc40m.shard --out $OUT \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --init-from $D/runs/bc_avlabels40m/net_e0.pt --lr-ft 3e-4 \
    --epochs 2 --batch 384 --fp32 >> $LOG 2>&1
  log "service: training done"
fi
NET=$OUT/net_e1_ema.bin
[ -f "$NET" ] || NET=$OUT/net_e0_ema.bin
run() { $T/fastchess \
    -engine cmd=$ENG name=$2 option.WeightsFile=$3 \
    -engine cmd=$ENG name=$4 option.WeightsFile=$5 \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 150 -rounds 75 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/$1.pgn > $D/sprt/$1.txt 2>&1
  log "match $1 done"; }
[ -f $D/sprt/bc40me3-vs-bc40m.pgn ] || run bc40me3-vs-bc40m bc40me3 "$NET" bc40m $D/runs/bc_avlabels40m/net_e0_ema.bin
[ -f $D/sprt/bc40me3-vs-v2pool.pgn ] || run bc40me3-vs-v2pool bc40me3 "$NET" v2pool $D/runs/bc_v2pool_1ep/net_e0_ema.bin
[ -f $D/sprt/bc40me3-vs-champ.pgn ] || run bc40me3-vs-champ bc40me3 "$NET" champ $D/runs/bc/net_q_v2.bin
log "=== chain7 service complete ==="
