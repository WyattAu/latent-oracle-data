#!/bin/bash
# Loop B GPU ablation: slice0, 98,750 steps, --distill1ply 64, λ=0.1.
# Waits for chain8mirror (the mirror A/B) to finish.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data/shards
R=/home/wyatt/data/runs/bcpool_distill
LOG=$D/../bcpool_distill.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while systemctl --user is-active --quiet chain8mirror; do sleep 300; done
log "mirror ablation done; distill ablation training"
mkdir -p $R
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
  --shard $D/bc_sub/slice0.full.shard --out $R \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --steps 98750 --batch 384 --lr 3e-4 --fp32 \
  --distill1ply 64 --distill-weight 0.1 >> $LOG 2>&1
log "distill training done; laddering vs bcpool and champ"
NET=$R/net_e0_ema.bin
$T/fastchess -engine cmd=$ENG name=bcpooldist option.WeightsFile=$NET \
  -engine cmd=$ENG name=bcpool option.WeightsFile=/home/wyatt/data/runs/bcpool_1ep/net_e0_ema.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpooldist-vs-bcpool.pgn > $D/sprt/bcpooldist-vs-bcpool.txt 2>&1
$T/fastchess -engine cmd=$ENG name=bcpooldist option.WeightsFile=$NET \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpooldist-vs-champ.pgn > $D/sprt/bcpooldist-vs-champ.txt 2>&1
log "=== distill ablation complete ==="
