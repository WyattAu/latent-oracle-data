#!/bin/bash
# Mirror-augmentation A/B: same slice0, 98,750 steps, --rank-mirror.
# Waits for chain8ladder (the baseline ladder) to finish.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data/shards
R=/home/wyatt/data/runs/bcpool_mirror
LOG=$D/../bcpool_mirror.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while systemctl --user is-active --quiet chain8ladder; do sleep 120; done
# fire the moment the marquee's TRAINING is done (its CPU ladders don't block us)
while ! grep -q "e3 training done" /home/wyatt/data/bc_marquee.log 2>/dev/null; do sleep 120; done
log "marquee training done; mirror ablation training (GPU now free)"
mkdir -p $R
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
  --shard $D/bc_sub/slice0.full.shard --out $R \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 1 --steps 98750 --batch 384 --lr 3e-4 --fp32 --rank-mirror >> $LOG 2>&1
log "mirror training done; laddering vs bcpool (no-mirror control) and champ"
NET=$R/net_e0_ema.bin
$T/fastchess -engine cmd=$ENG name=bcpoolmir option.WeightsFile=$NET \
  -engine cmd=$ENG name=bcpool option.WeightsFile=/home/wyatt/data/runs/bcpool_1ep/net_e0_ema.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpoolmir-vs-bcpool.pgn > $D/sprt/bcpoolmir-vs-bcpool.txt 2>&1
$T/fastchess -engine cmd=$ENG name=bcpoolmir option.WeightsFile=$NET \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpoolmir-vs-champ.pgn > $D/sprt/bcpoolmir-vs-champ.txt 2>&1
log "=== mirror ablation complete ==="
