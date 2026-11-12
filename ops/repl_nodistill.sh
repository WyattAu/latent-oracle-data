#!/bin/bash
# The INT8-recipe control: 80M x 3 epochs WITHOUT distill. If repl_nodist_q
# ties or beats repl_q, the distill leaves the INT8 recipe; if it loses,
# the distill survives INT8-normalized. Then: ladders vs repl_q and champ_q.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/repl_nodistill.log
R=$DATA/runs/repl_nodistill
COMBINED=$DATA/shards/bc_v2_combined.shard
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
disk_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }
while [ "$(disk_gb)" -lt 15 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done

mkdir -p $R
if [ ! -f "$R/net_e2_ema.bin" ]; then
  log "training: 80M x 3 epochs NO distill (625,000 steps batch 384 fp32) ckpt-every 5000 + resume"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
    --shard $COMBINED --out $R \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 3 --batch 384 --lr 3e-4 --fp32 \
    --ckpt-every 5000 --resume >> "$LOG" 2>&1 || { log "training FAILED"; exit 1; }
  log "training done"
fi
log "exporting INT8"
$PY $DATA_REPO/trainer/export_int8.py --net $R/net_e2_ema.pt --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 --out $R/repl_nodist_q.bin >> "$LOG" 2>&1 || { log "export FAILED"; exit 1; }
log "ladder: repl_nodist_q vs repl_q"
$TOOLS/fastchess -engine cmd=$ENG name=repl_nodist_q option.WeightsFile=$R/repl_nodist_q.bin \
  -engine cmd=$ENG name=repl_q option.WeightsFile=$DATA/runs/replication_80m3ep/net_e2_ema_q.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/repl-nodistq-vs-replq.pgn > $DATA/sprt/repl-nodistq-vs-replq.txt 2>&1
log "vs repl_q done: $(grep -c 'Finished game' $DATA/sprt/repl-nodistq-vs-replq.txt) games"
log "ladder: repl_nodist_q vs champ_q"
$TOOLS/fastchess -engine cmd=$ENG name=repl_nodist_q option.WeightsFile=$R/repl_nodist_q.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/repl-nodistq-vs-champq.pgn > $DATA/sprt/repl-nodistq-vs-champq.txt 2>&1
log "vs champ done: $(grep -c 'Finished game' $DATA/sprt/repl-nodistq-vs-champq.txt) games"
log "=== repl_nodistill complete ==="
