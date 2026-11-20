#!/bin/bash
# DEEPNOC DELTA #2: policy QAT. Same recipe as the replication (80M x 3ep)
# with --qat (projection QAT: the int8-linears' weights projected onto the
# s8 grid after every step — the int8 serving weights become training-
# native instead of post-hoc rounded).
# PRE-REGISTRATION: QAT-trained INT8 vs post-hoc-INT8 repl_nodist_q at
# matched serving precision. CONFIRMS if QAT wins >= +40. FALSIFIES if
# the tie band (-28..+28) holds. The QAT arm also ladders vs champ_q.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/ft_qat.log
R=$DATA/runs/ft_qat
V3=$DATA/shards/bc_v3_combined.shard
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
disk_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

mkdir -p $R
if [ ! -f "$R/net_e2_ema.bin" ]; then
  while [ "$(disk_gb)" -lt 15 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done
  log "training: 160M x 3 epochs + distill + policy QAT (1,250,000 steps batch 384 fp32) ckpt-every 5000 + resume"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
    --shard $V3 --out $R \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 3 --batch 384 --lr 3e-4 --fp32 \
    --distill1ply 16 --distill-weight 0.1 --qat \
    --ckpt-every 5000 --resume >> "$LOG" 2>&1 || { log "training FAILED"; exit 1; }
  log "training done"
fi

log "exporting INT8"
$PY $DATA_REPO/trainer/export_int8.py --net $R/net_e2_ema.pt --d 256 --layers 8 \
  --heads 8 --dff 1024 --dpol 128 --out $R/ft_qat_q.bin >> "$LOG" 2>&1 || { log "export FAILED"; exit 1; }
for i in $(seq 1 60); do [ -x "$ENG" ] && break; sleep 60; done

BASEQ=$DATA/runs/repl_nodistill/repl_nodist_q.bin
rm -f $DATA/sprt/ftqat-vs-replq.{pgn,txt}
log "ladder 1: ft_qat_q vs repl_nodist_q (matched INT8: the QAT delta)"
$TOOLS/fastchess -engine cmd=$ENG name=ft_qat option.WeightsFile=$R/ft_qat_q.bin \
  -engine cmd=$ENG name=repl_q option.WeightsFile=$BASEQ \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftqat-vs-replq.pgn > $DATA/sprt/ftqat-vs-replq.txt 2>&1
log "l1 done: $(grep -c 'Finished game' $DATA/sprt/ftqat-vs-replq.txt) games"

rm -f $DATA/sprt/ftqat-vs-champq.{pgn,txt}
log "ladder 2: ft_qat_q vs champ_q"
$TOOLS/fastchess -engine cmd=$ENG name=ft_qat option.WeightsFile=$R/ft_qat_q.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftqat-vs-champq.pgn > $DATA/sprt/ftqat-vs-champq.txt 2>&1
log "l2 done: $(grep -c 'Finished game' $DATA/sprt/ftqat-vs-champq.txt) games"
log "=== ft_qat complete ==="
