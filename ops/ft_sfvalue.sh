#!/bin/bash
# SF-VALUE FINE-TUNE EXPERIMENT: does a value head fine-tuned on deep-SF
# win_prob (instead of raw game results) make the AB search add strength?
# Chain: wait for av_value.shard -> mask -> fine-tune repl_nodist 1 epoch
# at lr 1e-4 -> INT8 export -> 3 ladders (AB vs searchless, AB vs champ_q,
# searchless vs champ_q — all INT8, matched precision).
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/ft_sfvalue.log
R=$DATA/runs/ft_sfvalue
VSH=$DATA/shards/av_value.shard
BASE=$DATA/runs/repl_nodistill/net_e2_ema.pt
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
disk_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

# 1. wait for the AV conversion (poll up to 6h)
for i in $(seq 1 36); do
  [ -f "$VSH" ] && [ "$(stat -c%s "$VSH")" -gt 2000000000 ] && break
  [ $((i % 6)) -eq 1 ] && log "waiting for av_value.shard (poll $i)"
  sleep 600
done
[ -f "$VSH" ] || { log "FATAL: av_value.shard never appeared"; exit 1; }
while [ "$(disk_gb)" -lt 15 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done

# 2. fine-tune the champion on SF value targets (1 epoch, low lr)
mkdir -p $R
if [ ! -f "$R/net_e0_ema.bin" ]; then
  log "fine-tune: repl_nodist -> SF value targets (1 epoch, lr 1e-4)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
    --shard $VSH --out $R \
    --init-from $BASE \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 1 --batch 384 --lr 3e-4 --fp32 --freeze-policy --policy-weight 0 \
    --ckpt-every 5000 --resume >> "$LOG" 2>&1 || { log "fine-tune FAILED"; exit 1; }
  log "fine-tune done"
fi

# 3. INT8 export
log "exporting INT8"
$PY $DATA_REPO/trainer/export_int8.py --net $R/net_e0_ema.pt --d 256 --layers 8 \
  --heads 8 --dff 1024 --dpol 128 --out $R/ft_sfvalue_q.bin >> "$LOG" 2>&1 || { log "export FAILED"; exit 1; }

# 4. ladders (INT8, matched precision)
Q=$R/ft_sfvalue_q.bin
BASEQ=$DATA/runs/repl_nodistill/repl_nodist_q.bin
CHAMPQ=$DATA/runs/bc/net_q_v2.bin
for i in $(seq 1 60); do [ -x "$ENG" ] && break; sleep 60; done

rm -f $DATA/sprt/ftsf-AB-vs-policy.{pgn,txt}
log "ladder 1: ft_sfvalue_q +AB(400k/8s) vs ft_sfvalue_q searchless, tc=60+1"
$TOOLS/fastchess -engine cmd=$ENG name=ft_ab option.WeightsFile=$Q option.SearchAB=true option.ABNodes=400000 option.ABTimeMs=8000 \
  -engine cmd=$ENG name=ft_pol option.WeightsFile=$Q \
  -each proto=uci tc=60+1 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 50 -rounds 25 -repeat -concurrency 4 \
  -pgnout file=$DATA/sprt/ftsf-AB-vs-policy.pgn > $DATA/sprt/ftsf-AB-vs-policy.txt 2>&1
log "l1 done: $(grep -c 'Finished game' $DATA/sprt/ftsf-AB-vs-policy.txt) games"

rm -f $DATA/sprt/ftsf-vs-champq.{pgn,txt}
log "ladder 2: ft_sfvalue_q searchless vs champ_q, tc=15+0.2"
$TOOLS/fastchess -engine cmd=$ENG name=ft_pol option.WeightsFile=$Q \
  -engine cmd=$ENG name=champ option.WeightsFile=$CHAMPQ \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftsf-vs-champq.pgn > $DATA/sprt/ftsf-vs-champq.txt 2>&1
log "l2 done: $(grep -c 'Finished game' $DATA/sprt/ftsf-vs-champq.txt) games"

rm -f $DATA/sprt/ftsf-vs-basenodq.{pgn,txt}
log "ladder 3: ft_sfvalue_q vs repl_nodist_q searchless, tc=15+0.2"
$TOOLS/fastchess -engine cmd=$ENG name=ft_pol option.WeightsFile=$Q \
  -engine cmd=$ENG name=base option.WeightsFile=$BASEQ \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftsf-vs-basenodq.pgn > $DATA/sprt/ftsf-vs-basenodq.txt 2>&1
log "l3 done: $(grep -c 'Finished game' $DATA/sprt/ftsf-vs-basenodq.txt) games"
log "=== ft_sfvalue experiment complete ==="
