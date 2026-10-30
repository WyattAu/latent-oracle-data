#!/bin/bash
# THE MARQUEE: one continuous pass over the 4 BC sub-slices (~450M records).
# Each slice trains --init-from the previous net; final ladder vs champ.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data/shards
R=/home/wyatt/data/runs/bc_marquee
LOG=$D/../bc_marquee.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
mkdir -p $R

declare -a SLICES=("$D/bc_sub/slice0.full.shard" "$D/bc_sub/slice1.full.shard" "$D/bc_sub/slice2.full.shard" "$D/bc_sub/slice3.full.shard")
declare -a STEPS=(117187 117187 101562 75499)
INIT=""

for idx in 0 1 2 3; do
  SH=${SLICES[$idx]}; ST=${STEPS[$idx]}
  OUT=$R/e$idx
  NET=$OUT/net_e0_ema.bin
  if [ -f "$NET" ]; then log "e$idx already trained ($NET)"; INIT=$OUT/net_e0.pt; continue; fi
  log "=== epoch $idx: $SH, steps=$ST, init=${INIT:-scratch} ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
    --shard $SH --out $OUT \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 1 --steps $ST --batch 384 --lr 3e-4 --fp32 \
    ${INIT:+--init-from $INIT --lr-ft 3e-4} >> $LOG 2>&1 || { log "e$idx training FAILED"; exit 1; }
  log "e$idx training done"
  INIT=$OUT/net_e0.pt
done

log "marquee training complete; laddering vs champ"
NET=$R/e3/net_e0_ema.bin
$T/fastchess -engine cmd=$ENG name=bcpool450 option.WeightsFile=$NET \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 250 -rounds 125 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpool450-vs-champ.pgn > $D/sprt/bcpool450-vs-champ.txt 2>&1
$T/fastchess -engine cmd=$ENG name=bcpool450 option.WeightsFile=$NET \
  -engine cmd=/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2 name=sf2000 option.UCI_LimitStrength=true option.UCI_Elo=2000 \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 250 -rounds 125 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpool450-vs-sf2000.pgn > $D/sprt/bcpool450-vs-sf2000.txt 2>&1
log "=== MARQUEE COMPLETE ==="
