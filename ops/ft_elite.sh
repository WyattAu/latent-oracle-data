#!/bin/bash
# THE DEEPNOC DELTA #1: fine-tune the champion on the elite (min-elo 2500)
# pool — 2 epochs at lr 1e-4, warm-started from repl_nodist. INT8-first
# ladders: vs the base (the elite-data delta) and vs champ_q.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/ft_elite.log
R=$DATA/runs/ft_elite
E1=$DATA/shards/elite_2026-06/bc.shard
E2=$DATA/shards/elite_2026-05/bc.shard
BASE=$DATA/runs/repl_nodistill/net_e2_ema.pt
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
disk_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }

# wait for ft_sfvalue to release the GPU (poll up to 12h)
for i in $(seq 1 72); do
  systemctl --user is-active --quiet ft-sfvalue || break
  [ $((i % 6)) -eq 1 ] && log "waiting for ft-sfvalue (poll $i)"
  sleep 600
done
while [ "$(disk_gb)" -lt 15 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done

# combine the elite shards
ELITE=$DATA/shards/elite_combined.shard
if [ ! -f "$ELITE" ]; then
  log "combining elite shards"
  $PY - "$E1" "$E2" "$ELITE.tmp" << 'COMBINE_PY'
import struct, sys
HEADER, REC = 16, 64
srcs, dst = sys.argv[1:3], sys.argv[3]
total = 0
with open(dst, "wb") as out:
    out.write(open(srcs[0], "rb").read(HEADER))
    for path in srcs:
        with open(path, "rb") as fh:
            fh.seek(HEADER)
            while True:
                chunk = fh.read(REC * 8192)
                if not chunk:
                    break
                assert len(chunk) % REC == 0, path + ": torn tail"
                out.write(chunk)
                total += len(chunk) // REC
with open(dst, "r+b") as out:
    out.seek(8)
    out.write(struct.pack("<Q", total))
print(f"combined {total} records")
COMBINE_PY
  mv "$ELITE.tmp" "$ELITE"
  log "elite combined: $(stat -c%s "$ELITE") bytes"
fi

mkdir -p $R
if [ ! -f "$R/net_e1_ema.bin" ]; then
  log "GM fine-tune: repl_nodist -> elite pool (2 epochs, lr 1e-4)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
    --shard $ELITE --out $R \
    --init-from $BASE \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 2 --batch 384 --lr 1e-4 --fp32 \
    --ckpt-every 5000 --resume >> "$LOG" 2>&1 || { log "fine-tune FAILED"; exit 1; }
  log "GM fine-tune done"
fi

log "exporting INT8"
$PY $DATA_REPO/trainer/export_int8.py --net $R/net_e1_ema.pt --d 256 --layers 8 \
  --heads 8 --dff 1024 --dpol 128 --out $R/ft_elite_q.bin >> "$LOG" 2>&1 || { log "export FAILED"; exit 1; }
for i in $(seq 1 60); do [ -x "$ENG" ] && break; sleep 60; done

rm -f $DATA/sprt/ftelite-vs-basenodq.{pgn,txt}
log "ladder 1: ft_elite_q vs repl_nodist_q searchless, tc=15+0.2"
$TOOLS/fastchess -engine cmd=$ENG name=ft_elite option.WeightsFile=$R/ft_elite_q.bin \
  -engine cmd=$ENG name=base option.WeightsFile=$DATA/runs/repl_nodistill/repl_nodist_q.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftelite-vs-basenodq.pgn > $DATA/sprt/ftelite-vs-basenodq.txt 2>&1
log "l1 done: $(grep -c 'Finished game' $DATA/sprt/ftelite-vs-basenodq.txt) games"

rm -f $DATA/sprt/ftelite-vs-champq.{pgn,txt}
log "ladder 2: ft_elite_q vs champ_q, tc=15+0.2"
$TOOLS/fastchess -engine cmd=$ENG name=ft_elite option.WeightsFile=$R/ft_elite_q.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/ftelite-vs-champq.pgn > $DATA/sprt/ftelite-vs-champq.txt 2>&1
log "l2 done: $(grep -c 'Finished game' $DATA/sprt/ftelite-vs-champq.txt) games"
log "=== ft_elite complete ==="
