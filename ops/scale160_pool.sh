#!/bin/bash
# THE SCALE-UP: 160M-pool (4 lichess months) x 3 epochs + distill = 480M
# samples (champ's budget, 2x the data diversity). Decisive run for
# champ-parity-or-better. Resume-capable + disk-guarded + INT8-first ladders.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/scale160.log
R=$DATA/runs/scale160_3ep
V3=$DATA/shards/bc_v3_combined.shard
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
disk_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }
while [ "$(disk_gb)" -lt 20 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done

# 1. wait for the downloads
for i in $(seq 1 240); do
  grep -q "scale-month downloads complete" $DATA/fetch_scale.log 2>/dev/null && break
  [ $((i % 6)) -eq 1 ] && log "waiting for downloads (poll $i)"
  sleep 600
done
grep -q "scale-month downloads complete" $DATA/fetch_scale.log 2>/dev/null || { log "FATAL: downloads never completed"; exit 1; }

# 2. shard the new months (skip-guarded)
for MONTH in 2026-06 2026-05; do
  SRC=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  DST=$DATA/shards/big_${MONTH}
  [ -f "$SRC" ] || { log "FATAL: missing $SRC"; exit 1; }
  if [ -f "$DST/bc.shard" ] && [ "$(stat -c%s "$DST/bc.shard")" -gt 2000000000 ]; then
    log "skip: $MONTH sharded"
  else
    log "sharding $MONTH"
    nice -n 19 $TOOLS/lo-data shard --in "$SRC" --out "$DST" \
      --min-elo 2000 --min-ply 8 --max-ply 120 \
      --max-positions 40000000 >> "$LOG" 2>&1 || { log "FATAL: shard $MONTH"; exit 1; }
    log "$MONTH: $(stat -c%s "$DST/bc.shard" 2>/dev/null || echo 0) bytes"
  fi
done

# 3. combine v2 (80M) + the two new months -> 160M
if [ ! -f "$V3" ]; then
  log "combining 80M + 40M + 40M -> $V3"
  $PY - "$DATA/shards/bc_v2_combined.shard" "$DATA/shards/big_2026-06/bc.shard" \
       "$DATA/shards/big_2026-05/bc.shard" "$V3.tmp" << 'COMBINE_PY'
import struct, sys
HEADER, REC = 16, 64
srcs, dst = sys.argv[1:4], sys.argv[4]
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
  mv "$V3.tmp" "$V3"
  log "v3 combined: $(stat -c%s "$V3") bytes"
fi

# 4. train 160M x 3 epochs + distill (1,250,000 steps at batch 384)
mkdir -p $R
if [ ! -f "$R/net_e2_ema.bin" ]; then
  while [ "$(disk_gb)" -lt 15 ]; do log "disk wait ($(disk_gb)G)"; sleep 600; done
  log "training: 160M x 3 epochs + distill (1,250,000 steps batch 384 fp32) ckpt-every 5000 + resume"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
    --shard $V3 --out $R \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 3 --batch 384 --lr 3e-4 --fp32 \
    --distill1ply 16 --distill-weight 0.1 \
    --ckpt-every 5000 --resume >> "$LOG" 2>&1 || { log "training FAILED"; exit 1; }
  log "training done"
fi

# 5. INT8 export + ladders
log "exporting INT8"
$PY $DATA_REPO/trainer/export_int8.py --net $R/net_e2_ema.pt --d 256 --layers 8 \
  --heads 8 --dff 1024 --dpol 128 --out $R/scale160_q.bin >> "$LOG" 2>&1 || { log "export FAILED"; exit 1; }
for i in $(seq 1 60); do [ -x "$ENG" ] && break; sleep 60; done
log "ladder: scale160_q vs champ_q"
$TOOLS/fastchess -engine cmd=$ENG name=scale160_q option.WeightsFile=$R/scale160_q.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/scale160q-vs-champq.pgn > $DATA/sprt/scale160q-vs-champq.txt 2>&1
log "vs champ done: $(grep -c 'Finished game' $DATA/sprt/scale160q-vs-champq.txt) games"
log "ladder: scale160_q vs repl_q"
$TOOLS/fastchess -engine cmd=$ENG name=scale160_q option.WeightsFile=$R/scale160_q.bin \
  -engine cmd=$ENG name=repl_q option.WeightsFile=$DATA/runs/replication_80m3ep/net_e2_ema_q.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/scale160q-vs-replq.pgn > $DATA/sprt/scale160q-vs-replq.txt 2>&1
log "vs repl_q done: $(grep -c 'Finished game' $DATA/sprt/scale160q-vs-replq.txt) games"
log "=== scale160 complete ==="
