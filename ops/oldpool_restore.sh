#!/bin/bash
# OLD-POOL RESTORE + DISTILL: the value-realism x method compound.
# Sources survive (lichess 2026-07/08 zst). Restore bc_v2_combined (~130M
# records with REAL game results), train the bcpool_distill recipe on it
# (98,750 steps batch 384 fp32 + Loop B distill), ladder vs champ and vs
# bcpooldist (does the +147 transfer across data pools?).
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
PY=/home/wyatt/.venvs/chess/bin/python
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
LOG=$DATA/oldpool_restore.log
R=$DATA/runs/oldpool_distill
COMBINED=$DATA/shards/bc_v2_combined.shard
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

# ---- 1. re-shard the two months (skip-guarded)
for MONTH in 2026-07 2026-08; do
  SRC=$DATA/lichess/lichess_db_standard_rated_${MONTH}.pgn.zst
  DST=$DATA/shards/big_${MONTH}
  [ -f "$SRC" ] || { log "FATAL: missing source $SRC"; exit 1; }
  if [ -f "$DST/bc.shard" ] && [ "$(stat -c%s "$DST/bc.shard")" -gt 100000000 ]; then
    log "skip: $MONTH already sharded"
  else
    log "sharding $MONTH (nice, 40M cap, min-elo 2000)"
    nice -n 19 $TOOLS/lo-data shard --in "$SRC" --out "$DST" \
      --min-elo 2000 --min-ply 8 --max-ply 120 \
      --max-positions 40000000 >> "$LOG" 2>&1 || { log "FATAL: shard $MONTH failed"; exit 1; }
    log "$MONTH done: $(stat -c%s "$DST/bc.shard" 2>/dev/null || echo 0) bytes"
  fi
done

# ---- 2. combine (one header, patched count)
if [ ! -f "$COMBINED" ]; then
  log "combining shards"
  $PY - "$DATA/shards/big_2026-07/bc.shard" "$DATA/shards/big_2026-08/bc.shard" \
       "$COMBINED.tmp" << 'COMBINE_PY'
import os, struct, sys
HEADER, REC = 16, 64
srcs, dst = sys.argv[1:3], sys.argv[2 + 1]
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
out = open(dst, "r+b")
out.seek(8)
out.write(struct.pack("<Q", total))
out.close()
print(f"combined {total} records")
COMBINE_PY
  mv "$COMBINED.tmp" "$COMBINED"
  log "combined: $(stat -c%s "$COMBINED") bytes"
fi

# ---- 3. mask (train.py's ensure_mask does it, but pre-verify: the
# integrity check regenerates anyway — just let the trainer handle it)

# ---- 4. train: the bcpool_distill recipe on the restored pool
mkdir -p $R
if [ ! -f "$R/net_e0_ema.bin" ]; then
  log "training: old pool + distill (98,750 steps batch 384 fp32)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY \
    $DATA_REPO/trainer/train.py \
    --shard $COMBINED --out $R \
    --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
    --epochs 1 --steps 98750 --batch 384 --lr 3e-4 --fp32 \
    --distill1ply 16 --distill-weight 0.1 >> "$LOG" 2>&1 || { log "training FAILED"; exit 1; }
  log "training done"
fi

# ---- 5. ladders
NET=$R/net_e0_ema.bin
log "ladder: oldpool_distill vs champ"
$TOOLS/fastchess -engine cmd=$ENG name=oldpool_distill option.WeightsFile=$NET \
  -engine cmd=$ENG name=champ option.WeightsFile=$DATA/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/oldpool-distill-vs-champ.pgn > $DATA/sprt/oldpool-distill-vs-champ.txt 2>&1
log "vs champ done: $(grep -c 'Finished game' $DATA/sprt/oldpool-distill-vs-champ.txt) games"
log "ladder: oldpool_distill vs bcpooldist"
$TOOLS/fastchess -engine cmd=$ENG name=oldpool_distill option.WeightsFile=$NET \
  -engine cmd=$ENG name=bcpooldist option.WeightsFile=$DATA/runs/bcpool_distill/net_e0_ema.bin \
  -each proto=uci tc=15+0.2 -openings file=$TOOLS/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$DATA/sprt/oldpool-distill-vs-bcpooldist.pgn > $DATA/sprt/oldpool-distill-vs-bcpooldist.txt 2>&1
log "vs bcpooldist done: $(grep -c 'Finished game' $DATA/sprt/oldpool-distill-vs-bcpooldist.txt) games"
log "=== oldpool restore complete ==="
