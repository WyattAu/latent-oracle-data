#!/bin/bash
# Chain 7: exposure test on the ChessBench pool. Continue bc40m for 2 more
# epochs (76M more samples; total 3 epochs = 114M) with the value targets
# patched in-place on the shard first (proper wdl triple).
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/chain7.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
OUT=$D/runs/bc40m_e3
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

log "patching wdl triples in-place on chessbench_bc40m.shard"
cd /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
$PY - << 'PYEOF' >> $LOG 2>&1
import numpy as np, struct, sys
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
p = "/home/wyatt/data/shards/chessbench_bc40m.shard"
from format import REC_DTYPE
RECSZ = REC_DTYPE.itemsize
off_wdl = 16 + 0  # computed below from the dtype
names = []
off = 0
for nm in REC_DTYPE.names:
    if nm == "wdl": off_wdl = 16 + off
    dt = REC_DTYPE.fields[nm][0]
    off += dt.itemsize * int(np.prod(dt.shape)) if dt.shape else dt.itemsize
assert off == RECSZ, (off, RECSZ)
with open(p, "r+b") as f:
    f.seek(0, 2)
    size = f.tell()
    n = (size - 16) // RECSZ
    for i in range(n):
        pos = 16 + i * RECSZ + off_wdl
        f.seek(pos)
        w, d, l = struct.unpack("<fff", f.read(12))
        pw = min(max(w, 0.0), 1.0)
        rest = (1.0 - pw) * 0.5
        f.seek(pos)
        f.write(struct.pack("<fff", pw, rest, rest))
        if i % 4000000 == 0:
            print(f"patched {i}/{n}", flush=True)
print("wdl patch done")
PYEOF
log "wdl patched; continuing bc40m for 2 more epochs"
mkdir -p $OUT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $TR/train.py \
  --shard $D/shards/chessbench_bc40m.shard --out $OUT \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --init-from $D/runs/bc_avlabels40m/net_e0.pt --lr-ft 3e-4 \
  --epochs 2 --batch 384 --fp32 >> $LOG 2>&1
log "training done"
run() { $T/fastchess \
    -engine cmd=$ENG name=$2 option.WeightsFile=$3 \
    -engine cmd=$ENG name=$4 option.WeightsFile=$5 \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 150 -rounds 75 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/$1.pgn > $D/sprt/$1.txt 2>&1
  log "match $1 done"; }
run bc40me3-vs-bc40m bc40me3 $OUT/net_e2_ema.bin bc40m $D/runs/bc_avlabels40m/net_e0_ema.bin
run bc40me3-vs-v2pool bc40me3 $OUT/net_e2_ema.bin v2pool $D/runs/bc_v2pool_1ep/net_e0_ema.bin
run bc40me3-vs-champ bc40me3 $OUT/net_e2_ema.bin champ $D/runs/bc/net_q_v2.bin
log "=== chain7 complete ==="
