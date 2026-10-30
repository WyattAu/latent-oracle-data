#!/bin/bash
# Behavioral-cloning bag -> single LOSH shard, 3-phase pipeline:
#   P1: 10 parallel converter slices (no python-chess)
#   P2: lo-data masks per part
#   P3: resolve factorized actions from masks (in place), then stream-merge
#       kept records into the final shard + rebuilt mask; parts deleted.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
BAG=$D/chessbench/behavioral_cloning_data.bag
PDIR=$D/shards/bc_parts
FINAL=$D/shards/chessbench_bc_full.shard
LOG=$D/convert_bc_bag.log
NPROC=6
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

TOTAL=$(python3 - << 'PY'
import struct
with open("/home/wyatt/data/chessbench/behavioral_cloning_data.bag", "rb") as f:
    import os
    f.seek(os.path.getsize("/home/wyatt/data/chessbench/behavioral_cloning_data.bag") - 8)
    (idx,) = struct.unpack("<Q", f.read(8))
print((os.path.getsize("/home/wyatt/data/chessbench/behavioral_cloning_data.bag") - idx) // 8 - 1)
PY
)
NPART=10
PER=$(( (TOTAL + NPART - 1) / NPART ))
log "TOTAL=$TOTAL NPART=$NPART PER=$PER"
mkdir -p $PDIR

log "P1: converting slices"
for ((p=0; p<NPART; p)); do
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 10; done
  S=$((p * PER)); E=$((S + PER))
  nice -n 10 $PY $TR/bc_to_shard_fast.py --bag $BAG --out $PDIR/part$p.shard --start $S --stop $E >> $LOG 2>&1 &
done
wait
log "P1 done"

log "P2: masks"
for p in $(seq 0 $((NPART-1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 10; done
  nice -n 10 /home/wyatt/tools/chess/lo-data masks --in $PDIR/part$p.shard --out $PDIR/part$p.shard.mask >> $LOG 2>&1 &
done
wait
log "P2 done"

log "P3: resolve"
for p in $(seq 0 $((NPART-1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$NPROC" ]; do sleep 10; done
  nice -n 10 $PY $TR/resolve_targets.py --shard $PDIR/part$p.shard --mask $PDIR/part$p.shard.mask >> $LOG 2>&1 &
done
wait
log "P3 done"

log "P4: streaming merge"
$PY - << 'PYEOF' >> $LOG 2>&1
import struct, sys, os
import numpy as np
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE, MaskSidecar
D = "/home/wyatt/data"
PDIR = D + "/shards/bc_parts"
NPART = 10
RECSZ = REC_DTYPE.itemsize
final = open(D + "/shards/chessbench_bc_full.shard.part", "wb")
final.write(struct.pack("<IHHQ", 0x48534F4C, 1, 16, 0))
pay = open(D + "/shards/chessbench_bc_full.shard.mask.part", "wb")
offs_tmp = open(D + "/shards/chessbench_bc_full.mask.offsets.tmp", "wb")
total = 0; kept = 0; pos = 0
for p in range(NPART):
    shard = f"{PDIR}/part{p}.shard"
    sc = MaskSidecar(shard + ".mask")
    n = sc.count
    rf = open(shard, "rb")
    for i in range(n):
        rf.seek(16 + i * RECSZ)
        rec = bytearray(rf.read(RECSZ))
        flat = sc.mask_indices(i).tolist()
        if rec[REC_DTYPE.fields["n_targets"][1]] == 1:
            final.write(bytes(rec)); kept += 1
            ent = b"".join(struct.pack("<H", x) for x in flat)
            pay.append(struct.pack("<B", len(flat))); pay.append(ent)
        else:
            final.write(bytes(rec))
            pay.append(struct.pack("<B", 0))
        offs_tmp.write(struct.pack("<Q", pos)); pos += 1 + 2 * len(flat)
        total += 1
    offs_tmp.write(struct.pack("<Q", pos))
    rf.close(); sc.close()
    os.remove(shard); os.remove(shard + ".mask")
    print(f"merged part {p}: total={total} kept={kept}", flush=True)
final.seek(8); final.write(struct.pack("<Q", total)); final.close()
offs_tmp.close(); pay.close()
with open(D + "/shards/chessbench_bc_full.shard.mask.part", "ab") as g:
    with open(D + "/shards/chessbench_bc_full.mask.offsets.tmp", "rb") as h:
        while True:
            chunk = h.read(1 << 24)
            if not chunk: break
            g.write(chunk)
os.remove(D + "/shards/chessbench_bc_full.mask.offsets.tmp")
# fix the mask header: magic, ver=1, res=0, count=total
with open(D + "/shards/chessbench_bc_full.shard.mask.part", "r+b") as g:
    g.seek(0)
    g.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, total))
print(f"MERGE COMPLETE total={total} kept={kept}", flush=True)
PYEOF
mv $D/shards/chessbench_bc_full.shard.part $FINAL
mv $D/shards/chessbench_bc_full.shard.mask.part $FINAL.mask
rmdir $PDIR 2>/dev/null
log "=== BC fast conversion complete: $FINAL ==="
