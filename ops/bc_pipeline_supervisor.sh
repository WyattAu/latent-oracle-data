#!/bin/bash
# Resumable supervisor for the BC-bag conversion. One tick per invocation;
# systemd Restart=on-failure relaunches after any kill.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
BAG=$D/chessbench/behavioral_cloning_data.bag
PDIR=$D/shards/bc_parts
FINAL=$D/shards/chessbench_bc_full.shard
LOG=$D/convert_bc_bag.log
NPART=10
TOTAL=527633464
PER=$(( (TOTAL + NPART - 1) / NPART ))
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

[ -f "$FINAL" ] && { log "final shard exists; supervisor done"; exit 0; }
# Kill orphaned workers from previous generations (they truncate the same
# part files and nothing ever completes). pkill excludes itself; the script
# path does not contain the pattern.
pkill -f "bc_to_shard_block.py" 2>/dev/null || true
  pkill -f "bc_to_shard_fast.py" 2>/dev/null || true
sleep 1
mkdir -p $PDIR

P1_TODO=()
for ((p=0; p<NPART; p++)); do
  grep -q "\[$PDIR/part$p.shard\] wrote" "$LOG" 2>/dev/null || P1_TODO+=($p)
done
if [ ${#P1_TODO[@]} -gt 0 ]; then
  log "P1 resuming: ${P1_TODO[*]}"
  for p in "${P1_TODO[@]}"; do
    S=$((p * PER)); E=$((S + PER))
    $PY $TR/bc_to_shard_block.py --bag $BAG --out $PDIR/part$p.shard --start $S --stop $E >> $LOG 2>&1 &
  done
  wait
  log "P1 tick complete"
  exit 0
fi

P2_TODO=()
for ((p=0; p<NPART; p++)); do
  [ -f "$PDIR/part$p.shard.mask.ready" ] || P2_TODO+=($p)
done
if [ ${#P2_TODO[@]} -gt 0 ]; then
  log "P2 resuming: ${P2_TODO[*]}"
  for p in "${P2_TODO[@]}"; do
    nice -n 10 /home/wyatt/tools/chess/lo-data masks --in $PDIR/part$p.shard --out $PDIR/part$p.shard.mask >> $LOG 2>&1 \
      && touch $PDIR/part$p.shard.mask.ready
  done
  log "P2 tick complete"
  exit 0
fi

P3_TODO=()
for ((p=0; p<NPART; p++)); do
  [ -f "$PDIR/part$p.resolved" ] || P3_TODO+=($p)
done
if [ ${#P3_TODO[@]} -gt 0 ]; then
  log "P3 resuming: ${P3_TODO[*]}"
  for p in "${P3_TODO[@]}"; do
    nice -n 10 $PY $TR/resolve_targets.py --shard $PDIR/part$p.shard --mask $PDIR/part$p.shard.mask >> $LOG 2>&1 \
      && touch $PDIR/part$p.resolved
  done
  log "P3 tick complete"
  exit 0
fi

log "P4 merging"
$PY - << 'PYEOF' >> $LOG 2>&1
import struct, sys, os
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from format import REC_DTYPE, MaskSidecar
D = "/home/wyatt/data"
PDIR = D + "/shards/bc_parts"
NPART = 10
RECSZ = REC_DTYPE.itemsize
final = open(D + "/shards/chessbench_bc_full.shard.part", "wb")
final.write(struct.pack("<IHHQ", 0x48534F4C, 1, 16, 0))
pay = open(D + "/shards/chessbench_bc_full.shard.mask.part", "wb")
offs = open(D + "/shards/chessbench_bc_full.mask.offsets.tmp", "wb")
total = 0; kept = 0; pos = 0
for p in range(NPART):
    shard = PDIR + "/part" + str(p) + ".shard"
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
            pay.write(struct.pack("<B", len(flat))); pay.write(ent)
        else:
            final.write(bytes(rec))
            pay.write(struct.pack("<B", 0))
            pos += 1
        offs.write(struct.pack("<Q", pos))
        if rec[REC_DTYPE.fields["n_targets"][1]] == 1:
            pos += 2 * len(flat)
        total += 1
    offs.write(struct.pack("<Q", pos))
    rf.close(); sc.close()
    os.remove(shard); os.remove(shard + ".mask")
    print("merged part " + str(p) + ": total=" + str(total) + " kept=" + str(kept), flush=True)
final.seek(8); final.write(struct.pack("<Q", total)); final.close()
offs.close(); pay.close()
with open(D + "/shards/chessbench_bc_full.shard.mask.part", "ab") as g:
    with open(D + "/shards/chessbench_bc_full.mask.offsets.tmp", "rb") as h:
        while True:
            chunk = h.read(1 << 24)
            if not chunk: break
            g.write(chunk)
os.remove(D + "/shards/chessbench_bc_full.mask.offsets.tmp")
with open(D + "/shards/chessbench_bc_full.shard.mask.part", "r+b") as g:
    g.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, total))
print("MERGE COMPLETE total=" + str(total) + " kept=" + str(kept), flush=True)
PYEOF
mv $D/shards/chessbench_bc_full.shard.part $FINAL
mv $D/shards/chessbench_bc_full.shard.mask.part $FINAL.mask
rmdir $PDIR 2>/dev/null
log "=== BC fast conversion complete: $FINAL ==="
