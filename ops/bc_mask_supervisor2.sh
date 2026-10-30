#!/bin/bash
set -u
D=/home/wyatt/data/shards
PDIR=$D/bcm2_parts
BIN=/home/wyatt/tools/chess/lo-data
LOG=$D/bc_mask2_sup.log
TOTAL=526359028
NPART=6
PER=$(( (TOTAL + NPART - 1) / NPART ))
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
[ -f "$D/chessbench_bc_full.shard.mask" ] && { log "final exists; done"; exit 0; }
mkdir -p $PDIR
pkill -f "bc_to_shard_block" 2>/dev/null || true

# Phase 1: slices + masks
for ((p=0; p<NPART; p++)); do
  [ -f "$PDIR/part$p.mask.ready" ] && continue
  S=$((p * PER)); N=$(( PER < TOTAL - S ? PER : TOTAL - S ))
  if [ ! -f "$PDIR/part$p.full.shard" ] || [ $(stat -c%s "$PDIR/part$p.full.shard") -ne $((16 + N*64)) ]; then
    log "extracting slice $p [$S, +$N)"
    dd if=$D/chessbench_bc_full.shard of=$PDIR/part$p.raw bs=4M iflag=skip_bytes,count_bytes skip=$((16 + S*64)) count=$((N*64)) status=none
    python3 -c "
import struct
open('$PDIR/part$p.hdr','wb').write(struct.pack('<IHHQ', 0x48534F4C, 1, 16, $N))"
    cat $PDIR/part$p.hdr $PDIR/part$p.raw > $PDIR/part$p.full.shard && rm -f $PDIR/part$p.hdr $PDIR/part$p.raw
  fi
  log "masking slice $p"
  $BIN masks --in $PDIR/part$p.full.shard --out $PDIR/part$p.full.shard.mask >> $LOG 2>&1 \
    && touch $PDIR/part$p.mask.ready || { log "slice $p FAILED"; exit 1; }
done
log "all slices masked"

# Phase 2: merge (append-as-you-go; delete parts after use to bound disk)
python3 - << 'PYEOF' >> $LOG 2>&1
import struct, os
import numpy as np
D = "/home/wyatt/data/shards"
PDIR = D + "/bcm2_parts"
NPART = 6
TOTAL = 526359028
PER = (TOTAL + NPART - 1) // NPART
RECSZ = 64
final = D + "/chessbench_bc_full.shard.mask"
tmp = final + ".building"
with open(tmp, "wb") as f:
    f.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
pos = 0
CH = 64 << 20
for p in range(NPART):
    S = p * PER
    n = min(PER, TOTAL - S)
    mp = f"{PDIR}/part{p}.full.shard.mask"
    sc = np.memmap(mp, dtype="<u8", mode="r", offset=16, shape=(n + 1,))
    a = np.asarray(sc, dtype=np.uint64)
    base = int(a[0])
    done = 0
    while done < n:
        k = min(8_000_000, n - done)
        seg = (a[done:done + k + 1] - base + pos).astype("<u8")
        f.write(seg.tobytes())
        done += k
    mf = open(mp, "rb")
    mf.seek(16 + (n + 1) * 8)
    left = int(a[n]) - base
    while left > 0:
        blk = mf.read(min(CH, left))
        if not blk: raise SystemExit(f"part {p} payload exhausted")
        f.write(blk); left -= len(blk)
    mf.close()
    pos += int(a[n]) - base
    del sc
    os.remove(mp); os.remove(f"{PDIR}/part{p}.full.shard")
    print(f"part {p} merged pos={pos}", flush=True)
# fix the final offset entry: the last written table entry must equal payload end
f = open(tmp, "r+b")
f.seek(16 + TOTAL * 8)
f.write(struct.pack("<Q", pos))
f.close()
os.replace(tmp, final)
print("MERGE COMPLETE", flush=True)
PYEOF
log "=== done ==="
