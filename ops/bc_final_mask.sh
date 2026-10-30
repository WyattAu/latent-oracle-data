#!/bin/bash
set -u
D=/home/wyatt/data/shards
LOG=$D/bc_final_mask.log
BIN=/home/wyatt/tools/chess/lo-data
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
SPLIT=430000000
NTAIL=96359028
A=$D/bc_sliceA.full.shard
AM=$A.mask
T=$D/bc_tail.full.shard
TM=$T.mask
FIN=$D/chessbench_bc_full.shard.mask.final2

if [ ! -f "$FIN" ] || [ $(stat -c%s "$FIN") -lt 1000000000 ]; then
  if [ ! -f "$A" ]; then
    log "extracting slice A"
    dd if=$D/chessbench_bc_full.shard of=$A.slice bs=4M iflag=skip_bytes,count_bytes skip=16 count=$((SPLIT*64)) status=none
    python3 -c "
import struct
open('$A.hdr','wb').write(struct.pack('<IHHQ', 0x48534F4C, 1, 16, $SPLIT))"
    cat $A.hdr $A.slice > $A && rm -f $A.hdr $A.slice
  fi
  if [ ! -f "$AM" ] || [ $(stat -c%s "$AM") -lt 1000000000 ]; then
    log "masking slice A"
    $BIN masks --in $A --out $AM >> $LOG 2>&1
  fi
  log "assembling final mask"
  python3 - << 'PYEOF' >> $LOG 2>&1
import struct
import numpy as np
D = "/home/wyatt/data/shards"
SPLIT = 430000000
NTAIL = 96359028
A = D + "/bc_sliceA.full.shard"
AM = A + ".mask"
T = D + "/bc_tail.full.shard"
TM = T + ".mask"
FIN = D + "/chessbench_bc_full.shard.mask.final2"
TOTAL = SPLIT + NTAIL
am = np.memmap(AM, dtype="<u8", mode="r", offset=16, shape=(SPLIT + 1,))
tm = np.memmap(TM, dtype="<u8", mode="r", offset=16, shape=(NTAIL + 1,))
af = open(AM, "rb"); af.seek(16 + (SPLIT + 1) * 8)
tf = open(TM, "rb"); tf.seek(16 + (NTAIL + 1) * 8)
new = open(FIN, "wb")
new.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
CH = 64 << 20
# offsets: slice A table then tail table rebased
a_last = int(am[SPLIT])
done = 0
while done < SPLIT:
    k = min(8_000_000, SPLIT - done)
    seg = np.asarray(am[done:done + k], dtype="<u8")
    new.write(seg.tobytes())
    done += k
new.write(np.uint64(a_last).astype("<u8").tobytes())
base = int(tm[0]); off = a_last
done = 0
while done < NTAIL:
    k = min(8_000_000, NTAIL - done)
    seg = (np.asarray(tm[done:done + k], dtype="<u8") - base + off + 1).astype("<u8")
    new.write(seg.tobytes())
    done += k
new.write(np.uint64(off + 1 + int(tm[NTAIL])).astype("<u8").tobytes())
# payloads
af.seek(16 + (SPLIT + 1) * 8)
left = int(am[SPLIT]) - int(am[0])
while left > 0:
    blk = af.read(min(CH, left))
    if not blk: raise SystemExit("A payload exhausted")
    new.write(blk); left -= len(blk)
tf.seek(16 + (NTAIL + 1) * 8)
left = int(tm[NTAIL]) - int(tm[0])
while left > 0:
    blk = tf.read(min(CH, left))
    if not blk: raise SystemExit("tail payload exhausted")
    new.write(blk); left -= len(blk)
new.close(); af.close(); tf.close()
print("FINAL2 ASSEMBLED", flush=True)
PYEOF
  log "final2 size: $(stat -c%s $FIN 2>/dev/null)"
fi
log "=== done ==="
