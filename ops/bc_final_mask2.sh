#!/bin/bash
set -u
D=/home/wyatt/data/shards
LOG=$D/bc_final_mask2.log
BIN=/home/wyatt/tools/chess/lo-data
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
declare -a STARTS=(0 150000000 300000000 430000000)
declare -a ENDS=(150000000 300000000 430000000 526359028)
FIN=$D/chessbench_bc_full.shard.mask.final2
[ -f "$FIN" ] && [ $(stat -c%s "$FIN") -gt 1000000000 ] && { log "already done"; exit 0; }
mkdir -p $D/bc_sub
for idx in 0 1 2 3; do
  S=${STARTS[$idx]}; E=${ENDS[$idx]}; N=$((E - S))
  A=$D/bc_sub/slice$idx.full.shard
  AM=$A.mask
  if [ ! -f "$AM" ] || [ $(stat -c%s "$AM") -lt 2000000000 ]; then
    if [ ! -f "$A" ] || [ $(stat -c%s "$A") -ne $((16 + N*64)) ]; then
      log "extracting sub-slice $idx [$S,$E)"
      dd if=$D/chessbench_bc_full.shard of=$A.slice bs=4M iflag=skip_bytes,count_bytes skip=$((16 + S*64)) count=$((N*64)) status=none
      python3 -c "
import struct
open('$A.hdr','wb').write(struct.pack('<IHHQ', 0x48534F4C, 1, 16, $N))"
      cat $A.hdr $A.slice > $A && rm -f $A.hdr $A.slice
    fi
    log "masking sub-slice $idx"
    $BIN masks --in $A --out $AM >> $LOG 2>&1 || { log "slice $idx mask FAILED"; exit 1; }
  fi
  log "sub-slice $idx mask ready ($(stat -c%s $AM) bytes)"
done

log "assembling final mask (4 spans)"
python3 - << 'PYEOF' >> $LOG 2>&1
import struct
import numpy as np
D = "/home/wyatt/data/shards"
SPANS = [(0, 150000000), (150000000, 300000000), (300000000, 430000000), (430000000, 526359028)]
TOTAL = 526359028
FIN = D + "/chessbench_bc_full.shard.mask.final2"
new = open(FIN + ".tmp", "wb")
new.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
CH = 64 << 20
pos = 0
for si, (s, e) in enumerate(SPANS):
    n = e - s
    A = f"{D}/bc_sub/slice{si}.full.shard"
    AM = A + ".mask"
    am = np.memmap(AM, dtype="<u8", mode="r", offset=16, shape=(n + 1,))
    af = open(AM.replace(".mask", ""), "rb")
    base = int(am[0])
    done = 0
    while done < n:
        k = min(8_000_000, n - done)
        seg = (np.asarray(am[done:done + k + 1], dtype=np.uint64) - base + pos).astype("<u8")
        new.write(seg.tobytes())
        done += k
    af.seek(16 + (n + 1) * 8)
    left = int(am[n]) - base
    while left > 0:
        blk = af.read(min(CH, left))
        if not blk: raise SystemExit(f"slice {si} payload exhausted")
        new.write(blk); left -= len(blk)
    pos += int(am[n]) - base
    af.close(); del am
    print(f"span {si} merged, pos={pos}", flush=True)
new.close()
os_replace = __import__("os").replace
os_replace(FIN + ".tmp", FIN)
print("FINAL2 COMPLETE", flush=True)
PYEOF
log "=== bc_final_mask2 done ==="
