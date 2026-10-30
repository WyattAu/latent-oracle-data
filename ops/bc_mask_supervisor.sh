#!/bin/bash
# Chunked BC mask regeneration supervisor. Idempotent per slice.
set -u
D=/home/wyatt/data/shards
PDIR=$D/bcm_parts
BIN=/home/wyatt/tools/chess/lo-data
LOG=$D/bc_mask_sup.log
TOTAL=526359028
NPART=6
PER=$(( (TOTAL + NPART - 1) / NPART ))
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
[ -f "$D/chessbench_bc_full.shard.mask.final" ] && { log "final mask exists; done"; exit 0; }
mkdir -p $PDIR

# slice shards: dd once (marker = .slice.ready)
for ((p=0; p<NPART; p++)); do
  [ -f "$PDIR/part$p.ready" ] && continue
  S=$((p * PER)); N=$(( PER < TOTAL - S ? PER : TOTAL - S ))
  dd if=$D/chessbench_bc_full.shard of=$PDIR/part$p.shard bs=4M iflag=skip_bytes,count_bytes skip=$((16 + S*64)) count=$((N*64)) status=none
  python3 -c "
import struct
open('$PDIR/part$p.hdr','wb').write(struct.pack('<IHHQ', 0x48534F4C, 1, 16, $N))"
  cat $PDIR/part$p.hdr $PDIR/part$p.shard > $PDIR/part$p.full.shard && rm -f $PDIR/part$p.hdr $PDIR/part$p.shard
  touch $PDIR/part$p.ready
  log "slice $p ready ($N records)"
done

# masks: one per slice (marker = .mask.ready)
for ((p=0; p<NPART; p++)); do
  [ -f "$PDIR/part$p.mask.ready" ] && continue
  log "masking slice $p"
  $BIN masks --in $PDIR/part$p.full.shard --out $PDIR/part$p.full.shard.mask >> $LOG 2>&1 \
    && touch $PDIR/part$p.mask.ready || { log "slice $p mask FAILED"; exit 1; }
done
log "all slices masked"

# assemble: header + rebased offsets + concatenated payloads
 python3 - << 'PYEOF' >> $LOG 2>&1
import struct
import numpy as np
D = "/home/wyatt/data/shards"
PDIR = D + "/bcm_parts"
NPART = 6
TOTAL = 526359028
PER = (TOTAL + NPART - 1) // NPART
new = open(D + "/chessbench_bc_full.shard.mask.final", "wb")
new.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
offs_out = open("/home/wyatt/data/shards/bcm_parts/final_offsets.tmp", "wb")
pay_out = open("/home/wyatt/data/shards/bcm_parts/final_payload.tmp", "wb")
pos = 0
CH = 64 << 20
for p in range(NPART):
    S = p * PER
    sc = np.memmap(f"{PDIR}/part{p}.full.shard.mask", dtype="<u8", mode="r", offset=16,
                   shape=(min(PER, TOTAL - S) + 1,))
    a = np.asarray(sc, dtype=np.uint64)
    base = int(a[0])
    n = len(a) - 1
    # offsets: rebase to global payload coordinates, vectorized chunks
    done = 0
    while done < n:
        k = min(8_000_000, n - done)
        seg = a[done:done + k + 1] - base + pos
        offs_out.write(np.asarray(seg, dtype="<u8").tobytes())
        done += k
    # payload: bytes [base, base+end] of this part's mask
    mf = open(f"{PDIR}/part{p}.full.shard.mask", "rb")
    mf.seek(16 + (n + 1) * 8)
    left = int(a[-1]) - base
    while left > 0:
        blk = mf.read(min(CH, left))
        if not blk: raise SystemExit("payload exhausted")
        pay_out.write(blk)
        left -= len(blk)
    mf.close()
    pos += int(a[-1]) - base
    del sc
    print(f"part {p} merged, pos={pos}", flush=True)
offs_out.close(); pay_out.close(); new.close()
# final = header + offsets + payload
with open(D + "/chessbench_bc_full.shard.mask.final", "r+b") as f:
    with open("/home/wyatt/data/shards/bcm_parts/final_offsets.tmp", "rb") as h:
        while True:
            chunk = h.read(1 << 24)
            if not chunk: break
            f.write(chunk)
    with open("/home/wyatt/data/shards/bcm_parts/final_payload.tmp", "rb") as h:
        while True:
            chunk = h.read(1 << 24)
            if not chunk: break
            f.write(chunk)
print("MASK ASSEMBLY COMPLETE", flush=True)
PYEOF
[ -s "$D/chessbench_bc_full.shard.mask.final" ] && log "final mask written"
rm -rf $PDIR
log "=== BC mask supervisor complete ==="
