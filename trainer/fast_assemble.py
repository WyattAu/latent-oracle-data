import struct
import numpy as np

CUR = "/home/wyatt/data/shards/chessbench_bc_full.shard.mask"
FIX = "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/asm/fixed_offsets.part"
TAIL = "/home/wyatt/data/shards/bc_tail.full.shard.mask"
NEW = "/home/wyatt/data/shards/chessbench_bc_full.shard.new"
TOTAL = 526_359_028
SPLIT = 430_000_000
NTAIL = TOTAL - SPLIT
PPREFIX_BASE = 8_421_744_480
CH = 64 << 20

def run_span(srcf, src_base, table, lo, n, pos, out_off, out_pay, tag):
    """Copy entry bytes [table[lo], table[lo+n]] and emit n+1 offsets."""
    done = 0
    while done < n:
        k = min(8_000_000, n - done)
        a = np.asarray(table[lo + done:lo + done + k + 1], dtype=np.uint64)
        rel = (a - a[0]).astype(np.uint64)
        abspos = pos + rel
        out_off.write(abspos[:k].astype("<u8").tobytes())   # last = next chunk's first
        span0 = src_base + int(a[0]); span1 = src_base + int(a[k])
        srcf.seek(span0)
        left = span1 - span0
        while left > 0:
            blk = srcf.read(min(CH, left))
            if not blk: raise SystemExit(f"{tag}: src exhausted at {left}")
            out_pay.write(blk)
            left -= len(blk)
        pos += int(a[k] - a[0])
        done += k
    out_off.write(np.uint64(pos).astype("<u8").tobytes())   # final entry
    return pos

new = open(NEW, "wb")
new.write(struct.pack("<IHHQ", 0x48534F4C, 1, 0, TOTAL))
out_off = open("/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/asm/final_offsets.tmp", "wb")
out_pay = open("/home/wyatt/data/shards/asm/final_payload.tmp", "wb")

cur = open(CUR, "rb")
fixed = np.memmap(FIX, dtype="<u8", mode="r", shape=(SPLIT + 1,))
pos = run_span(cur, PPREFIX_BASE, fixed, 0, SPLIT, 0, out_off, out_pay, "cur")
print(f"span cur done pos={pos}", flush=True)

tf = open(TAIL, "rb"); tf.seek(16 + NTAIL * 8)
tail = np.memmap(TAIL, dtype="<u8", mode="r", offset=16, shape=(NTAIL + 1,))
pos = run_span(tf, 16 + NTAIL * 8, tail, 0, NTAIL, pos, out_off, out_pay, "tail")
print(f"span tail done pos={pos}", flush=True)

out_off.close(); out_pay.close(); new.close(); cur.close(); tf.close()
print("ASSEMBLY DONE", pos, flush=True)
