import struct
import numpy as np

D = "/home/wyatt/data/shards"
CUR = D + "/chessbench_bc_full.shard.mask"          # [hdr 16][P 31,102,633,462]
FIX = D + "/chessbench_bc_full.shard.mask.fixed"    # offsets [0, 430M]
S3  = D + "/bc_sub/slice3.full.shard.mask"          # offsets+payload [430M, total]
NEW = D + "/chessbench_bc_full.shard.mask.final"
TOTAL = 526_359_028
SPLIT = 430_000_000
NTAIL = TOTAL - SPLIT
CH = 64 << 20

fixed = np.memmap(FIX, dtype="<u8", mode="r", offset=16, shape=(SPLIT + 1,))
s3 = np.memmap(S3, dtype="<u8", mode="r", offset=16, shape=(NTAIL + 1,))
pos_split = int(fixed[SPLIT])
s3_base = int(s3[0])

cur = open(CUR, "rb")
cur.seek(16)                                          # P starts here
new = open(NEW, "wb")
new.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
# offsets table
done = 0
while done < SPLIT + 1:
    k = min(8_000_000, SPLIT + 1 - done)
    new.write(np.asarray(fixed[done:done + k], dtype="<u8").tobytes())
    done += k
done = 1
while done < NTAIL + 1:
    k = min(8_000_000, NTAIL + 1 - done)
    seg = (np.asarray(s3[done:done + k], dtype=np.uint64) - np.uint64(s3_base)
           + np.uint64(pos_split)).astype("<u8")
    new.write(seg.tobytes())
    done += k
# payload: stream from the current file
left = 31_102_633_462
while left > 0:
    blk = cur.read(min(CH, left))
    if not blk: raise SystemExit("cur exhausted")
    new.write(blk); left -= len(blk)
cur.close(); new.close()
import os
print("REBUILT", os.path.getsize(NEW), "expected", 16 + 4_210_872_232 + 31_102_633_462, flush=True)
