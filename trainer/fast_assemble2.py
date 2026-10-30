import struct
import os
import numpy as np

D = "/home/wyatt/data/shards"
SPANS = [(0, 150000000), (150000000, 300000000), (300000000, 430000000), (430000000, 526359028)]
TOTAL = 526359028
NEW = D + "/chessbench_bc_full.shard.mask.final2"
CH = 64 << 20

new = open(NEW, "wb")
new.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
out_off = open(D + "/asm/fa_offsets.tmp", "wb")
out_pay = open(D + "/asm/fa_payload.tmp", "wb")
pos = 0
for si, (s, e) in enumerate(SPANS):
    n = e - s
    A = f"{D}/bc_sub/slice{si}.full.shard"
    AM = A + ".mask"
    am = np.memmap(AM, dtype="<u8", mode="r", offset=16, shape=(n + 1,))
    af = open(AM, "rb")
    base = int(am[0])
    done = 0
    while done < n:
        k = min(8_000_000, n - done)
        seg = (np.asarray(am[done:done + k + 1], dtype=np.uint64) - base + pos).astype("<u8")
        out_off.write(seg.tobytes())
        done += k
    af.seek(16 + (n + 1) * 8)
    left = int(am[n]) - base
    while left > 0:
        blk = af.read(min(CH, left))
        if not blk: raise SystemExit(f"slice {si} payload exhausted")
        out_pay.write(blk); left -= len(blk)
    pos += int(am[n]) - base
    af.close(); del am
    print(f"span {si} done pos={pos}", flush=True)
out_off.close(); out_pay.close(); new.close()
# final = header + offsets + payload
with open(NEW, "wb") as f:
    f.write(struct.pack("<IHHQ", 0x534D4F4C, 1, 0, TOTAL))
with open(D + "/asm/fa_offsets.tmp", "rb") as h:
    while True:
        chunk = h.read(CH)
        if not chunk: break
        f.write(chunk)
with open(D + "/asm/fa_payload.tmp", "rb") as h:
    while True:
        chunk = h.read(CH)
        if not chunk: break
        f.write(chunk)
os.replace(NEW, NEW.replace(".new", ""))
print("FINAL ASSEMBLED", os.path.getsize(NEW.replace(".new", "")), flush=True)
