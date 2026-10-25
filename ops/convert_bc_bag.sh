#!/bin/bash
# Full behavioral-cloning bag -> LOSH shard. ~330M records expected (36GB bag),
# ~14% skipped (factorized-action ambiguity). Value targets neutral — the bag
# has no results and the engine plays policy argmax.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/convert_bc_bag.log
echo "[$(date '+%m-%d %H:%M')] converting BC bag (36GB) -> chessbench_bc_full.shard" >> "$LOG"
nice -n 10 $PY $TR/bc_to_shard.py \
  --bag $D/chessbench/behavioral_cloning_data.bag \
  --out $D/shards/chessbench_bc_full.shard >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] conversion done: $(stat -c%s $D/shards/chessbench_bc_full.shard 2>/dev/null) bytes" >> "$LOG"
echo "[$(date '+%m-%d %H:%M')] generating mask sidecar" >> "$LOG"
LO_LODATA_BIN=/home/wyatt/tools/chess/lo-data nice -n 10 $PY - << 'PYEOF' >> "$LOG" 2>&1
import subprocess, os
from format import count_records
shard = "/home/wyatt/data/shards/chessbench_bc_full.shard"
mask = shard + ".mask"
total = count_records(shard)
r = subprocess.run(["/home/wyatt/tools/chess/lo-data", "masks", "--in", shard, "--out", mask],
                   capture_output=True, text=True)
ok = os.path.exists(mask) and count_records(mask) >= total
print(f"mask gen rc={r.returncode} ok={ok} {r.stderr[-300:] if r.returncode else ''}", flush=True)
PYEOF
echo "[$(date '+%m-%d %H:%M')] === BC conversion complete ===" >> "$LOG"
