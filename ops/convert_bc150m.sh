#!/bin/bash
set -u
PY=/home/wyatt/.venvs/chess/bin/python
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
D=/home/wyatt/data
LOG=$D/convert_bc150m.log
echo "[$(date '+%m-%d %H:%M')] converting 16 bags -> bc150m" >> "$LOG"
nice -n 10 $PY $TR/bag_to_shard.py \
  --bags $D/chessbench/action_value-0000{0,1,2,3,4,5,6,7,8,9}-of-02148_data.bag \
         $D/chessbench/action_value-00{10,11,12,13,14,15}-of-02148_data.bag \
  --out $D/shards/chessbench_bc150m.shard >> "$LOG" 2>&1
echo "[$(date '+%m-%d %H:%M')] conversion done: $(ls -la $D/shards/chessbench_bc150m.shard 2>/dev/null | awk '{print $5}') bytes" >> "$LOG"
