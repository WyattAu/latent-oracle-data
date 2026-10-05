#!/bin/bash
# Wait for BC-v1 training to finish, then train Lc0 model
while pgrep -f "train.py.*bc_v1" > /dev/null; do sleep 300; done
echo "[$(date '+%m-%d %H:%M:%S')] BC-v1 training done — starting Lc0-SP training" >> /home/wyatt/data/lc0_pipeline.log
cd /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
/tmp/opencode/venv/bin/python trainer/train.py \
  --shard /home/wyatt/data/shards/lc0_sp/bc.shard \
  --out /home/wyatt/data/runs/lc0_sp \
  --epochs 5 --batch 512 >> /home/wyatt/data/lc0_pipeline.log 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] Lc0-SP training complete" >> /home/wyatt/data/lc0_pipeline.log
