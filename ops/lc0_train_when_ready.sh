#!/bin/bash
# Wait for the shard file to have content, then train
while true; do
  SIZE=$(stat -c%s /home/wyatt/data/shards/lc0_sp/bc.shard 2>/dev/null || echo 0)
  if [ "$SIZE" -gt 1000000 ]; then
    echo "[$(date)] shard ready: $SIZE bytes — starting training" >> /home/wyatt/data/lc0_pipeline.log
    break
  fi
  sleep 30
done
/tmp/opencode/venv/bin/python /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/train.py \
  --shard /home/wyatt/data/shards/lc0_sp/bc.shard \
  --out /home/wyatt/data/runs/lc0_sp \
  --epochs 3 --batch 512 >> /home/wyatt/data/lc0_pipeline.log 2>&1
echo "[$(date)] lc0 training done" >> /home/wyatt/data/lc0_pipeline.log
