#!/bin/bash
cd /home/wyatt/data/chessbench
F="behavioral_cloning_data.bag"
URL="https://storage.googleapis.com/searchless_chess/data/train/${F}"
for attempt in $(seq 1 100); do
  curl -sS -C - --max-time 28800 -o "$F" "$URL" && break
  echo "[$(date '+%m-%d %H:%M')] attempt $attempt failed" >> fetch_bc.log
  sleep 30
done
echo "[$(date '+%m-%d %H:%M')] BC bag done: $(stat -c%s "$F") bytes" >> fetch_bc.log
