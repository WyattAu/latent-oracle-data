#!/bin/bash
# Fetch ChessBench action_value train shards 4..15 (~1.2GB each, resumable).
cd /home/wyatt/data/chessbench
for N in $(seq 4 15); do
  F=$(printf "action_value-%05d-of-02148_data.bag" "$N")
  URL="https://storage.googleapis.com/searchless_chess/data/train/${F}"
  EXPECTED=$(curl -sI --max-time 30 "$URL" | grep -i content-length | grep -oE '[0-9]+')
  ACTUAL=$(stat -c%s "$F" 2>/dev/null || echo 0)
  if [ -n "$EXPECTED" ] && [ "$EXPECTED" = "$ACTUAL" ]; then
    echo "[$(date '+%m-%d %H:%M')] $F already complete ($ACTUAL bytes)" >> fetch_av.log
    continue
  fi
  echo "[$(date '+%m-%d %H:%M')] downloading $F (expect $EXPECTED)" >> fetch_av.log
  for attempt in $(seq 1 100); do
    curl -sS -C - --max-time 14400 -o "$F" "$URL" && break
    echo "[$(date '+%m-%d %H:%M')] attempt $attempt for $F failed" >> fetch_av.log
    sleep 20
  done
  echo "[$(date '+%m-%d %H:%M')] $F done: $(stat -c%s "$F" 2>/dev/null || echo 0) bytes" >> fetch_av.log
done
echo "[$(date '+%m-%d %H:%M')] ALL FETCHES COMPLETE" >> fetch_av.log
