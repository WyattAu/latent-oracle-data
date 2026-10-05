#!/bin/bash
cd /home/wyatt/data/lichess
for MONTH in 2026-08 2026-09; do
  URL="https://database.lichess.org/standard/lichess_db_standard_rated_${MONTH}.pgn.zst"
  FILE="lichess_db_standard_rated_${MONTH}.pgn.zst"
  if [ -f "$FILE" ]; then
    EXPECTED=$(curl -sI "$URL" | grep -i content-length | grep -oE '[0-9]+')
    ACTUAL=$(stat -c%s "$FILE" 2>/dev/null || echo 0)
    if [ "$EXPECTED" = "$ACTUAL" ]; then
      echo "[$(date)] $FILE already complete ($ACTUAL bytes)" >> fetch.log
      continue
    fi
  fi
  echo "[$(date)] downloading $FILE" >> fetch.log
  for attempt in $(seq 1 100); do
    curl -sS -C - -o "$FILE" "$URL" && break
    echo "attempt $attempt for $FILE failed" >> fetch.log
    sleep 20
  done
  echo "[$(date)] $FILE done: $(stat -c%s "$FILE" 2>/dev/null || echo 0) bytes" >> fetch.log
done
echo "[$(date)] ALL DOWNLOADS COMPLETE" >> fetch.log
