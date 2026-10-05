#!/bin/bash
# When the running labeler exits, swap in the new lo-data binary (progress
# logging + everything current) so phase_c2's 5M stage uses it.
while pgrep -f "lo-data label" > /dev/null; do sleep 60; done
if [ -f /home/wyatt/tools/chess/lo-data.next ]; then
  mv /home/wyatt/tools/chess/lo-data.next /home/wyatt/tools/chess/lo-data
  echo "[$(date '+%m-%d %H:%M:%S')] swapped durable lo-data binary" >> /home/wyatt/data/pipeline.log
fi
