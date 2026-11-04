#!/bin/bash
# Start compound38m as soon as the soft-ab arms are done training (its
# 4.2GiB fits alongside the ladder engines, not alongside arm training).
# Trigger: arm HARD's final export. Fallback: if soft-ab crashed or is
# still training after 8h, retry every 5 min until no soft_ab train.py.
set -u
T=/home/wyatt/data/runs/soft_ab/hard/net_e19_ema.bin
LOG=/home/wyatt/data/compound38m_watch.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }
for i in $(seq 1 480); do
  [ -s "$T" ] && break
  sleep 60
done
while ps -eo args | grep -q "[t]rain\.py.*soft_ab"; do
  log "soft_ab train.py still running — waiting 5 min"
  sleep 300
done
log "trigger: arm HARD export present; starting compound38m"
systemd-run --user --unit=compound38m \
  --description="38M BC + distill + mirror" \
  -p Restart=on-failure -p RestartSec=30 \
  /home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/ops/compound38m.sh \
  >> "$LOG" 2>&1
log "compound38m unit: $?"
