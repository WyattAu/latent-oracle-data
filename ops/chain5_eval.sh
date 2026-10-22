#!/bin/bash
# Chain 5: cross-arm puzzle diagnostics after chain4's trainings finish.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer
LOG=$D/chain5.log
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while ! grep -q "chain4 complete" $D/chain4.log 2>/dev/null; do sleep 300; done
log "chain4 done; diagnostics start"

if [ -f $D/runs/bc_avlabels40m/net_e0.pt ]; then
  log "puzzles: BC control (40M, one-hot)"
  $PY $TR/score_puzzles.py --net $D/runs/bc_avlabels40m/net_e0.pt \
    --puzzles $D/puzzles_500.jsonl >> $LOG 2>&1
fi
if [ -f $D/runs/bc_v3gab40m/net_e0.pt ]; then
  log "puzzles: GAB-v3"
  $PY $TR/score_puzzles.py --net $D/runs/bc_v3gab40m/net_e0.pt --gab \
    --puzzles $D/puzzles_500.jsonl >> $LOG 2>&1
fi
AVCK=$(ls -t $D/runs/av_shard40m/av_step*.pt 2>/dev/null | head -1)
if [ -n "$AVCK" ]; then
  log "puzzles: AV arm ($AVCK)"
  $PY $TR/eval_chessbench_av.py --ckpt "$AVCK" \
    --puzzles $D/puzzles_500.jsonl --num-puzzles 500 >> $LOG 2>&1
fi
log "=== chain5 complete ==="
