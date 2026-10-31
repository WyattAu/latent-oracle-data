#!/bin/bash
# Gate 3: engine-ready measurement of the release candidate.
# Fires after Gate 2's INT8 ladders complete. Runs:
#   1. SF-limit ladder for the release candidate (2000/2200/2400)
#   2. DeepNoc 10k puzzle benchmark on the release candidate
# The release candidate = whichever net won Gate 1 (bcpool450 or champ).
set -u
D=/home/wyatt/data/shards
LOG=$D/../gate3_measure.log
PY=/home/wyatt/.venvs/chess/bin/python
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
SF=$T/stockfish/stockfish-ubuntu-x86-64-avx2
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while systemctl --user is-active --quiet gate2; do sleep 300; done
log "gate2 done; Gate 3 measurement starts"

NET=$D/../bc_marquee/e3/net_e0_int8.bin
[ -f "$NET" ] || NET=$D/runs/bc/net_q_v2.bin
log "release candidate: $NET"

# 1) SF-limit ladder
for ELO in 2000 2200 2400; do
  $T/fastchess -engine cmd=$ENG name=rc option.WeightsFile=$NET \
    -engine cmd=$SF name=sf$ELO option.UCI_LimitStrength=true option.UCI_Elo=$ELO \
    -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
    -games 150 -rounds 75 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/rc-vs-sf$ELO.pgn > $D/sprt/rc-vs-sf$ELO.txt 2>&1
  log "SF-limit-$ELO ladder done"
done

# 2) DeepNoc puzzle benchmark (10k suite, jsonl)
CUDA_VISIBLE_DEVICES="" timeout 3600 $PY trainer/score_puzzles.py \
  --net $D/../bc_marquee/e3/net_e0_ema.pt \
  --puzzles $D/chessbench/puzzles_deepnoc.jsonl --batch 128 >> $LOG 2>&1
log "DeepNoc puzzle benchmark done"

log "=== Gate 3 complete: the public measurement is ready ==="
