#!/bin/bash
# Absolute Elo anchor for the new champion (bc40m) vs SF limit-strength.
set -u
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
D=/home/wyatt/data
NEW=$D/runs/bc_avlabels40m/net_e0_ema.bin
for ELO in 2000 2200 2400; do
  $T/fastchess \
    -engine cmd=$ENG name=bc40m option.WeightsFile=$NEW \
    -engine cmd=/home/wyatt/tools/chess/stockfish name=sf$ELO option.UCI_LimitStrength=true option.UCI_Elo=$ELO \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/bc40m-vs-sf$ELO.pgn > $D/sprt/bc40m-vs-sf$ELO.txt 2>&1
  echo "[$(date '+%H:%M')] sf$ELO done" >> $D/sprt/sf_calib_bc40m.log
done
echo "[$(date '+%H:%M')] all done" >> $D/sprt/sf_calib_bc40m.log
