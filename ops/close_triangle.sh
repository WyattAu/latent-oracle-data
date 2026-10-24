#!/bin/bash
set -u
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
D=/home/wyatt/data
$T/fastchess \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -engine cmd=$ENG name=bc23m-fp32 option.WeightsFile=$D/runs/bc/net_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$T/openings.epd format=epd order=random \
  -games 100 -rounds 50 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/champ-vs-bc23m-audit.pgn > $D/sprt/champ-vs-bc23m-audit.txt 2>&1
echo "[$(date '+%H:%M')] champ-vs-bc23m done" >> $D/sprt/audit.log
