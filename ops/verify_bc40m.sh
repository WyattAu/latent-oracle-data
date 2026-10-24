#!/bin/bash
set -u
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
D=/home/wyatt/data
$T/fastchess \
  -engine cmd=$ENG name=bc40m option.WeightsFile=$D/runs/bc_avlabels40m/net_e0_ema.bin \
  -engine cmd=$ENG name=bc23m-fp32 option.WeightsFile=$D/runs/bc/net_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$T/openings.epd format=epd order=random \
  -games 200 -rounds 100 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bc40m-vs-bc23m-fp32.pgn > $D/sprt/bc40m-vs-bc23m-fp32.txt 2>&1
echo "[$(date '+%H:%M')] done" >> $D/sprt/verify_bc40m.log
