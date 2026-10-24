#!/bin/bash
set -u
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
D=/home/wyatt/data
# 1) netless vs bc23m-fp32 (calibrates the "no net loaded" penalty)
$T/fastchess \
  -engine cmd=$ENG name=netless \
  -engine cmd=$ENG name=bc23m-fp32 option.WeightsFile=$D/runs/bc/net_e1.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$T/openings.epd format=epd order=random \
  -games 100 -rounds 50 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/netless-vs-bc23m.pgn > $D/sprt/netless-vs-bc23m.txt 2>&1
echo "[$(date '+%H:%M')] netless done" >> $D/sprt/audit.log
# 2) re-run bc40m vs champ
$T/fastchess \
  -engine cmd=$ENG name=bc40m option.WeightsFile=$D/runs/bc_avlabels40m/net_e0_ema.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 \
  -openings file=$T/openings.epd format=epd order=random \
  -games 100 -rounds 50 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bc40m-vs-champ-audit.pgn > $D/sprt/bc40m-vs-champ-audit.txt 2>&1
echo "[$(date '+%H:%M')] bc40m-vs-champ rerun done" >> $D/sprt/audit.log
# 3) re-run bc40m vs SF2000
$T/fastchess \
  -engine cmd=$ENG name=bc40m option.WeightsFile=$D/runs/bc_avlabels40m/net_e0_ema.bin \
  -engine cmd=/home/wyatt/tools/chess/stockfish/stockfish-ubuntu-x86-64-avx2 name=sf2000 option.UCI_LimitStrength=true option.UCI_Elo=2000 \
  -each proto=uci tc=15+0.2 \
  -openings file=$T/openings.epd format=epd order=random \
  -games 100 -rounds 50 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bc40m-vs-sf2000-audit.pgn > $D/sprt/bc40m-vs-sf2000-audit.txt 2>&1
echo "[$(date '+%H:%M')] bc40m-vs-sf2000 rerun done" >> $D/sprt/audit.log
