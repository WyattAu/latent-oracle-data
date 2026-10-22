#!/bin/bash
# INT8 replication: does the +233 survive fresh calibration?
# A: bc/net_e1.bin (FP32) vs /tmp/repro_shard.bin (fresh calibrated INT8 of the same net)
# B: champion net_q.bin (old calibration) vs /tmp/repro_shard.bin (fresh calibration)
set -u
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
D=/home/wyatt/data
cp /tmp/repro_shard.bin $D/runs/bc/net_e1_qfresh.bin
run() { # tag, nameA, binA, nameB, binB
  $T/fastchess \
    -engine cmd=$ENG name=$2 option.WeightsFile=$3 \
    -engine cmd=$ENG name=$4 option.WeightsFile=$5 \
    -each proto=uci tc=15+0.2 \
    -openings file=$T/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 6 \
    -pgnout file=$D/sprt/$1.pgn > $D/sprt/$1.txt 2>&1
  echo "[$(date '+%H:%M')] $1 done" >> $D/sprt/int8_repro.log
}
echo "[$(date '+%H:%M')] start" > $D/sprt/int8_repro.log
run fp32-vs-int8fresh fp32 $D/runs/bc/net_e1.bin int8fresh $D/runs/bc/net_e1_qfresh.bin
run champ-vs-int8fresh champ $D/runs/bc/net_q_v2.bin int8fresh $D/runs/bc/net_e1_qfresh.bin
echo "[$(date '+%H:%M')] all done" >> $D/sprt/int8_repro.log
