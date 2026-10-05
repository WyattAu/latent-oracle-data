#!/bin/bash
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
NET=/home/wyatt/data/runs/lc0_sp/net_e4.bin
DATA=/home/wyatt/data

$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=lc0sp-v0 option.WeightsFile="$NET" \
  -engine cmd="$SF" name=sf16-p1 option.Threads=1 option.Hash=16 \
  -each proto=uci tc=60+6 plies=1 \
  -openings file="$TOOLS/openings.epd" format=epd order=random \
  -games 200 -rounds 100 -repeat -concurrency 5 \
  -pgnout file="$DATA/sprt/lc0sp-v0-p1.pgn" \
  > "$DATA/sprt/lc0sp-v0-p1.txt" 2>&1
