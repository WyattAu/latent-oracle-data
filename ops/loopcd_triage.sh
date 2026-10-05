#!/bin/bash
# HPO stage B early: LoopCD/recycling grid on existing BC-v0 weights.
# Fast triage (15+0.2, 200 games, concurrency 3) — ranks configs; winners
# graduate to official 60+6 SPRT later. CPU-only; polite to the 5M labeler.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
NET=$DATA/runs/bc/net_e1.bin
LOG=$DATA/loopcd_triage.log

run_cfg() {  # $1 tag  $2 recycle  $3 alpha
  local txt=$DATA/sprt/loopcd-$1.txt
  [ -s "$txt" ] && grep -q "Finished match" "$txt" && return 0
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=bc0-$1 option.WeightsFile=$NET \
      option.RecyclePasses=$2 option.LoopCDAlpha=$3 \
    -engine cmd="$SF" name=sf16-p1 option.Threads=1 option.Hash=16 \
    -each proto=uci tc=15+0.2 plies=1 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 6 \
    -pgnout file=$DATA/sprt/loopcd-$1.pgn > "$txt" 2>&1
  local elo
  elo=$(grep -oE 'Elo: [-0-9]+' "$txt" | grep -oE '[-0-9]+' | head -1)
  echo "[$(date '+%H:%M')] R=$2 a=$3 -> Elo $elo" >> "$LOG"
}

log "waiting for keep_best (last gate of the chain) to exit..."
while pgrep -f "keep_best.sh" > /dev/null; do sleep 600; done
echo "[$(date '+%m-%d %H:%M')] LoopCD triage start (BC-v0 net, 15+0.2, 200 games, conc 6)" >> "$LOG"
run_cfg base 1 0.0
run_cfg r2a25 2 0.25
run_cfg r2a50 2 0.5
run_cfg r2a100 2 1.0
run_cfg r4a25 4 0.25
run_cfg r4a50 4 0.5
run_cfg r4a100 4 1.0
echo "[$(date '+%m-%d %H:%M')] LoopCD triage complete" >> "$LOG"
