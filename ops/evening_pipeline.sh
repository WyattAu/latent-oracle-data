#!/bin/bash
# Evening pipeline: RecyclePasses sweep, then external SF calibration.
# All matches use the NEW champion net_q_v2.bin. Sequential (6 threads).
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
CHAMP=$DATA/runs/bc/net_q_v2.bin
LOG=$DATA/evening_pipeline.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

run_match() {  # $1 tag  $2.. extra fastchess args
  local tag=$1; shift
  local pgn=$DATA/sprt/$tag.pgn txt=$DATA/sprt/$tag.txt
  [ -s "$txt" ] && grep -q "Finished match" "$txt" && { log "$tag already done"; return 0; }
  log "starting $tag"
  $TOOLS/fastchess -recover "$@" \
    -each proto=uci tc=15+0.2 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 200 -rounds 100 -repeat -concurrency 6 \
    -pgnout file=$pgn > "$txt" 2>&1
  log "finished $tag"
}

# RecyclePasses sweep: RP=2,4,8 vs RP=1 (both sides, same champion blob)
for RP in 2 4 8; do
  run_match "rp$RP-vs-rp1" \
    -engine cmd="$ENGINE" name=rp$RP option.WeightsFile=$CHAMP option.RecyclePasses=$RP \
    -engine cmd="$ENGINE" name=rp1  option.WeightsFile=$CHAMP option.RecyclePasses=1
done

# External calibration: SF limit-strength ladder (fastchess txt Elo is from
# the FIRST engine's perspective = our champion; PGN recount authoritative)
for ELO in 2000 2200 2400; do
  run_match "cal-sf$ELO" \
    -engine cmd="$ENGINE" name=champ option.WeightsFile=$CHAMP \
    -engine cmd="$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2" name=sf$ELO \
      option.UCI_LimitStrength=true option.UCI_Elo=$ELO
done

log "=== evening pipeline complete ==="
