#!/bin/bash
# Keep-best gate: after phase_post finishes, SPRT the intermediate epoch
# exports (net_e0, net_e1) against the pool. The chain already SPRTs net_e2.
# Verdict is logged; the winning blob is copied to runs/<name>/net_best.bin.
# Waits for BOTH the phase_post chain and the 5M labeling to free the CPUs.
set -u
DATA=/home/wyatt/data
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
SF=$TOOLS/stockfish/stockfish-ubuntu-x86-64-avx2
LOG=$DATA/keep_best.log

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- memory preflight (2026-10-05: OOM kills took out the d10 labeler at
# 3.9M/4M records and the DiffuSearch sample build; another session on this
# box routinely eats 25+ GB). Wait for real headroom instead of dying.
ram_avail_mb() { awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; }
ram_total_mb() { awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo; }
wait_for_ram() {
  local need=${1:-3000} waited=0
  while [ "$(ram_avail_mb)" -lt "$need" ]; do
    if [ $((waited % 1800)) -eq 0 ]; then
      log "ram wait: need ${need}MB, have $(ram_avail_mb)MB of $(ram_total_mb)MB"
    fi
    sleep 300; waited=$((waited+300))
  done
  log "ram ok: $(ram_avail_mb)MB available (needed ${need}MB)"
}
sprt_one() {  # $1 name  $2 blob path  $3 plies -> echoes "W L D" or fails
  # Ranking gate, not a strength measurement: all epochs are compared at
  # identical settings, so 100 games at 15+1.5 ranks them as well as 200 at
  # 60+6 while costing ~1/8 the CPU. Absolute strength is measured by the
  # phase_post SPRTs at proper time controls.
  local txt=$DATA/sprt/kb-$1-p$3.txt
  $TOOLS/fastchess \
    -engine cmd="$ENGINE" name=kb-$1 option.WeightsFile=$2 \
    -engine cmd="$SF" name=sf16-p$3 option.Threads=1 option.Hash=16 \
    -each proto=uci tc=15+1.5 plies=$3 \
    -openings file=$TOOLS/openings.epd format=epd order=random \
    -games 100 -rounds 50 -repeat -concurrency 4 \
    -pgnout file=$DATA/sprt/kb-$1-p$3.pgn > "$txt" 2>&1
  grep -q "Finished match" "$txt" || return 1
  grep -E "Elo:|Games:" "$txt" | head -2
}

log "=== keep-best gate armed: waiting for phase_post + labeling to finish ==="
while pgrep -f "phase_post.sh" > /dev/null || pgrep -f "lo-data label" > /dev/null \
   || pgrep -f "dw_retrain.sh" > /dev/null; do sleep 600; done

gate() {  # $1 run dir name (e.g. bc_v1)
  local dir=$DATA/runs/$1
  local best_blob="" best_txt=""
  for e in 0 1 2; do
    local blob=$dir/net_e$e.bin
    [ -f "$blob" ] || continue
    local res
    if res=$(sprt_one "$1-e$e" "$blob" 1); then
      log "[$1 net_e$e] $res"
      # score extraction: prefer higher Elo line; fallback = last Games line order
      # fastchess prints RUNNING SPRT estimates mid-match; the verdict is the
      # LAST Elo line (head -1 picked a 14-game snapshot -- same bug class as
      # trainer/analyze_verdicts.py).
      # fastchess prints "Elo: X, nElo: Y" on ONE line, so `grep -oE 'Elo:'`
      # matches the nElo value too -- and `tail -1` then picks exactly the
      # wrong number. Strip the nElo clause before extracting, and keep the
      # FINAL verdict line (fastchess also prints running snapshots).
      local elo best_elo
      elo=$(echo "$res" | sed 's/, nElo:.*//' | grep -oE 'Elo: [-0-9.]+' \
            | tail -1 | grep -oE '[-0-9.]+')
      best_elo=$(echo "$best_txt" | sed 's/, nElo:.*//' | grep -oE 'Elo: [-0-9.]+' \
            | tail -1 | grep -oE '[-0-9.]+')
      if [ -z "$best_txt" ] || [ "${elo:--9999}" -gt "${best_elo:--9999}" ]; then
        best_txt="$res"; best_blob=$blob
      fi
    else
      log "[$1 net_e$e] SPRT FAILED"
    fi
  done
  if [ -n "$best_blob" ]; then
    cp "$best_blob" "$dir/net_best.bin"
    # the AV phase warm-starts from the gate winner's .pt; without this it
    # silently fell back to net_e2.pt and ignored the ranking
    best_pt=${best_blob%.bin}.pt
    if [ -f "$best_pt" ]; then
      cp "$best_pt" "$dir/net_best.pt"
      log "[$1] checkpoint -> $(basename "$best_pt")"
    else
      log "[$1] WARNING: no .pt beside $(basename "$best_blob")"
    fi
    log "[$1] BEST = $(basename "$best_blob") — copied to net_best.bin ($best_txt)"
  fi
}

# Gate every run produced by the current chains
for run in bc_v1 bc_v1f dist1m_dw; do
  [ -d "$DATA/runs/$run" ] && gate "$run"
done
log "=== keep-best gate complete ==="
