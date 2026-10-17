#!/bin/bash
# RL admission gate (Loop 2 ruling, see docs/RESEARCH-LOOP2-RL.md).
#
# An RL run may touch the engine ONLY if its checkpoint beats ITS OWN BC
# teacher over 300 games at tc=15+0.2 (pair-model, engine-name recount).
# GRPO failed this at 15/15 checkpoints; that is the whole reason it is
# refuted instead of tuned.
#
# Usage: rl_gate.sh <candidate.bin> <teacher.bin> [games]
# Exit 0 = PASS (candidate is at least not worse: lower 95% bound > -25 Elo),
#        1 = FAIL or match error.
set -u
DATA=/home/wyatt/data
PY=/home/wyatt/.venvs/chess/bin/python
TOOLS=/home/wyatt/tools/chess
ENGINE=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
TR=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer

CAND=${1:?usage: rl_gate.sh <candidate.bin> <teacher.bin> [games]}
TEACH=${2:?usage: rl_gate.sh <candidate.bin> <teacher.bin> [games]}
GAMES=${3:-300}
ROUNDS=$((GAMES / 2))
TAG=$(basename "$(dirname "$CAND")")-vs-$(basename "$(dirname "$TEACH")")
PGN=$DATA/sprt/rlgate-$TAG-$GAMES.pgn
TXT=$DATA/sprt/rlgate-$TAG-$GAMES.txt

[ -f "$CAND" ]  || { echo "FATAL: candidate missing: $CAND"; exit 1; }
[ -f "$TEACH" ] || { echo "FATAL: teacher missing: $TEACH"; exit 1; }
[ -x "$ENGINE" ] || { echo "FATAL: engine missing: $ENGINE"; exit 1; }

# candidate FIRST so fastchess txt Elo reads from the candidate's perspective;
# the engine-name recount below is the authoritative number either way.
$TOOLS/fastchess \
  -engine cmd="$ENGINE" name=rl-candidate option.WeightsFile=$CAND \
  -engine cmd="$ENGINE" name=rl-teacher   option.WeightsFile=$TEACH \
  -each proto=uci tc=15+0.2 \
  -openings file=$TOOLS/openings.epd format=epd order=random \
  -games "$GAMES" -rounds "$ROUNDS" -repeat -concurrency 6 \
  -pgnout file="$PGN" > "$TXT" 2>&1 || { echo "FAIL: fastchess error, see $TXT"; exit 1; }

echo "=== rl_gate: $CAND vs $TEACH, $GAMES games, tc=15+0.2 ==="
grep -E "Elo:|Games:" "$TXT" | tail -2
echo "--- pair-model recount by engine name (authoritative):"
$PY $TR/sprt_pentanomial.py "$PGN"

# PASS requires the 95% lower bound above -25 Elo (a null result is not a pass:
# RL must add information beyond imitation, so "indistinguishable" fails).
$PY - "$PGN" <<'PYEOF'
import re, subprocess, sys, os
pgn = sys.argv[1]
out = subprocess.run(
    ["/home/wyatt/.venvs/chess/bin/python",
     "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/sprt_pentanomial.py",
     pgn], capture_output=True, text=True).stdout
print(out, end="")
m = re.search(r"Elo \(pair model\):\s*([-+0-9.]+)\s*\+-\s*([0-9.]+)", out)
if not m:
    print("FAIL: could not parse pair-model Elo (see above)")
    sys.exit(1)
elo, se = float(m.group(1)), float(m.group(2))
lo = elo - 1.96 * se
print(f"pair-model Elo = {elo:+.1f} +- {se:.1f}; lower 95% bound = {lo:+.1f}")
if lo > -25.0:
    print("RESULT: PASS — RL added information beyond its teacher")
    sys.exit(0)
print("RESULT: FAIL — no demonstrated gain over the BC teacher")
sys.exit(1)
PYEOF
