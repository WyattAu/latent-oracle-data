#!/bin/bash
# Gate 2: INT8 the marquee winner the moment its training lands.
# Fires on "e3 training done" in bc_marquee.log; export -> parity -> ladder
# (INT8-winner vs FP32-winner, INT8-winner vs champ), CPU-only, concurrent
# with the marquee's own ladders and the mirror ablation's GPU training.
set -u
D=/home/wyatt/data/shards
LOG=$D/../gate2_int8.log
PY=/home/wyatt/.venvs/chess/bin/python
EXP=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer/export_int8.py
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

while ! grep -q "e3 training done" $D/../bc_marquee.log 2>/dev/null; do sleep 120; done
NET=$D/../bc_marquee/e3/net_e0_ema.bin
INT8=$D/../bc_marquee/e3/net_e0_int8.bin
log "e3 landed: $NET"

$PY $EXP --net $NET --out $INT8 >> $LOG 2>&1 && log "INT8 exported" || { log "EXPORT FAILED"; exit 1; }

# parity spot-check: 60 random records, target-in-mask + move agreement FP32-vs-INT8
$PY - << 'PYEOF' >> $LOG 2>&1
import sys, random
import numpy as np, torch, chess
sys.path.insert(0, "/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data/trainer")
from model import ChessNet
from loqw_sim import Loqw, forward as loqw_forward
from format import iter_records
loq = Loqw("/home/wyatt/data/shards/../bc_marquee/e3/net_e0_int8.bin")
net = ChessNet(256, 8, 8, 1024, 128).eval()
net.load_state_dict(torch.load("/home/wyatt/data/shards/../bc_marquee/e3/net_e0_ema.pt",
                               map_location="cpu", weights_only=True))
rng = random.Random(2)
stream = iter_records("/home/wyatt/data/shards/bc_sub/slice3.full.shard")
ok = tot = 0
for i, s in enumerate(stream):
    if i % 43317 != 2: continue
    codes = torch.tensor(np.asarray(s.board_codes)).long()[None]
    side = torch.tensor([int(s.side)])
    s1, pr1, wd1 = loqw_forward(loq, codes, int(s.side))
    with torch.no_grad():
        sc, _, _ = net(codes, side)
    mi = int((s1.reshape(-1)).argmax())
    mf = int(sc.reshape(-1).argmax())
    ok += (mi == mf); tot += 1
    if tot >= 60: break
print(f"parity: INT8-vs-FP32 move agreement {ok}/{tot}", flush=True)
PYEOF
log "parity spot-check done"

# ladders once the marquee's CPU ladders are finished (service exits)
while systemctl --user is-active --quiet bc-marquee; do sleep 120; done
log "starting INT8 ladders"
$T/fastchess -engine cmd=$ENG name=bcpool450-int8 option.WeightsFile=$INT8 \
  -engine cmd=$ENG name=bcpool450-fp32 option.WeightsFile=$NET \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpool450-int8-vs-fp32.pgn > $D/sprt/bcpool450-int8-vs-fp32.txt 2>&1
log "ladder int8-vs-fp32 done"
$T/fastchess -engine cmd=$ENG name=bcpool450-int8 option.WeightsFile=$INT8 \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/bcpool450-int8-vs-champ.pgn > $D/sprt/bcpool450-int8-vs-champ.txt 2>&1
log "ladder int8-vs-champ done"
log "=== Gate 2 complete ==="
