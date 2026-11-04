#!/bin/bash
# Soft SF-policy A/B (RESEARCH-CROSSFIELD: the sharp-vs-learnable experiment)
# Identical records, identical exposure; the ONLY variable is the policy
# target: SF multipv soft distribution (arm soft) vs the single hard SF move
# (arm hard). Verdict: 150-game ladder soft vs hard; placement: vs champ.
set -u
PY=/home/wyatt/.venvs/chess/bin/python
D=/home/wyatt/data
SH=$D/shards/bc_sub
R=$D/runs/soft_ab
LOG=$D/soft_ab.log
ENG=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle/build/release/latent-oracle
T=/home/wyatt/tools/chess
DATA_REPO=/home/wyatt/dev/src/github.com/WyattAu/latent-oracle-data
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$LOG"; }

# 1. wait for compound38m to release the GPU
while systemctl --user is-active --quiet compound38m; do sleep 300; done
log "compound38m finished; GPU free"

# 2. wait for the 5 soft-label workers to finish
for i in $(seq 1 240); do
  done_n=$(grep -l "wrote " $D/soft100k_w?.log 2>/dev/null | wc -l)
  [ "$done_n" -ge 5 ] && break
  sleep 60
done
log "soft-label workers done ($done_n/5)"

# 3. extract the mini shard with remapped indices
mkdir -p $R
$PY $DATA_REPO/trainer/extract_shard.py --shard $SH/slice0.full.shard \
  --jsonl $SH/soft_part0.jsonl,$SH/soft_part1.jsonl,$SH/soft_part2.jsonl,$SH/soft_part3.jsonl,$SH/soft_part4.jsonl \
  --out $SH/soft100k.shard >> $LOG 2>&1 || { log "extract FAILED"; exit 1; }
N=$(grep -c "" $SH/soft100k.shard.jsonl)
log "mini shard: $N labelled records"

# 4. arm soft: same records, soft policy targets
log "arm SOFT: training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
  --shard $SH/soft100k.shard --out $R/soft \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 20 --batch 256 --lr 3e-4 --fp32 \
  --soft-labels $SH/soft100k.shard.jsonl >> $LOG 2>&1 || { log "arm SOFT FAILED"; exit 1; }
log "arm SOFT done"

# 5. arm hard: identical exposure, hard one-hot targets (the control)
log "arm HARD: training"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY $DATA_REPO/trainer/train.py \
  --shard $SH/soft100k.shard --out $R/hard \
  --d 256 --layers 8 --heads 8 --dff 1024 --dpol 128 \
  --epochs 20 --batch 256 --lr 3e-4 --fp32 >> $LOG 2>&1 || { log "arm HARD FAILED"; exit 1; }
log "arm HARD done"

# 6. verdict ladder: soft vs hard
log "ladder: soft vs hard"
$T/fastchess -engine cmd=$ENG name=soft option.WeightsFile=$R/soft/net_e19_ema.bin \
  -engine cmd=$ENG name=hard option.WeightsFile=$R/hard/net_e19_ema.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/soft100k-vs-hard100k.pgn > $D/sprt/soft100k-vs-hard100k.txt 2>&1
log "soft-vs-hard ladder done"

# 7. placement ladder: soft vs champ
log "ladder: soft vs champ"
$T/fastchess -engine cmd=$ENG name=soft100k option.WeightsFile=$R/soft/net_e19_ema.bin \
  -engine cmd=$ENG name=champ option.WeightsFile=$D/runs/bc/net_q_v2.bin \
  -each proto=uci tc=15+0.2 -openings file=$T/openings.epd format=epd order=random \
  -games 150 -rounds 75 -repeat -concurrency 6 \
  -pgnout file=$D/sprt/soft100k-vs-champ.pgn > $D/sprt/soft100k-vs-champ.txt 2>&1
log "soft-vs-champ ladder done"
log "=== soft A/B complete ==="
