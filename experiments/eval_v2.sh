#!/usr/bin/env bash
# The deciding model comparison on the test-v2 evaluation set (test-010…039, 122 planted patterns):
# Qwen3-1.7B-Base first (it reads ~5x faster, so its results come first), then Ouro-1.4B at 4 loops.
# Each model reads every world and dreams on its own notes with recipe default@2 (30-minute cap per world,
# which no earlier run reached). Resumable: reading skips records already read, and a world whose dream
# result exists is skipped. Log: results/runs/eval_v2/log.txt. Report: experiments/eval_v2_report.py
cd "$(dirname "$0")/.."
LOG=results/runs/eval_v2/log.txt
mkdir -p results/runs/eval_v2
WORLDS=$(for i in $(seq 10 39); do printf 'worlds/test/test-%03d ' "$i"; done)
run() { echo "== $* $(date '+%F %T')" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "failed: $* $(date '+%F %T')" >> "$LOG"; }

for model in qwen3-1.7b-base ouro-1.4b; do
  run uv run python experiments/phase3_read_worlds.py $WORLDS --model "$model"
  for w in $WORLDS; do
    name=$(basename "$w")
    if [ -f "results/phase3/dream_${name}-v2-${model}.json" ]; then continue; fi
    run uv run python experiments/phase3_dream.py "$w" --model "$model" --tag "v2-$model" --recipe default@2 --fresh --max-minutes 30
  done
  echo "finished $model $(date '+%F %T')" >> "$LOG"
done
echo "done $(date '+%F %T')" >> "$LOG"
