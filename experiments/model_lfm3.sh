#!/usr/bin/env bash
# LFM3-1.7B through the same checks as the other candidates, once the GPU is free (after model_e2e.sh):
# the bake-off (notes, yes/no, LoRA), then the end-to-end read and dream on val-003 and val-004, on its own
# notes and with Ouro checking them. Log: results/runs/bakeoff/lfm3.log
cd "$(dirname "$0")/.."
LOG=results/runs/bakeoff/lfm3.log
T5='--with transformers==5.24.0 --with peft==0.22.0'
run() { echo "== $* $(date +%T)" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "failed: $*" >> "$LOG"; }
until grep -q "^done " results/runs/bakeoff/e2e.log 2>/dev/null; do sleep 30; done   # one model in the GPU at a time
run uv run $T5 python experiments/model_bakeoff.py --model lfm3-1.7b-base
run uv run $T5 python experiments/phase3_read_worlds.py worlds/val/val-003 worlds/val/val-004 --model lfm3-1.7b-base
for w in worlds/val/val-003 worlds/val/val-004; do
  run uv run $T5 python experiments/phase3_dream.py "$w" --model lfm3-1.7b-base --tag e2e-lfm3-1.7b-base --fresh --max-minutes 30
  run uv run python experiments/phase3_dream.py "$w" --model ouro-1.4b --notes-from lfm3-1.7b-base \
      --tag e2e-lfm3-1.7b-base-read-ouro-checked --fresh --max-minutes 30
done
echo "done $(date +%T)" >> "$LOG"
