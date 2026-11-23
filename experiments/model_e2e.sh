#!/usr/bin/env bash
# The bake-off end to end: each candidate reads whole validation worlds (val-003, val-004) and dreams on its
# own notes; one pairing dreams with Ouro on a faster model's notes (fast reader, careful checker).
# Scores land in results/phase3/dream_<world>-e2e-<pairing>.json; log: results/runs/bakeoff/e2e.log
#
#   experiments/model_e2e.sh qwen3.5-2b-base qwen3-1.7b-base      # challengers (Ouro-1.4B always runs)
cd "$(dirname "$0")/.."
LOG=results/runs/bakeoff/e2e.log
WORLDS="worlds/val/val-003 worlds/val/val-004"
T5='--with transformers==5.24.0 --with peft==0.22.0'
env_for() { [ "$1" = "qwen3.5-2b-base" ] && echo "$T5" || echo ""; }   # Qwen3.5 needs transformers 5
run() { echo "== $* $(date +%T)" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "failed: $*" >> "$LOG"; }

CHALLENGERS="$*"
for m in ouro-1.4b $CHALLENGERS; do
  run uv run $(env_for "$m") python experiments/phase3_read_worlds.py $WORLDS --model "$m"
done
for w in $WORLDS; do
  for m in ouro-1.4b $CHALLENGERS; do
    run uv run $(env_for "$m") python experiments/phase3_dream.py "$w" --model "$m" --tag "e2e-$m" --fresh --max-minutes 30
  done
  first=${CHALLENGERS%% *}
  [ -n "$first" ] && run uv run python experiments/phase3_dream.py "$w" --model ouro-1.4b --notes-from "$first" \
                         --tag "e2e-$first-read-ouro-checked" --fresh --max-minutes 30
done
echo "done $(date +%T)" >> "$LOG"
