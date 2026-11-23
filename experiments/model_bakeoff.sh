#!/usr/bin/env bash
# The model bake-off: every candidate through experiments/model_bakeoff.py, one at a time (one model in the
# GPU at a time). Optionally waits for each model's download to finish first.
#
#   experiments/model_bakeoff.sh [download-log]      # log: results/runs/bakeoff/log.txt
cd "$(dirname "$0")/.."
LOG=results/runs/bakeoff/log.txt
mkdir -p results/runs/bakeoff
DL=${1:-}
wait_for() {   # until the download log says this folder is done; false if it failed
  [ -z "$DL" ] && return 0
  until grep -qE "^(ok|FAILED) $1( |$)" "$DL"; do sleep 20; done
  grep -qE "^ok $1( |$)" "$DL"
}
run() { echo "== $* $(date +%T)" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "failed: $*" >> "$LOG"; }

run uv run python experiments/model_bakeoff.py --model ouro-1.4b --loops 4
run uv run python experiments/model_bakeoff.py --model ouro-1.4b --loops 2 --tasks notes,yesno
for pair in "ouro-2.6b Ouro-2.6B" "minicpm5-1b MiniCPM5-1B" "qwen3-1.7b-base Qwen3-1.7B-Base" "minicpm5-2b MiniCPM5-2B"; do
  set -- $pair
  if wait_for "$2"; then run uv run python experiments/model_bakeoff.py --model "$1"; else echo "skipped $1: download failed" >> "$LOG"; fi
done
# Qwen3.5 needs transformers 5; an overlay environment keeps the project's pinned version for Ouro.
if wait_for Qwen3.5-2B-Base; then
  run uv run --with "transformers==5.24.0" python experiments/model_bakeoff.py --model qwen3.5-2b-base
else echo "skipped qwen3.5-2b-base: download failed" >> "$LOG"; fi
echo "done $(date +%T)" >> "$LOG"
