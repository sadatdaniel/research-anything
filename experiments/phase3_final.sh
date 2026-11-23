#!/usr/bin/env bash
# Phase 3 final runs on the untouched test worlds: dream, B0, the report, and the engine end to end.
# Each step logs to results/runs/phase3/; a failing step doesn't stop the later ones.
cd "$(dirname "$0")/.."
WORLDS="test-000 test-002 test-003 test-004 test-005 test-006"
PATHS=$(for w in $WORLDS; do printf 'worlds/test/%s ' "$w"; done)
for w in $WORLDS; do
  echo "dream $w $(date +%T)"
  uv run python experiments/phase3_dream.py "worlds/test/$w" --tag final --fresh --max-minutes 30 > "results/runs/phase3/dream_$w-final.log" 2>&1 || echo "  dream $w failed"
done
echo "b0 $(date +%T)"
uv run python experiments/phase3_b0.py $PATHS > results/runs/phase3/b0_test.log 2>&1 || echo "  b0 failed"
echo "report $(date +%T)"
uv run python experiments/phase3_report.py --tag final $PATHS > results/runs/phase3/report_test.log 2>&1 || echo "  report failed"
echo "engine e2e $(date +%T)"
uv run python experiments/phase3_engine_e2e.py > results/runs/phase3/engine_e2e.log 2>&1 || echo "  engine e2e failed"
echo "done $(date +%T)"
