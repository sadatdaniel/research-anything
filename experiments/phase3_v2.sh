#!/usr/bin/env bash
# Phase 3, second evaluation: recipe default@1 against default@2 on test worlds nothing has looked at
# (test-007, 008, 009). Each step logs to results/runs/phase3/; a failing step doesn't stop the rest.
cd "$(dirname "$0")/.."
WORLDS="test-007 test-008 test-009"
PATHS=$(for w in $WORLDS; do printf 'worlds/test/%s ' "$w"; done)
echo "read $(date +%T)"
uv run python experiments/phase3_read_worlds.py $PATHS > results/runs/phase3/read_v2.log 2>&1 || echo "  read failed"
for recipe in default@2 default@1; do
  tag=${recipe//@/-v}
  for w in $WORLDS; do
    echo "dream $w $recipe $(date +%T)"
    uv run python experiments/phase3_dream.py "worlds/test/$w" --tag "$tag" --recipe "$recipe" --fresh --max-minutes 30 > "results/runs/phase3/dream_$w-$tag.log" 2>&1 || echo "  dream $w $recipe failed"
  done
done
echo "b0 $(date +%T)"
uv run python experiments/phase3_b0.py $PATHS > results/runs/phase3/b0_test_v2.log 2>&1 || echo "  b0 failed"
cp results/phase3/b0_test.json results/phase3/b0_test-v2worlds.json 2>/dev/null
for tag in default-v2 default-v1; do
  echo "report $tag $(date +%T)"
  uv run python experiments/phase3_report.py --tag "$tag" $PATHS > "results/runs/phase3/report_$tag.log" 2>&1 || echo "  report $tag failed"
done
echo "done $(date +%T)"
