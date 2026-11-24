#!/usr/bin/env bash
# README screenshots: headless Chrome against the app's deep links, with a throwaway profile (never your own).
#   1. start the app on a demo library:  uv run python -m app.server --library results/runs/screens/library --port 8767
#   2. experiments/screenshots.sh [http://127.0.0.1:8767]
# Writes docs/images/*.png at 1280x800.
cd "$(dirname "$0")/.."
BASE=${1:-http://127.0.0.1:8767}
OUT=docs/images
PROFILE=$(mktemp -d)
mkdir -p "$OUT"
shot() {   # name, hash
  google-chrome --headless=new --disable-gpu --no-first-run --no-default-browser-check --hide-scrollbars \
    --user-data-dir="$PROFILE" --window-size=1280,800 --force-device-scale-factor=1 --virtual-time-budget=9000 \
    --screenshot="$OUT/$1.png" "$BASE/$2" >/dev/null 2>&1 && echo "wrote $OUT/$1.png" || echo "failed $1"
}
shot library      ""
shot overview     "#memory=kroguld&tab=overview"
shot timeline     "#memory=kroguld&tab=explore&view=timeline"
shot graph        "#memory=kroguld&tab=explore&view=graph"
shot questions    "#memory=kroguld&tab=questions&qmode=bank"
shot ask          "#memory=kroguld&tab=questions&qmode=ask&ask=flood|famine"
rm -r -- "$PROFILE"
