"""Does the confidence score mean what it says? On fake worlds we know which theories are true, so we
can check: of the theories it scored 80-100, how many match a planted pattern? (A calibrated score:
about 90% of them.) Every theory counts, whatever its stage, except those covered by a bigger one.

  uv run python experiments/phase3_calibration.py --tag final      # the dream_<world>-<tag> brains

Writes results/phase3/calibration-<tag>.json: per score band, how many theories and how many were
true; the Brier score (mean squared gap between score/100 and truth: 0 is perfect, 0.25 is a coin
flip); and the same for the confirm/reject decision alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronicles.scorer import score_world  # noqa: E402
from learner import theories as th  # noqa: E402

BANDS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    rows = []
    for db in sorted(glob.glob(str(ROOT / "results/runs/phase3/dream" / f"*-{args.tag}.sqlite"))):
        world = Path(db).name[: -len(f"-{args.tag}.sqlite")]
        split = world.split("-")[0]
        truth = json.loads((ROOT / "worlds" / split / world / "truth.json").read_text())
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        for t in th.all_theories(conn):
            if t["stage"] == "superseded":
                continue
            s = score_world(truth, [th.as_claim(t)])
            rows.append({"world": world, "id": t["id"], "type": t["type"], "stage": t["stage"],
                         "confidence": t["confidence"], "true": bool(s["rows"][0]["matched"])})
    bands = []
    for lo, hi in BANDS:
        b = [r for r in rows if lo <= r["confidence"] < hi]
        bands.append({"band": f"{lo}-{min(hi, 100)}", "theories": len(b), "true": sum(r["true"] for r in b),
                      "share_true": round(sum(r["true"] for r in b) / len(b), 2) if b else None,
                      "mean_score": round(sum(r["confidence"] for r in b) / len(b) / 100, 2) if b else None})
    brier = sum((r["confidence"] / 100 - r["true"]) ** 2 for r in rows) / max(1, len(rows))
    ece = sum(b["theories"] * abs(b["share_true"] - b["mean_score"]) for b in bands if b["theories"]) / max(1, len(rows))
    conf = [r for r in rows if r["stage"] == "confirmed"]
    out = {"tag": args.tag, "theories": len(rows), "true": sum(r["true"] for r in rows), "bands": bands,
           "brier": round(brier, 3), "ece": round(ece, 3),
           "confirmed": {"n": len(conf), "true": sum(r["true"] for r in conf)},
           "rejected_but_true": [r for r in rows if r["stage"] == "rejected" and r["true"]]}
    path = ROOT / "results" / "phase3" / f"calibration-{args.tag}.json"
    path.write_text(json.dumps(out, indent=2))
    for b in bands:
        print(f"  score {b['band']:>6}: {b['theories']:3d} theories, {b['true']:2d} true"
              + (f" ({b['share_true']:.0%}; scored {b['mean_score']:.0%})" if b["theories"] else ""))
    print(f"  Brier {out['brier']}, ECE {out['ece']}; confirmed {out['confirmed']}; "
          f"rejected but true: {len(out['rejected_but_true'])}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
