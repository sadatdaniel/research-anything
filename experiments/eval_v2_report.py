"""Report for the test-v2 comparison: each model's recall and precision, and the paired difference, with
95% intervals from resampling whole worlds (patterns in one world aren't independent).

  uv run python experiments/eval_v2_report.py            # whatever has finished so far

Reads results/phase3/dream_test-0NN-v2-<model>.json; writes results/eval_v2/summary.json.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen3-1.7b-base", "ouro-1.4b")
WORLDS = [f"test-{i:03d}" for i in range(10, 40)]
RESAMPLES = 10_000


def per_world(model: str) -> dict[str, dict]:
    out = {}
    for w in WORLDS:
        p = ROOT / "results" / "phase3" / f"dream_{w}-v2-{model}.json"
        if p.exists():
            d = json.loads(p.read_text())
            c = d["confirmed"]
            out[w] = {"found": c["found"], "planted": c["planted"], "claims": c["claims"],
                      "true_claims": c["claims"] - c["false_claims"] - c["decoy_hits"], "decoys": c["decoy_hits"],
                      "predictions": c.get("predictions") or 0, "predictions_correct": c.get("predictions_correct") or 0,
                      "dream_seconds": d["dream_seconds"], "per_type": c["per_type"]}
    return out


def ratio(rows: list[dict], num: str, den: str) -> float | None:
    d = sum(r[den] for r in rows)
    return sum(r[num] for r in rows) / d if d else None


def interval(worlds: list[str], stat, rng: random.Random) -> list[float] | None:
    vals = []
    for _ in range(RESAMPLES):
        v = stat([rng.choice(worlds) for _ in worlds])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 3), round(vals[int(0.975 * len(vals)) - 1], 3)]


def main() -> int:
    rng = random.Random(0)
    data = {m: per_world(m) for m in MODELS}
    report = {"worlds_done": {m: len(v) for m, v in data.items()}, "models": {}}
    for m, rows in data.items():
        if not rows:
            continue
        ws = sorted(rows)
        types = {}
        for r in rows.values():
            for t, v in r["per_type"].items():
                a = types.setdefault(t, [0, 0]); a[0] += v.get("found", 0); a[1] += v.get("planted", 0)
        report["models"][m] = {
            "worlds": len(ws), "planted": sum(rows[w]["planted"] for w in ws),
            "recall": round(ratio(list(rows.values()), "found", "planted"), 3),
            "recall_95": interval(ws, lambda s: ratio([rows[w] for w in s], "found", "planted"), rng),
            "precision": round(ratio(list(rows.values()), "true_claims", "claims") or 0, 3),
            "precision_95": interval(ws, lambda s: ratio([rows[w] for w in s], "true_claims", "claims"), rng),
            "decoys": sum(rows[w]["decoys"] for w in ws),
            "predictions": f"{sum(rows[w]['predictions_correct'] for w in ws)}/{sum(rows[w]['predictions'] for w in ws)}",
            "dream_minutes_per_world": round(sum(rows[w]["dream_seconds"] for w in ws) / len(ws) / 60, 1),
            "per_type": {t: f"{a[0]}/{a[1]}" for t, a in types.items()},
        }
    both = sorted(set(data[MODELS[0]]) & set(data[MODELS[1]]))
    if both:
        a, b = data[MODELS[0]], data[MODELS[1]]
        diff = lambda s: ratio([a[w] for w in s], "found", "planted") - ratio([b[w] for w in s], "found", "planted")  # noqa: E731
        report["paired"] = {"worlds": len(both), f"recall_{MODELS[0]}_minus_{MODELS[1]}": round(diff(both), 3),
                            "recall_difference_95": interval(both, diff, rng)}
    out = ROOT / "results" / "eval_v2" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
