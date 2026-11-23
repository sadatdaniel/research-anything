"""Phase 3 summary over a set of worlds: dream mode against the baselines, and the H1 curve.

  uv run python experiments/phase3_report.py --tag final worlds/test/test-000 worlds/test/test-002 ...

Reads results/phase3/dream_<world>-<tag>.json (from phase3_dream.py) and results/phase3/b0_<split>.json
(from phase3_b0.py, if present), runs B2 itself, and writes results/phase3/summary_<split>-<tag>.json.

  B0  just ask: the model reads windows of records and is asked for patterns
  B1  search without testing: every proposal from the final knowledge, untested
  B2  a script: keyword clusters plus a period finder, no model
  dream  confirmed theories (proposed, tested, source-checked), against active dream time
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronicles.baseline_b2 import load_records, run_b2  # noqa: E402
from chronicles.scorer import aggregate, load_world_truth, score_world  # noqa: E402

KEYS = ("claims", "found", "planted", "recall", "precision", "decoy_hits", "predictions", "predictions_correct")


def micro(rows: list[dict]) -> dict:
    claims = sum(r["claims"] for r in rows)
    true = sum(round(r["precision"] * r["claims"]) for r in rows)
    found, planted = sum(r["found"] for r in rows), sum(r["planted"] for r in rows)
    preds, good = sum(r["predictions"] for r in rows), sum(r["predictions_correct"] for r in rows)
    return {"worlds": len(rows), "claims": claims, "found": found, "planted": planted,
            "recall": round(found / planted, 3) if planted else 0.0,
            "precision": round(true / claims, 3) if claims else None,
            "decoy_hits": sum(r["decoy_hits"] for r in rows),
            "prediction_accuracy": round(good / preds, 3) if preds else None, "predictions": preds}


def per_type(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        for t, v in r["per_type"].items():
            acc = out.setdefault(t, {"planted": 0, "found": 0})
            acc["planted"] += v["planted"]
            acc["found"] += v["found"]
    for v in out.values():
        v["recall"] = round(v["found"] / v["planted"], 3)
    return out


def at_time(curve: list[dict], seconds: float) -> dict:
    best = curve[0]
    for p in curve:
        if p["dream_seconds"] <= seconds:
            best = p
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    worlds = [Path(w) for w in args.worlds]
    split = worlds[0].parent.name
    dreams = [json.loads((ROOT / "results" / "phase3" / f"dream_{w.name}-{args.tag}.json").read_text()) for w in worlds]
    b2 = [score_world(load_world_truth(w), run_b2(load_records(w))) for w in worlds]
    b0_path = ROOT / "results" / "phase3" / f"b0_{split}.json"
    b0 = None
    if b0_path.exists():
        b0_all = {x["world"]: x for x in json.loads(b0_path.read_text())["worlds"]}
        if all(w.name in b0_all for w in worlds):
            b0 = {"recall": round(sum(b0_all[w.name]["found"] for w in worlds) /
                                  sum(b0_all[w.name]["planted"] for w in worlds), 3),
                  "claims": sum(b0_all[w.name]["claims"] for w in worlds),
                  "found": sum(b0_all[w.name]["found"] for w in worlds),
                  "decoy_hits": sum(b0_all[w.name]["decoy_hits"] for w in worlds)}
    grid = [0, 15, 30, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800]
    end = max(d["dream_seconds"] for d in dreams)
    grid = [g for g in grid if g < end] + [end]
    curve = []
    for g in grid:
        rows = [at_time(d["curve"], g) for d in dreams]
        curve.append({"dream_seconds": round(g, 1), **micro(rows)})
    report = {
        "split": split, "tag": args.tag, "worlds": [w.name for w in worlds],
        "dream": {**micro([d["confirmed"] for d in dreams]), "per_type": per_type([d["confirmed"] for d in dreams])},
        "b1_all_proposals": {**micro([d["b1_all_proposals"] for d in dreams]),
                             "per_type": per_type([d["b1_all_proposals"] for d in dreams])},
        "b2_script": {k: v for k, v in aggregate(b2).items() if k != "per_type"} | {"per_type": aggregate(b2)["per_type"]},
        "b0_just_ask": b0,
        "dream_seconds_per_world": [d["dream_seconds"] for d in dreams],
        "questions_per_world": [d["counts"]["questions"] for d in dreams],
        "h1_curve": curve,
    }
    out = ROOT / "results" / "phase3" / f"summary_{split}-{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("dream", "b1_all_proposals", "b2_script", "b0_just_ask")}, indent=2))
    print("H1 curve (dream seconds: found/planted, precision):")
    for p in curve:
        print(f"  {p['dream_seconds']:7.0f}s  {p['found']}/{p['planted']}  precision {p['precision']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
