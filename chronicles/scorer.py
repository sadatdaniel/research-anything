"""Score a learner's claims against a world's answer key.

Claim format (one JSON object per claim; a claims file is a JSON list of them):

  {
    "type": "periodic" | "cause_effect" | "chain" | "identity",
    "record_ids": ["r0011", "r0040", ...],     # the visible records the connection is built from
    "params": {"period": 72} | {"lag": 2} | {"lags": [1, 2]} | {"aliases": [...]},
    "prediction": {"year": 1262, "description": "..."} | null,   # optional: what happens next
    "evidence": [{"record_id": "r0011", "quote": "exact words from that record"}],   # checked by the learner
    "summary": "free text"
  }

Matching a claim to a planted pattern (or a decoy):
  * same type;
  * it shares at least min(3, size of the pattern) records with the pattern, AND at least half of the
    claim's records belong to the pattern (so a claim can't win by listing everything);
  * parameters within tolerance: period +/- max(2, 5%), lag exact if <= 2 else +/- 1, chain lags likewise.
    Identity needs no parameters.

Predictions count only if they are about the hidden years and land close to a held-out occurrence:
  * periodic: within max(2 x jitter, 5% of the period). Every occurrence is shifted by up to +/- jitter
    years, so even a correct "last sighting + period" can be off by twice the jitter;
  * cause_effect and chain: within 1 year.

Metrics: recall (planted patterns found), precision (claims that match a planted pattern),
decoy hits, duplicates, and prediction accuracy against the held-out years.

  python -m chronicles.scorer --world worlds/test/test-000 --claims claims.json [--out score.json]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

TYPES = ("periodic", "cause_effect", "chain", "identity")


@dataclass
class Claim:
    type: str
    record_ids: list[str]
    params: dict
    prediction: dict | None = None
    evidence: list | None = None
    summary: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(type=d["type"], record_ids=list(d.get("record_ids", [])), params=dict(d.get("params") or {}),
                   prediction=d.get("prediction"), evidence=d.get("evidence"), summary=d.get("summary", ""))


def _lag_ok(got, want) -> bool:
    try:
        got, want = int(got), int(want)
    except (TypeError, ValueError):
        return False
    return got == want if want <= 2 else abs(got - want) <= 1


def params_ok(claim: Claim, target: dict) -> bool:
    want, got = target.get("params", {}), claim.params
    t = target["type"]
    if t == "periodic":
        if "period" not in got or "period" not in want:
            return False
        p = float(want["period"])
        return abs(float(got["period"]) - p) <= max(2.0, 0.05 * p)
    if t == "cause_effect":
        return "lag" in got and "lag" in want and _lag_ok(got["lag"], want["lag"])
    if t == "chain":
        g, w = got.get("lags") or [], want.get("lags") or []
        return len(g) == len(w) and all(_lag_ok(a, b) for a, b in zip(g, w))
    return True  # identity: the records say it all


def overlap_ok(claim_ids: set[str], target_ids: set[str]) -> bool:
    if not claim_ids or not target_ids:
        return False
    shared = len(claim_ids & target_ids)
    return shared >= min(3, len(target_ids)) and shared / len(claim_ids) >= 0.5


def matches(claim: Claim, target: dict) -> bool:
    return (claim.type == target["type"] and overlap_ok(set(claim.record_ids), set(target["record_ids"]))
            and params_ok(claim, target))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def prediction_ok(claim: Claim, pattern: dict, heldout_from: int) -> bool:
    pred = claim.prediction or {}
    try:
        year = int(pred["year"])
    except (KeyError, TypeError, ValueError):
        return False
    if year < heldout_from:
        return False  # only predictions about the unseen future count
    if pattern["type"] == "periodic":
        tol = max(2.0 * pattern.get("jitter", 2), 0.05 * float(pattern["params"]["period"]))
    else:
        tol = 1.0
    return any(abs(year - y) <= tol for y in pattern.get("heldout_years", []))


def score_world(truth: dict, claims: list[dict | Claim]) -> dict:
    claims = [c if isinstance(c, Claim) else Claim.from_dict(c) for c in claims]
    patterns, decoys, cut = truth["patterns"], truth["decoys"], truth["heldout_from"]
    found: dict[str, int] = {}
    rows = []
    for i, claim in enumerate(claims):
        candidates = [p for p in patterns if matches(claim, p)]
        best = max(candidates, key=lambda p: _jaccard(set(claim.record_ids), set(p["record_ids"])), default=None)
        decoy = next((d for d in decoys if matches(claim, d)), None) if best is None else None
        row = {"claim": i, "type": claim.type, "matched": best["id"] if best else None,
               "decoy": decoy["id"] if decoy else None, "duplicate": False,
               "predicted": bool(claim.prediction), "prediction_correct": None}
        if best is not None:
            row["duplicate"] = best["id"] in found
            found[best["id"]] = found.get(best["id"], 0) + 1
            if claim.prediction:
                row["prediction_correct"] = prediction_ok(claim, best, cut)
        rows.append(row)

    n = len(claims)
    true_claims = sum(1 for r in rows if r["matched"])
    per_type = {}
    for t in TYPES:
        planted = [p for p in patterns if p["type"] == t]
        if planted:
            per_type[t] = {"planted": len(planted), "found": sum(1 for p in planted if p["id"] in found)}
    predictable = [r for r in rows if r["matched"] and r["predicted"] and not r["duplicate"]]
    return {
        "claims": n,
        "planted": len(patterns),
        "found": len(found),
        "recall": len(found) / len(patterns) if patterns else 0.0,
        "precision": true_claims / n if n else 0.0,
        "true_claims": true_claims,
        "duplicates": sum(1 for r in rows if r["duplicate"]),
        "decoy_hits": sum(1 for r in rows if r["decoy"]),
        "false_claims": sum(1 for r in rows if not r["matched"] and not r["decoy"]),
        "predictions": len(predictable),
        "predictions_correct": sum(1 for r in predictable if r["prediction_correct"]),
        "per_type": per_type,
        "found_ids": sorted(found),
        "rows": rows,
    }


def aggregate(scores: list[dict]) -> dict:
    """Micro-average over worlds: every planted pattern and every claim counts once."""
    tot = {k: sum(s[k] for s in scores) for k in ("claims", "planted", "found", "true_claims", "duplicates",
                                                   "decoy_hits", "false_claims", "predictions",
                                                   "predictions_correct")}
    per_type: dict[str, dict[str, int]] = {}
    for s in scores:
        for t, v in s["per_type"].items():
            acc = per_type.setdefault(t, {"planted": 0, "found": 0})
            acc["planted"] += v["planted"]
            acc["found"] += v["found"]
    for v in per_type.values():
        v["recall"] = v["found"] / v["planted"] if v["planted"] else 0.0
    return {
        "worlds": len(scores), **tot,
        "recall": tot["found"] / tot["planted"] if tot["planted"] else 0.0,
        "precision": tot["true_claims"] / tot["claims"] if tot["claims"] else 0.0,
        "prediction_accuracy": tot["predictions_correct"] / tot["predictions"] if tot["predictions"] else None,
        "per_type": per_type,
    }


def truth_as_claims(truth: dict) -> list[dict]:
    """The answer key written as claims: a perfect learner's output. Used to test the scorer."""
    out = []
    for p in truth["patterns"]:
        pred = None
        if p.get("heldout_years"):
            pred = {"year": p["heldout_years"][0], "description": "from the answer key"}
        out.append({"type": p["type"], "record_ids": p["record_ids"], "params": p["params"], "prediction": pred})
    return out


def load_world_truth(world_dir: str | Path) -> dict:
    return json.loads((Path(world_dir) / "truth.json").read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chronicles.scorer")
    ap.add_argument("--world", required=True)
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    score = score_world(load_world_truth(args.world), json.loads(Path(args.claims).read_text()))
    text = json.dumps(score, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps({k: v for k, v in score.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
