"""Baseline B0, "just ask": Ouro reads the records directly and is asked for patterns. No notes, no search,
no tests. The records don't fit in its memory at once (a looped model's cache costs ~0.8 MB per token),
so it reads them in windows of consecutive records and is asked about each window.

  uv run python experiments/phase3_b0.py worlds/test/test-000 worlds/test/test-003 ...

Answers are free text. Each "Pattern:" line becomes a claim: the records whose years the line mentions,
typed periodic if it talks about a regular return, else cause_effect, with the period or lag read from
the years. Writes results/phase3/b0_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chronicles.scorer import aggregate, score_world  # noqa: E402
from learner.model import load_ouro  # noqa: E402

PROMPT = """Below are records from an old chronicle, in order.

{records}

Question: Which kinds of events in these records come back at regular intervals, or are always followed by another kind of event a few years later? Give one pattern per line, with the years, like this:
Pattern: a red comet returns about every 40 years (1201, 1241, 1282)
Pattern: a drought in Velsham is followed by a famine one year later (1210 -> 1211, 1250 -> 1251)

Pattern:"""


def to_claims(text: str, records: list[dict]) -> list[dict]:
    claims = []
    by_year: dict[int, list[str]] = {}
    for r in records:
        by_year.setdefault(r["year"], []).append(r["id"])
    for line in ("Pattern:" + text).split("Pattern:"):
        line = line.strip().splitlines()[0] if line.strip() else ""
        years = sorted({int(y) for y in re.findall(r"\b(\d{3,4})\b", line) if int(y) in by_year})
        if len(years) < 2:
            continue
        rids = [rid for y in years for rid in by_year[y]]
        diffs = [b - a for a, b in zip(years, years[1:])]
        if re.search(r"every|regular|return|again|interval|cycle", line, re.I) or min(diffs) > 5:
            claims.append({"type": "periodic", "record_ids": rids, "params": {"period": round(median(diffs))},
                           "prediction": {"year": years[-1] + round(median(diffs))}, "summary": line})
        else:
            small = [d for d in diffs if d <= 5] or [1]
            claims.append({"type": "cause_effect", "record_ids": rids, "params": {"lag": round(median(small))},
                           "prediction": None, "summary": line})
    return claims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--window", type=int, default=70, help="records per window")
    ap.add_argument("--loops", type=int, default=4)
    args = ap.parse_args()
    lm = load_ouro(loops=args.loops)
    scores, per_world = [], []
    for w in map(Path, args.worlds):
        records = [json.loads(line) for line in (w / "records.jsonl").open()]
        truth = json.loads((w / "truth.json").read_text())
        claims, raw = [], []
        t0 = time.perf_counter()
        step = args.window * 3 // 4  # windows overlap by a quarter
        for i in range(0, len(records), step):
            win = records[i:i + args.window]
            text = "\n".join(f"{r['year']}: {r['text']}" for r in win)
            out = lm.generate([PROMPT.format(records=text)], max_new_tokens=160)[0]
            raw.append(out)
            claims += to_claims(out, win)
            if i + args.window >= len(records):
                break
        s = score_world(truth, claims)
        scores.append(s)
        per_world.append({"world": w.name, "seconds": round(time.perf_counter() - t0, 1),
                          **{k: s[k] for k in ("claims", "found", "planted", "recall", "precision", "decoy_hits")},
                          "answers": raw})
        print(f"{w.name}: {s['found']}/{s['planted']} found, {s['claims']} claims, precision {s['precision']:.2f}",
              flush=True)
    report = {"baseline": "B0 just ask", "window": args.window, "aggregate": aggregate(scores), "worlds": per_world}
    split = Path(args.worlds[0]).parent.name
    out = ROOT / "results" / "phase3" / f"b0_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
