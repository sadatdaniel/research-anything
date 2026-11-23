"""Phase 3: how well does Ouro answer dream mode's two yes/no questions? Graded against the answer key.

  uv run python experiments/phase3_prompt_check.py worlds/val/val-000 [--loops 4]

  same kind?   every pair of note labels in the world; truth = the labels' records are mostly the same kind
  about?       records asked about their true kind (should be yes) and about another kind seen in the
               world (should be no), including the planted figure-of-speech traps ("a flood of pilgrims")
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learner.dream import ABOUT, SAME_KIND, kind_phrase  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.model import load_ouro  # noqa: E402


def rates(rows: list[tuple[float, bool]], at: float) -> dict:
    tp = sum(1 for p, y in rows if p >= at and y)
    fp = sum(1 for p, y in rows if p >= at and not y)
    pos = sum(1 for _, y in rows if y)
    neg = len(rows) - pos
    return {"threshold": at, "yes_when_true": round(tp / pos, 3) if pos else None,
            "yes_when_false": round(fp / neg, 3) if neg else None, "pos": pos, "neg": neg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--about", type=int, default=120, help="records to ask about")
    args = ap.parse_args()
    world = Path(args.world)
    truth = json.loads((world / "truth.json").read_text())
    conn = sqlite3.connect(ROOT / "results" / "runs" / "phase3" / f"{world.name}.sqlite", isolation_level=None)
    k = Knowledge(conn)
    lm = load_ouro(loops=args.loops)

    label_kinds: dict[str, Counter] = defaultdict(Counter)
    for e in k.entries:
        label_kinds[e.label][truth["records"][e.record_id]["kind"]] += 1
    major = {lab: c.most_common(1)[0][0] for lab, c in label_kinds.items()}
    pairs = list(combinations(sorted(major), 2))
    t0 = time.perf_counter()
    ps = lm.p_yes([SAME_KIND.format(a=a, b=b) for a, b in pairs], batch=16)
    same_s = time.perf_counter() - t0
    same_rows = [(p, major[a] == major[b]) for (a, b), p in zip(pairs, ps)]
    mistakes = sorted(((p, a, b) for (a, b), p in zip(pairs, ps) if (p >= 0.8) != (major[a] == major[b])), reverse=True)

    rng = random.Random(0)
    kinds = sorted({e.kind for e in k.entries})
    sample = rng.sample(k.entries, min(args.about, len(k.entries)))
    qs = []
    for e in sample:
        true_kind = truth["records"][e.record_id]["kind"]
        own = e.kind if major.get(e.label) == true_kind else None
        if own:
            qs.append((e, own, True))
        other = rng.choice([x for x in kinds if major.get(x) not in (true_kind, None) and x != e.kind] or kinds)
        qs.append((e, other, False))
    traps = [e for e in k.entries if "flood of" in e.text.lower() and truth["records"][e.record_id]["kind"] != "flood"]
    qs += [(e, "flood", False) for e in traps]
    t0 = time.perf_counter()
    ps2 = lm.p_yes([ABOUT.format(text=e.text, kind=kind_phrase(kd)) for e, kd, _ in qs], batch=16)
    about_s = time.perf_counter() - t0
    about_rows = [(p, y) for (_, _, y), p in zip(qs, ps2)]
    trap_ps = [round(p, 3) for (e, kd, y), p in zip(qs, ps2) if e in traps and kd == "flood"]

    report = {
        "world": world.name, "loops": args.loops,
        "same_kind": {"pairs": len(pairs), "seconds": round(same_s, 1),
                      "at_0.5": rates(same_rows, 0.5), "at_0.8": rates(same_rows, 0.8), "at_0.9": rates(same_rows, 0.9),
                      "mistakes_at_0.8": [[round(p, 3), a, b] for p, a, b in mistakes[:30]]},
        "about": {"questions": len(qs), "seconds": round(about_s, 1),
                  "at_0.5": rates(about_rows, 0.5), "at_0.8": rates(about_rows, 0.8),
                  "figure_of_speech_traps": trap_ps,
                  "wrong_at_0.5": [[round(p, 3), kd, y, e.text[:90]] for (e, kd, y), p in zip(qs, ps2)
                                   if (p >= 0.5) != y][:30]},
    }
    out = ROOT / "results" / "phase3" / f"prompts_{world.name}_loops{args.loops}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
