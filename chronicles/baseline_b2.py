"""B2: a deliberately dumb, non-ML baseline. If it solves the worlds, the test is too easy.

* periodic:     for each keyword, look for years forming an arithmetic run (period P, +/- 2 years, one gap allowed)
* cause_effect: for each pair of keywords (a, b) and lag 1..5, count how often b follows a after exactly that lag
* chain:        join two cause_effect links a->b and b->c that line up at least twice
* identity:     not attempted (a keyword script can't resolve who is who)

  python -m chronicles.baseline_b2 --world worlds/test/test-000 --out claims.json
  python -m chronicles.baseline_b2 --split worlds/test          # score every world in a split
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STOP = set("""
year years grace reckoning remembered that this there their them they then than with from were when which
what into over under upon many much more most some such been being also very great after before other others
called nothing note said says happened later omen chronicler hand grew weary seen came come went made make
like long fell rose stood stayed held began first last every each could would should shall will night nights
days weeks whole
""".split())
WORD = re.compile(r"[a-z][a-z\-]+")


def keywords(text: str) -> set[str]:
    out = set()
    for w in WORD.findall(text.lower()):
        w = w.strip("-")
        if len(w) < 4 or w in STOP:
            continue
        if w.endswith("s") and len(w) > 4:
            w = w[:-1]
        out.add(w)
    return out


def load_records(world_dir: str | Path) -> list[dict]:
    with (Path(world_dir) / "records.jsonl").open() as fh:
        return [json.loads(line) for line in fh]


def _index(records: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        for w in keywords(r["text"]):
            idx[w].append(r)
    return idx


def _progression(years: list[int], period: int, tol: int = 2) -> list[int]:
    """Longest run y0, y0+P, y0+2P, ... (each within tol, at most one missing step in a row)."""
    best: list[int] = []
    ys = sorted(set(years))
    for y0 in ys:
        run, cur, misses = [y0], y0, 0
        while misses <= 1:
            target = cur + period
            hit = next((y for y in ys if abs(y - target) <= tol and y > cur), None)
            if hit is None:
                misses += 1
                cur = target
            else:
                run.append(hit)
                cur, misses = hit, 0
            if cur > ys[-1] + period:
                break
        if len(run) > len(best):
            best = run
    return best


def periodic_claims(idx: dict[str, list[dict]], n_records: int, pmin: int = 20, pmax: int = 200,
                    top: int = 10) -> list[dict]:
    cands = []
    for w, recs in idx.items():
        if not 4 <= len(recs) <= max(40, n_records // 8):
            continue
        ys = sorted({r["year"] for r in recs})
        periods = {b - a for i, a in enumerate(ys) for b in ys[i + 1:] if pmin <= b - a <= pmax}
        best_run, best_p = [], None
        for p in sorted(periods):
            run = _progression(ys, p)
            if len(run) > len(best_run):
                best_run, best_p = run, p
        if len(best_run) >= 4:
            gaps = [b - a for a, b in zip(best_run, best_run[1:])]
            period = round(sum(g if g < 1.5 * best_p else g / 2 for g in gaps) / len(gaps))
            ids = sorted({r["id"] for r in recs if r["year"] in set(best_run)})
            cands.append((len(best_run), {
                "type": "periodic", "record_ids": ids, "params": {"period": period},
                "prediction": {"year": best_run[-1] + period, "description": f"'{w}' again"},
                "summary": f"records with '{w}' recur every ~{period} years"}))
    return _dedupe([c for _, c in sorted(cands, key=lambda x: -x[0])], top)


def cause_effect_links(idx: dict[str, list[dict]], max_lag: int = 5) -> list[tuple[int, str, str, int, list[int]]]:
    years = {w: {r["year"] for r in recs} for w, recs in idx.items() if 3 <= len(recs) <= 30}
    links = []
    for a, ya in years.items():
        for b, yb in years.items():
            if a == b:
                continue
            for lag in range(1, max_lag + 1):
                hits = sorted(y for y in ya if y + lag in yb)
                if len(hits) >= 3 and len(hits) / len(ya) >= 0.5:
                    links.append((len(hits), a, b, lag, hits))
    links.sort(key=lambda x: -x[0])
    return links


def cause_effect_claims(idx, links, last_year: int, top: int = 10) -> list[dict]:
    out = []
    for n, a, b, lag, hits in links:
        ids = sorted({r["id"] for r in idx[a] if r["year"] in hits} |
                     {r["id"] for r in idx[b] if r["year"] - lag in hits})
        latest = max(r["year"] for r in idx[a])
        pred = {"year": latest + lag, "description": f"'{b}' after '{a}'"} if latest + lag > last_year else None
        out.append({"type": "cause_effect", "record_ids": ids, "params": {"lag": lag}, "prediction": pred,
                    "summary": f"'{a}' is followed by '{b}' after {lag} year(s) ({n} times)"})
    return _dedupe(out, top)


def chain_claims(idx, links, top: int = 5) -> list[dict]:
    out = []
    by_head = defaultdict(list)
    for n, a, b, lag, hits in links[:400]:
        by_head[a].append((b, lag, set(hits)))
    for a, outs in by_head.items():
        for b, l1, h1 in outs:
            for c, l2, h2 in by_head.get(b, []):
                if c == a:
                    continue
                starts = sorted(y for y in h1 if y + l1 in h2)
                if len(starts) >= 2:
                    ids = sorted({r["id"] for r in idx[a] if r["year"] in starts} |
                                 {r["id"] for r in idx[b] if r["year"] - l1 in starts} |
                                 {r["id"] for r in idx[c] if r["year"] - l1 - l2 in starts})
                    out.append((len(starts), {"type": "chain", "record_ids": ids, "params": {"lags": [l1, l2]},
                                              "prediction": None, "summary": f"'{a}' -> '{b}' -> '{c}'"}))
    return _dedupe([c for _, c in sorted(out, key=lambda x: -x[0])], top)


def _dedupe(claims: list[dict], top: int) -> list[dict]:
    kept: list[dict] = []
    for c in claims:
        s = set(c["record_ids"])
        if any(k["type"] == c["type"] and len(s & set(k["record_ids"])) / max(1, len(s)) >= 0.5 for k in kept):
            continue
        kept.append(c)
        if len(kept) >= top:
            break
    return kept


def run_b2(records: list[dict]) -> list[dict]:
    idx = _index(records)
    links = cause_effect_links(idx)
    last_year = max(r["year"] for r in records)
    return (periodic_claims(idx, len(records)) + cause_effect_claims(idx, links, last_year)
            + chain_claims(idx, links))


def main(argv: list[str] | None = None) -> int:
    from chronicles.scorer import aggregate, load_world_truth, score_world

    ap = argparse.ArgumentParser(prog="python -m chronicles.baseline_b2")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--world")
    g.add_argument("--split", help="a folder of worlds: run and score them all")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.world:
        claims = run_b2(load_records(args.world))
        if args.out:
            Path(args.out).write_text(json.dumps(claims, indent=2))
        score = score_world(load_world_truth(args.world), claims)
        print(json.dumps({k: v for k, v in score.items() if k != "rows"}, indent=2))
        return 0
    worlds = sorted(p for p in Path(args.split).iterdir() if (p / "truth.json").exists())
    scores = [score_world(load_world_truth(w), run_b2(load_records(w))) for w in worlds]
    agg = aggregate(scores)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
