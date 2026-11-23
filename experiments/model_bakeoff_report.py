"""One table from the bake-off results (results/bakeoff/*.json): prints Markdown, writes summary.json.

  uv run python experiments/model_bakeoff_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bakeoff"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def row(r: dict) -> dict:
    n, yn, lo = r.get("notes") or {}, [y for y in r.get("yesno") or [] if "about" in y], r.get("lora") or {}
    return {
        "model": r["label"], "looped": r.get("looped"), "params_b": r.get("params_b"), "weights_mb": r.get("weights_gpu_mb"),
        "load_error": r.get("load_error"),
        "events": n.get("event_accuracy"), "places": n.get("place_recall"), "links": n.get("link_recall"),
        "links_true": n.get("link_precision"), "format": n.get("format_complete"),
        "records_per_min": n.get("records_per_minute"), "notes_error": n.get("error"),
        "about_auc": mean([y["about"]["auc"] for y in yn]),
        "about_yes_true": mean([y["about"]["at_0.5"]["yes_when_true"] for y in yn]),
        "about_yes_false": mean([y["about"]["at_0.5"]["yes_when_false"] for y in yn]),
        "traps_yes": sum(y["about"]["traps_said_yes"] for y in yn), "traps": sum(y["about"]["traps"] for y in yn),
        "same_auc": mean([y["same_kind"]["auc"] for y in yn]),
        "questions_per_s": mean([y["questions_per_second"] for y in yn]),
        "yesno_error": next((y.get("error") for y in r.get("yesno") or [] if "error" in y), None),
        "lora_fits": lo.get("fits"), "lora_mb": lo.get("peak_gpu_mb"), "lora_step_s": lo.get("step_seconds"),
        "lora_error": lo.get("error") if not lo.get("fits") else None,
    }


def fmt(v, pct=False):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.0%}" if pct else str(v)


def main() -> int:
    rows = [row(json.loads(p.read_text())) for p in sorted(OUT.glob("*.json")) if p.name != "summary.json"]
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2))
    print("| Model | Looped | Params | Events | Places | Links | Records/min | 'About?' AUC | yes if true / yes if false | Traps said yes | 'Same kind?' AUC | Questions/s | LoRA step fits (peak MB) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["load_error"]:
            print(f"| {r['model']} | | | didn't load: {r['load_error'][:80]} |||||||||")
            continue
        lora = f"{fmt(r['lora_fits'])} ({r['lora_mb']})" if r["lora_fits"] else (fmt(r["lora_fits"]) + (" (out of memory)" if r["lora_error"] else ""))
        print(f"| {r['model']} | {fmt(r['looped'])} | {r['params_b']}B | {fmt(r['events'], True)} | {fmt(r['places'], True)} | "
              f"{fmt(r['links'], True)} | {fmt(r['records_per_min'])} | {fmt(r['about_auc'])} | "
              f"{fmt(r['about_yes_true'], True)} / {fmt(r['about_yes_false'], True)} | {r['traps_yes']}/{r['traps']} | "
              f"{fmt(r['same_auc'])} | {fmt(r['questions_per_s'])} | {lora} |")
        for k in ("notes_error", "yesno_error", "lora_error"):
            if r[k]:
                print(f"|  ↳ {k}: {r[k][:140]} |||||||||||||")
    return 0


if __name__ == "__main__":
    sys.exit(main())
