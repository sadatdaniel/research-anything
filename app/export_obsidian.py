"""Export what the learner found as a folder of Markdown notes that Obsidian (or any Markdown app) opens.

  uv run python -m app.export_obsidian --db results/runs/phase3/dream/val-001-dev4.sqlite --out findings/val-001

One note per connection, per record, per kind of event and per place or person, joined by [[wikilinks]],
so Obsidian's graph view and backlinks show the connections as they are. Front matter carries the stage,
type and confidence. Evidence quotes link to their record's note, which holds the record's full text.

  README.md         start here: what was found, best first
  Connections/      one note per theory (confirmed, being checked, or a consequence of another)
  Records/          one note per source record: year, text, kind, places
  Kinds/            one note per kind of event, listing its records
  Places/           one note per place or person name, with its other names and where it lies
  Timeline.md       the records century by century
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learner import theories as th  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402

SHOW = ("confirmed", "prediction-tested", "superseded")
STAGE_WORDS = {"confirmed": "confirmed", "prediction-tested": "tested, source check pending",
               "superseded": "superseded", "rejected": "rejected", "proposed": "proposed"}


def safe(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "", name).strip()[:80] or "untitled"


def fm(d: dict) -> str:
    lines = ["---"]
    for k, v in d.items():
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines + ["---", ""])


def export(db: Path, out: Path, include_rejected: bool = False) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    k = Knowledge(conn)
    theories = [t for t in th.all_theories(conn) if t["stage"] in SHOW or (include_rejected and t["stage"] == "rejected")]
    theories.sort(key=lambda t: (t["stage"] != "confirmed", -t["confidence"]))
    if out.exists():
        shutil.rmtree(out)
    for sub in ("Connections", "Records", "Kinds", "Places"):
        (out / sub).mkdir(parents=True)

    def rec_link(rid: str) -> str:
        e = k.by_id[rid]
        return f"[[Records/{rid}|{e.year} {e.kind}]]"

    title = {t["id"]: safe(f"{t['id']} {t['summary'].split(' (')[0]}") for t in theories}
    in_theory: dict[str, list[int]] = defaultdict(list)
    for t in theories:
        for rid in t["record_ids"]:
            in_theory[rid].append(t["id"])

    # connections
    for t in theories:
        kinds = sorted({g[0] for g in (t["groups"] or [])})
        places = sorted({g[1] for g in (t["groups"] or []) if g[1]})
        body = [fm({"type": t["type"], "stage": t["stage"], "confidence": t["confidence"], "params": t["params"],
                    "prediction": t["prediction"], "tags": ["connection", t["type"], t["stage"]]}),
                f"# {t['summary']}", "",
                f"**Stage:** {STAGE_WORDS[t['stage']]} · **confidence {t['confidence']}/100** "
                f"(chance {t['score_parts']['chance']}, predictions {t['score_parts']['prediction']}, "
                f"support {t['score_parts']['support']}, source check {t['score_parts']['source']})", ""]
        if t["stage"] != "confirmed" and t["reason"]:
            body += [f"> {t['reason']}", ""]
        if t["prediction"] and t["prediction"].get("year"):
            body += [f"**Expects next:** about {t['prediction']['year']}", ""]
        if kinds or places:
            body += ["**About:** " + ", ".join([f"[[Kinds/{safe(x)}|{x}]]" for x in kinds] +
                                                 [f"[[Places/{safe(x)}|{x}]]" for x in places]), ""]
        body += ["## Tests", ""] + [f"- {'✅' if x['passed'] else '❌'} **{x['test']}**: {x['detail']}"
                                    for x in (t["tests"] or [])] + [""]
        if t["evidence"]:
            body += ["## Evidence, word for word", ""]
            for ev in t["evidence"]:
                body += [f"- {rec_link(ev['record_id'])}: “{ev['quote']}”"]
            body.append("")
        body += ["## Records", ""] + [f"- {rec_link(r)}: {k.by_id[r].text}" for r in
                                      sorted(t["record_ids"], key=lambda r: k.by_id[r].year)]
        (out / "Connections" / f"{title[t['id']]}.md").write_text("\n".join(body) + "\n")

    # records
    by_kind: dict[str, list[str]] = defaultdict(list)
    by_name: dict[str, list[str]] = defaultdict(list)
    for e in k.entries:
        by_kind[e.kind].append(e.record_id)
        for n in e.names:
            by_name[n].append(e.record_id)
        links = [f"[[Kinds/{safe(e.kind)}|{e.kind}]]"] + [f"[[Places/{safe(n)}|{n}]]" for n in sorted(e.names)]
        body = [fm({"year": e.year, "kind": e.kind, "noted_as": e.label, "tags": ["record"]}),
                f"# {e.year}: {e.kind}", "", f"> {e.text}", "", "**Mentions:** " + ", ".join(links), ""]
        if in_theory.get(e.record_id):
            body += ["**Part of:** " + ", ".join(f"[[Connections/{title[i]}]]" for i in in_theory[e.record_id]), ""]
        (out / "Records" / f"{e.record_id}.md").write_text("\n".join(body))

    for kind, rids in by_kind.items():
        body = [fm({"tags": ["kind"], "records": len(rids)}), f"# {kind}", "",
                f"{len(rids)} records.", ""] + [f"- {rec_link(r)}" for r in rids]
        (out / "Kinds" / f"{safe(kind)}.md").write_text("\n".join(body) + "\n")

    others: dict[str, set[str]] = defaultdict(set)
    for w, c in k.canon_of.items():
        if w != c:
            others[c].add(w)
    for name, rids in by_name.items():
        body = [fm({"tags": ["place-or-name"], "aliases": sorted(others.get(name, []))}), f"# {name}", ""]
        if others.get(name):
            body += [f"**Also written:** {', '.join(sorted(others[name]))} (stated in the records)", ""]
        if k.parents.get(name):
            body += ["**Lies in / on:** " + ", ".join(f"[[Places/{safe(p)}|{p}]]" for p in sorted(k.parents[name])), ""]
        body += [f"{len(rids)} records.", ""] + [f"- {rec_link(r)}" for r in rids]
        (out / "Places" / f"{safe(name)}.md").write_text("\n".join(body) + "\n")

    timeline = ["# Timeline", ""]
    century = None
    for e in k.entries:
        c = e.year // 100
        if c != century:
            century = c
            timeline += ["", f"## {c * 100}s", ""]
        timeline.append(f"- {rec_link(e.record_id)}: {e.text}")
    (out / "Timeline.md").write_text("\n".join(timeline) + "\n")

    conf = [t for t in theories if t["stage"] == "confirmed"]
    readme = [f"# What the learner found ({db.stem})", "",
              f"{len(k.entries)} records, {k.first_year}–{k.last_year}. "
              f"{len(conf)} confirmed connections; open the graph view to see how they link records, kinds and places.", "",
              "## Confirmed, best first", ""]
    readme += [f"- [[Connections/{title[t['id']]}|{t['summary']}]] — confidence {t['confidence']}" for t in conf] or ["- none yet"]
    rest = [t for t in theories if t["stage"] != "confirmed"]
    if rest:
        readme += ["", "## Also kept (being checked, or a consequence of another connection)", ""]
        readme += [f"- [[Connections/{title[t['id']]}|{t['summary']}]] — {STAGE_WORDS[t['stage']]}" for t in rest]
    readme += ["", "Start points: [[Timeline]], the `Kinds/` and `Places/` folders. Every quote links to its record."]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    return {"connections": len(theories), "records": len(k.entries), "kinds": len(by_kind), "places": len(by_name)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.export_obsidian")
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-rejected", action="store_true")
    args = ap.parse_args(argv)
    print(export(Path(args.db).resolve(), Path(args.out), args.include_rejected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
