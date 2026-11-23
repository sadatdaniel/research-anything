"""Model bake-off: is the looped Ouro still the best model for this learner, or is a newer one better?

  uv run python experiments/model_bakeoff.py --model ouro-1.4b --loops 4
  uv run --with "transformers==5.24.0" python experiments/model_bakeoff.py --model qwen3.5-2b-base

Measured on validation worlds the prompts were never tuned on (the prompts were written with Ouro on
val-000 and val-001, so that would favour it), with identical inputs for every model:

  notes   read the first 100 records of val-003 with the day-reading prompt, graded against the answer key
          (events, places, links), plus records per minute
  yesno   dream mode's two yes/no questions on val-001 and val-002. Every model gets the same questions,
          built from the notebook Ouro made, so only the answers differ. Reported: how often it says yes when
          the answer is yes and when it's no (at 0.5), and AUC (how well its yes-probability separates
          true from false, whatever threshold; 0.5 is chance, 1.0 is perfect)
  lora    whether one LoRA training step on 512 tokens fits in this GPU (sleep needs it), its time and memory

Writes results/bakeoff/<model>@<loops>.json. Each task is run on its own, so a model that runs out of
memory in one still gets scored on the others.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from chronicles.notes_eval import grade  # noqa: E402
from learner import notebook  # noqa: E402
from learner.dream import ABOUT, SAME_KIND, kind_phrase  # noqa: E402
from learner.focused import FEW_SHOT, read_batch  # noqa: E402
from learner.knowledge import Knowledge  # noqa: E402
from learner.model import MODELS, load_model  # noqa: E402

OUT = ROOT / "results" / "bakeoff"
RUNS = ROOT / "results" / "runs" / "bakeoff"
NOTE_KEYS = ("format_complete", "event_accuracy", "place_recall", "place_precision", "link_recall", "link_precision",
             "colour_captured", "direction_captured", "duration_captured")


def mb(x: float) -> int:
    return round(x / 2**20)


def auc(rows: list[tuple[float, bool]]) -> float | None:
    pos = [p for p, y in rows if y]
    neg = [p for p, y in rows if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def rates(rows: list[tuple[float, bool]], at: float) -> dict:
    pos = [p for p, y in rows if y]
    neg = [p for p, y in rows if not y]
    return {"yes_when_true": round(sum(p >= at for p in pos) / len(pos), 3) if pos else None,
            "yes_when_false": round(sum(p >= at for p in neg) / len(neg), 3) if neg else None}


def p_yes_adaptive(lm, prompts: list[str], batch: int = 16) -> list[float]:
    """p_yes in batches, halving the batch when it doesn't fit (retrying outside the except block)."""
    out, i = [], 0
    while i < len(prompts):
        got = None
        try:
            got = lm.p_yes(prompts[i:i + batch], batch=batch)
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
        if got is None:
            torch.cuda.empty_cache()
            batch = max(1, batch // 2)
            continue
        out += got
        i += batch
    return out


def task_notes(lm, label: str, world: Path, limit: int) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    db = RUNS / f"{label}_{world.name}.sqlite"
    db.unlink(missing_ok=True)
    conn = sqlite3.connect(db, isolation_level=None)
    notebook.ingest_world(conn, world)
    todo = notebook.unread(conn, limit)
    tokens = len(lm.tokenizer(FEW_SHOT)["input_ids"]) + 60 + 110
    batch = lm.safe_batch(tokens, cap=8)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        notes_ = read_batch(lm, chunk)
        conn.execute("BEGIN IMMEDIATE")
        for r, n in zip(chunk, notes_):
            notebook.save_note(conn, n, r["text"], lm.loops)
        conn.execute("COMMIT")
    seconds = time.perf_counter() - t0
    truth = json.loads((world / "truth.json").read_text())
    sources = {r[0]: r[1] for r in conn.execute("SELECT record_id, text FROM sources")}
    by_record: dict[str, list] = {}
    for link in notebook.links(conn):
        by_record.setdefault(link["record_id"], []).append(link)
    rows = [{**n, "links": by_record.get(n["record_id"], [])} for n in notebook.notes(conn)]
    report = grade(truth, rows, sources)
    return {**{k: report.get(k) for k in NOTE_KEYS}, "world": world.name, "records": len(todo), "batch": batch,
            "seconds_per_record": round(seconds / max(1, len(todo)), 2),
            "records_per_minute": round(60 * len(todo) / seconds, 1), "peak_gpu_mb": mb(torch.cuda.max_memory_allocated()),
            "wrong_events": report.get("wrong_events", [])[:12]}


def task_yesno(lm, world_name: str, about_n: int = 120) -> dict:
    """The same questions as experiments/phase3_prompt_check.py, from the same notebook for every model."""
    world = ROOT / "worlds" / "val" / world_name
    truth = json.loads((world / "truth.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "nb.sqlite"
        shutil.copy(ROOT / "results" / "runs" / "phase3" / f"{world_name}.sqlite", db)
        k = Knowledge(sqlite3.connect(db, isolation_level=None))
    label_kinds: dict[str, Counter] = defaultdict(Counter)
    for e in k.entries:
        label_kinds[e.label][truth["records"][e.record_id]["kind"]] += 1
    major = {lab: c.most_common(1)[0][0] for lab, c in label_kinds.items()}
    pairs = list(combinations(sorted(major), 2))
    rng = random.Random(0)
    kinds = sorted({e.kind for e in k.entries})
    qs = []
    for e in rng.sample(k.entries, min(about_n, len(k.entries))):
        true_kind = truth["records"][e.record_id]["kind"]
        if major.get(e.label) == true_kind:
            qs.append((e, e.kind, True))
        other = rng.choice([x for x in kinds if major.get(x) not in (true_kind, None) and x != e.kind] or kinds)
        qs.append((e, other, False))
    traps = [e for e in k.entries if "flood of" in e.text.lower() and truth["records"][e.record_id]["kind"] != "flood"]
    qs += [(e, "flood", False) for e in traps]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ps_same = p_yes_adaptive(lm, [SAME_KIND.format(a=a, b=b) for a, b in pairs])
    ps_about = p_yes_adaptive(lm, [ABOUT.format(text=e.text, kind=kind_phrase(kd)) for e, kd, _ in qs])
    seconds = time.perf_counter() - t0
    same = [(p, major[a] == major[b]) for (a, b), p in zip(pairs, ps_same)]
    about = [(p, y) for (_, _, y), p in zip(qs, ps_about)]
    return {"world": world_name,
            "about": {"questions": len(about), "auc": auc(about), "at_0.5": rates(about, 0.5),
                      "traps_said_yes": sum(p >= 0.5 for (e, kd, y), p in zip(qs, ps_about) if e in traps), "traps": len(traps)},
            "same_kind": {"questions": len(same), "auc": auc(same), "at_0.5": rates(same, 0.5), "at_0.8": rates(same, 0.8)},
            "questions_per_second": round((len(same) + len(about)) / seconds, 1), "peak_gpu_mb": mb(torch.cuda.max_memory_allocated())}


def task_lora(lm, world: Path) -> dict:
    from peft import LoraConfig, get_peft_model

    model = lm.model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = [t for t in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if t in names]
    lora = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=targets))
    text = "\n".join(json.loads(line)["text"] for line in (world / "records.jsonl").read_text().splitlines()[:80])
    ids = lm.tokenizer(text, return_tensors="pt")["input_ids"][0][:512].unsqueeze(0).to("cuda")
    optim = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-4)
    lora.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = lora(input_ids=ids, labels=ids).loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return {"fits": True, "seq_len": int(ids.shape[1]), "targets": targets, "step_seconds": round(sum(times[1:]) / 2, 2),
            "peak_gpu_mb": mb(torch.cuda.max_memory_allocated())}


def run(name: str, fn) -> dict:
    t0 = time.perf_counter()
    try:
        out = fn()
    except torch.OutOfMemoryError as exc:
        out = {"fits": False, "error": "out of GPU memory: " + str(exc).splitlines()[0][:160]}
    except Exception as exc:  # report and keep going: one failing task mustn't hide the others
        out = {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "trace": traceback.format_exc()[-1200:]}
    torch.cuda.empty_cache()
    out["task_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  {name}: {json.dumps({k: v for k, v in out.items() if k not in ('wrong_events', 'trace')})[:600]}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--loops", type=int, default=4, help="looped models only")
    ap.add_argument("--tasks", default="notes,yesno,lora")
    ap.add_argument("--notes-world", default="worlds/val/val-003")
    ap.add_argument("--notes-limit", type=int, default=100)
    args = ap.parse_args()
    import transformers

    label = f"{args.model}@{args.loops}" if args.model.startswith("ouro") else args.model
    tasks = args.tasks.split(",")
    report = {"model": args.model, "hub_id": MODELS[args.model][0], "label": label,
              "transformers": transformers.__version__, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(0), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"== {label}", flush=True)
    t0 = time.perf_counter()
    try:
        lm = load_model(args.model, loops=args.loops)
    except Exception as exc:
        report["load_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        print("  load failed:", report["load_error"], flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{label}.json").write_text(json.dumps(report, indent=2))
        return 1
    report.update(load_seconds=round(time.perf_counter() - t0, 1), loops=lm.loops, looped=lm.looped,
                  params_b=round(sum(p.numel() for p in lm.model.parameters()) / 1e9, 2),
                  weights_gpu_mb=mb(torch.cuda.memory_allocated()))
    print(f"  loaded in {report['load_seconds']} s: {report['params_b']}B parameters, {report['weights_gpu_mb']} MB", flush=True)
    world = ROOT / args.notes_world
    if "notes" in tasks:
        report["notes"] = run("notes", lambda: task_notes(lm, label, world, args.notes_limit))
    if "yesno" in tasks:
        report["yesno"] = [run(f"yesno {w}", lambda w=w: task_yesno(lm, w)) for w in ("val-001", "val-002")]
    if "lora" in tasks:
        report["lora"] = run("lora", lambda: task_lora(lm, world))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.json"
    if path.exists():   # re-running one task updates that task and keeps the others' results
        old = json.loads(path.read_text())
        old.update({k: v for k, v in report.items()})
        report = old
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
