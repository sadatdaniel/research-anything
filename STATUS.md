# STATUS: where Sandbox Learner stands

> **Read this first when picking the work back up.** It's updated at every milestone. The full design and success criteria live in [PLAN.md](PLAN.md); this file says what's done, what's in progress, what was decided, and what comes next.

_Last updated: 2026-11-23 (Phase 3 built; mindpacks and recipes; test-world runs pending)_

## In one paragraph
A small looped language model ([Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)) that lives in a local sandbox and keeps learning from a folder of documents:
- **By day** it reads and takes notes.
- **At night** it "dreams": it hunts for connections between distant records and checks each one against the source, against held-back records, and against shuffled dates.
- **In deep sleep** it trains its confirmed findings into a LoRA adapter.

A resource-aware cycle engine decides when it may use the GPU. The core claim to prove (H2) is that it **learns how to learn**: after sleeping, it finds connections faster in material it has never seen. All scoring uses generated fake chronicles with planted patterns, because the model already knows real history.

## Phase status

| Phase | What | Status | Evidence |
|---|---|---|---|
| 0 | Setup and feasibility | ✅ GO | [results/phase0/README.md](results/phase0/README.md) |
| 1 | Fake-chronicle generator, scorer, keyword baseline (B2) | ✅ GO: B2 9–12% (< 30%); blind readers found 7/7 patterns, 0 false, 0 decoys | [results/phase1/README.md](results/phase1/README.md) |
| 1b | Resource-aware cycle engine | ✅ GPU handed back in 0.9 s | [results/phase0/README.md](results/phase0/README.md#cycle-engine-phase-1b-real-gpu-check) |
| 2 | Notebook and focused reading (day mode) | ✅ GO: untouched test world: events 95%, places 98% correct, kept links 100% true; engine end-to-end passes | [results/phase2/README.md](results/phase2/README.md) |
| 3 | Dream mode and verification; first app window | 🔄 built; val worlds 75–80% recall at 75–100% precision; test-world runs and H1 report pending | below; `results/phase3/` |
| 4 | Sleep (LoRA consolidation, review, memory audit); H2 and H4, with baseline B4 and the H2 ablations | not started | n/a |
| 5 | Depth vs breadth (loops vs hours); H3 | not started | n/a |
| 6 | Disguised real comet test; ask interface | not started | n/a |
| 7 | Public release: AppImage and Windows zip | not started | n/a |
| 8 | Android companion | later | n/a |

## Key numbers so far
- **Ouro-1.4B on the RTX 4050 (6 GB):**
  - 2.7 GB in GPU memory;
  - 13.9 tokens/s at 4 loops, batch 1 (7.1 at 8 loops), and **41.7 tokens/s at batch 4**;
  - LoRA training fits: 4.1 GB at 512 tokens, 5.0 GB at 1024.
- **Cycle engine:** hands the GPU back in **0.9 s** when another program grabs it, and is back at work 3.1 s after it closes.
- **Keyword baseline B2:** **9%** recall on the test split, 6% on transfer, with 3–4% precision (16% and 10% before the location mentions).
- **Blind readers** (fresh agents that saw only the records): A, on the 41-record sample, scored 3/3; B, on a 218-record world, scored 4/4. No false claims, no decoys; predictions 6/7.
- **Note-taking at 4 loops vs 2** (14 records): 4 loops is far better (name links 5/5 vs 3/5, map links 3/3 vs 1/3; 2 loops invents details) but slower (4.6 vs 1.75 s/record). **Day reading uses 4 loops.**

## Decisions and gotchas (don't re-learn these)
- **transformers 4.57.1, not the 4.54.1 the model card recommends.** Ouro's bundled code now needs ≥ 4.56 and fails on 4.54.1 with "key_cache has no setter".
- **Two Ouro bugs are fixed at load time in [learner/model.py](learner/model.py)** (the model files stay untouched):
  1. The cache never fills `layers`, so transformers built a decode mask covering only the newest token, and padded batches produced garbage. We patch `get_mask_sizes`.
  2. PyTorch's fast attention kernel rejects the head-broadcast mask, so we expand it to a contiguous tensor.
- **Loop count** is set with `LoadedModel.set_loops(n)`, which changes both `config.total_ut_steps` and `model.total_ut_steps`. Ouro always runs every loop; its early-exit gate only picks which loop's output to use.
- **The model runs in a separate worker process.** PyTorch keeps its CUDA context until the process exits, so REST means ending the worker. The worker dies with the engine (`PR_SET_PDEATHSIG` on Linux, plus a parent check).
- **Exactly-once work:** an item is computed outside any transaction, and its results are written and the item marked done in one short transaction. A `kill -9` therefore never loses or repeats work.
- **STARS has no license file**, so we write our own regularizer from the paper. The 730M looped fallback (`ifm-ai/xllm-loop`) is Apache-2.0.
- **Engine GPU-memory budget** (measured): 4.0 GB to read or dream, 5.2 GB to sleep.
- **Night rule:** unread material comes first even at night (you can't dream about what you haven't read), and no dream work means REST. Hard stops and the minimum time in a state still apply.
- **Day reading uses 4 loops.** 2 loops is 2.6× faster but invents details and misses links.
- **Only *stated* links are facts** (`links.stated = 1`, from `learner/sourcecheck.py`). Inferred links are hypotheses for dream mode to test.
- **Development hygiene:** prompt and heuristic work uses `val` worlds; reported numbers come from test worlds never tuned on (`test-001` is spent, `test-002` was used for the Phase 2 final).
- **Looped models have a 4× key/value cache:** Ouro stores one entry per layer *per loop*, about 0.8 MB per token per prompt at 4 loops. A 700-token few-shot prompt therefore allows a batch of only about 2–3. `LoadedModel.safe_batch()` sizes batches from free memory, and `read_batch` splits a batch in half when it runs out of memory. Retry **outside** the `except` block, because the traceback keeps the failed attempt's tensors alive.
- **What's new and what isn't** (see [RELATED_WORK.md](RELATED_WORK.md), checked 2026-11-23):
  - LLM "sleep" with consolidation and dreaming is published (Google 2026, Auto-Dreamer, Lee et al.), and so is "learning how to learn" from self-written training data (SEAL). We cite these; we don't claim them.
  - H2 is still open only in its narrow form: sleeping on *checked* findings and *search outcomes* makes a small model faster at discovering unstated patterns in new material.
  - To claim it, Phase 4 must beat **B4** (the same lessons given as written notes in the prompt, no weight update) and must show that sleeping on search outcomes beats sleeping on facts only.
- **Dream-mode calibration** (Phase 3):
  - The label question "same kind?" only screens; merges need a majority both ways from the record question.
  - Source checks only reject on a clear no (< 0.25).
  - A move needs ≥ 0.8 and the model's preference over the record's own kind.
- **`executescript()` commits any open transaction.** Create tables *before* `BEGIN` (the `ensure_schema` functions skip it when the tables exist).
- **Periodic prediction tolerance** is `max(2 × jitter, 5% of the period)`. Each occurrence is shifted by up to ±jitter, so a correct "last + period" can be off by twice the jitter. Found by blind reader A.

## Phase 3: where it stands (dream mode and verification)

**Built and tested (69 tests pass):**
- `learner/knowledge.py`: builds the index each time it's needed.
  - Event kinds come from the notes, plus the model's re-reading answers.
  - Names are capitalised words in the source that never appear in lower case.
  - Aliases come only from *stated* `same_as` links (union-find); the map from stated `in` / `on the` links.
  - **Merges are decided by majority vote in both directions** ("are these 'hunger' records about famine, and these 'famine' records about hunger?"). One-way likeness, as in a sea storm being flood-like, never merges.
  - A record moves to another kind only on a confident repair answer that the model prefers to its own kind. It leaves a kind only on a clear "no" (below 0.25).
- `learner/patterns.py`: loose proposers for periodic, cause_effect, chain and identity. Periodic fits are ranked by net hits, so half the true period can't win.
- `learner/verify.py`: the model-free tests.
  - **prediction:** periodic is refit on the learner's own earlier 85% of years and must predict its latest 15%; for cause_effect and chains, at least 75% of causes must be followed by their effect;
  - **chance:** periodic uses a shuffle test (at most 2% of shuffles may fit as well); lagged patterns use a binomial test (p ≤ 1e-4);
  - **shape:** never two returns missed in a row;
  - **lifetime** (identity) and **verbatim quotes**. Evidence is the record's main sentence.
- `learner/theories.py`: stages (proposed → prediction-tested → confirmed / rejected / superseded) with full history stamped with dream seconds. It keeps a snapshot of each claim when confirmed, so H1 curves are exact. A **confidence score from 0 to 100** combines chance, predictions, support and source check (a summary, not a probability).
- `learner/dream.py`: `DreamWorkload`. Each plan rebuilds knowledge, re-runs the search and tests, syncs theories, then queues batches of yes/no questions in this order:
  1. source checks;
  2. "own kind" questions, after any confident move;
  3. repairs, near missing returns or effects, for candidates that aren't chance;
  4. merge samples, 3 records per side, for screened pairs;
  5. cheap label screens.

  A chain's own cause→effect links are marked "a consequence of chain N". `LearnerWorkload` combines day and dream with one shared model; the engine CLI calls it `--workload learner`.
- `learner/model.py`: `p_yes()` is one forward pass comparing yes and no logits (no generation, no KV cache).
- **App** (`app/`), opened read-only over the brain:
  - **Library** of skills: every brain in a folder; select several to combine (preview), compare, or ask across them.
  - **Status bar.**
  - **Skill view:** Overview (big numbers, strongest findings, start-here questions, open questions, "nothing new for N minutes"), Explore (Timeline and an **Obsidian-style Graph** with filters, search, zoom and backlinks), Connections (confidence score with parts, tests, evidence), Ask (search across the loaded skills until Phase 6), Compare (themes only here, in both, only there), and Journal.
  - **Settings** (Learning, Machine limits, Appearance, Recipes, Updates and privacy, Storage, About) are saved to `config.toml`; the engine reads `[engine]`.
  - **Themes** are one block of variables each in `app/web/themes.css`, seven so far: Instrument, Swiss, Brutalist, Terminal, Blueprint, Mono, Synthwave. The picker previews each in its own colours.
  - Not-yet-working parts carry a "preview" tag.
  - **Obsidian export** (`python -m app.export_obsidian`, or the zip button) writes linked Markdown notes for every connection, record, kind and place.

**Also built after the first Phase 3 commit:**
- **Mindpacks** (`learner/pack.py`, `.mindpack`): export, verify, preview and diff, and import.
  - Data only, with a sha256 for every file.
  - Refused: path tricks, zip bombs, unknown formats and unknown recipes.
  - Import builds the new brain in a temporary file and swaps it in with one rename. The previous version is kept under `.versions/`.
  - A pack from a different base model imports read-only.
  - CLI: `python -m learner.pack export|preview|import`.
- **Recipes** (`learner/recipe.py`): a registry of pipeline stages and implementations, plus the `default`, `strict` and `lenient` recipes.
  - All dream thresholds now come from the active recipe.
  - Each brain keeps its own recipe (`engine_kv.recipe`), and every pack records the one it was learned with.
- **App:**
  - **Library** in three views: Cards, Map, Table.
    - Cards: time strip, confirmed count, average score, discovery sparkline, top findings, shared themes and the verdict.
    - Map: skills and the themes they share.
    - Table: sortable.
  - Real **Export mindpack** and **Import mindpack**, with a preview and a diff before anything is written.
  - Server library mode: `--library DIR --worlds worlds`, where each brain is a skill.
- **PLAN.md** gained the mindpack format, git upstreams for packs, the swappable-pipeline table (papers to try at each stage), and the **rediscovery track**: Halley's comet, eclipse cycles, Semmelweis and John Snow from real records, disguised.

**Dev results on val worlds** (real Ouro, notes from day reading):

| | val-000 | val-001 (never tuned on) |
|---|---|---|
| Planted patterns found (confirmed) | 3/4 | 4/5 |
| Confirmed claims that are true | 3/4 (the 4th is the two ends of the planted chain) | 4/4 |
| Decoys believed | 0 | 0 |
| Predictions right | 2/2 | 3/3 |
| B1 (every proposal, untested) | 75% recall, 13% precision | 100% recall, 6% precision |
| Dream time until it ran out of work | 4.9 min | 6.5 min |

Misses come from the reader: two white locusts written as "the winged plague, snowy" (the model gives 0.41 for "locusts"), and a chain step split between the labels "flood" and "dikes broken".

**What the model's yes/no answers are worth** (`results/phase3/prompts_*.json`, graded against the answer key):
- "Is this record about X?" says yes to 99–100% of true cases and 1–2% of false ones at 0.5, after adding one figurative example to the prompt.
- "Are these two labels the same kind?" is poorly calibrated: at a usable threshold it would make about 50 false merges per world. So it only *chooses which pairs to re-read*; merges are decided by the record question.

**Next (in order):**
1. Day-read test-000 and test-003…006. This is running in the background; it's resumable (`experiments/phase3_read_worlds.py`).
2. Dream on test-000, 002, 003, 004, 005 and 006 (`experiments/phase3_dream.py <world> --tag final`).
3. B0 (`experiments/phase3_b0.py`), then `experiments/phase3_report.py --tag final <worlds>`, giving the H1 curves against B0, B1 and B2.
4. Engine end to end with the full learner (`experiments/phase3_engine_e2e.py`).
5. Write `results/phase3/README.md` and decide GO or NO-GO.

**Honest note for H1:** on worlds this size, dreaming runs out of useful comparisons within minutes (all findings by about 1–4 min, then merge checks that add little). "Recall at 8 h ≥ recall at 1 h + 20 points" can't be tested as written on 200–400-record worlds. The report should show recall against dream time and say so. Bigger worlds, or real books, are where time should matter.

**Ideas you added during Phase 3 (now in PLAN.md):**
- skills (export, import, versions, "needs N nights", combining skills, Obsidian folder, recipes, a graph explorer);
- answering only from loaded skills, with "I don't know";
- comparing sleep strategies from the papers in Phase 4;
- deeper patterns (rule flips, impossible timelines, unreliable sources, open questions);
- a generic pattern grammar with a "mystery" planted type;
- evaluation on public benchmarks and real Gutenberg books (planted and unplanted);
- confidence scores and consequences.

## Open questions for you
- none right now

## Git
- Commits are made as `sadatdaniel <sadatdaniel@gmail.com>`, set in this repository only. **Never add a Claude co-author line.**

## How to pick up
```bash
cd ~/Documents/claude_projects/sandbox-learner
uv sync                                    # Python environment (PyTorch cu128, transformers 4.57.1)
uvx --from "huggingface_hub[cli]>=0.34,<1.0" hf download ByteDance/Ouro-1.4B --local-dir learner-data/models/Ouro-1.4B   # if missing
CUDA_VISIBLE_DEVICES="" uv run pytest -q chronicles/tests engine/tests learner/tests app/tests   # all 74 tests (~2.5 min)
uv run python -m app.server --library results/runs/phase3/library --worlds worlds          # the app, on http://127.0.0.1:8765
uv run python -m chronicles.generator splits --out worlds           # regenerate the 45 worlds (gitignored)
uv run python -m chronicles.baseline_b2 --split worlds/test         # baseline score
uv run python -m engine.check_yield                                 # real-GPU yield check
uv run python experiments/phase0_feasibility.py                     # model speed, memory and LoRA fit
```

## Layout
| Folder | What |
|---|---|
| `chronicles/` | world generator, phrasing banks, scorer, keyword baseline B2, sample world (`samples/small_world.md`) |
| `engine/` | cycle engine: sensors, policy (day/dream/deep sleep/rest), SQLite store, worker process, scheduler, CLI |
| `learner/` | model (Ouro + fixes, `p_yes`), notebook, focused (day reading), knowledge, patterns, verify, theories, dream; sleep to come |
| `app/` | local server (read-only API + settings), one-page app (`web/index.html`), themes (`web/themes.css`), Obsidian export |
| `experiments/` | measurement scripts |
| `results/` | phase reports and raw numbers |
| `learner-data/` (gitignored) | models; later the learner's brain (notebook, adapters, history) |
| `worlds/` (gitignored) | generated worlds |

## Log
- **2026-11-23:**
  - Plan approved. Phase 0 GO; cycle engine built (yield 0.9 s).
  - Chronicle generator, scorer and B2 built; B2 started at 33%, and after widening the phrasing it's at 16%.
  - Blind reader A scored 3/3, which exposed the scorer's prediction-tolerance bug (fixed) and the geography gap (in progress).
  - Git repository initialized.
  - Blind reader B scored 4/4, which exposed the "two drops in a row" flaw (fixed). Phase 1 GO.
  - Phase 2 notebook, reader and grader built; 4 loops beats 2 for note-taking.
  - First commit `c1b4262` (as sadatdaniel, no co-author).
  - Full note-taking run on `test-001`; links source-checked; night policy fixed (read first, rest when there's nothing to dream about); prompt sharpened.
  - End-to-end engine run passes (GPU freed in 1.1 s, 41/41 notes, rests when done). Phase 2 GO on untouched `test-002`.
  - Related-work check ([RELATED_WORK.md](RELATED_WORK.md)): the sleep framing is published; the narrow H2 test isn't, as far as the check went. Added baseline B4 (written notes instead of sleep) and the H2 ablations to PLAN.md.
  - Phase 3 built: knowledge index, proposers, tests, theory stages with confidence, dream workload, Obsidian export, and the app (library, overview, explore with timeline and graph, connections, ask-as-search, compare, journal, settings, seven themes). val-000: 3/4 found; val-001: 4/5, all claims true. Test-world runs pending.
