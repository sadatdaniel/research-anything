# STATUS: where Learn Anything stands

> **Read [OBJECTIVE.md](OBJECTIVE.md) first (the why and what we've learned), then this file.** It's updated at every milestone. The full design and success criteria live in [PLAN.md](PLAN.md); this file says what's done, what's in progress, what was decided, and what comes next.

_Last updated: 2026-11-23 (Phase 3: first test evaluation done; recipe default@2 being compared on fresh test worlds; question bank; app restructured around memories)_

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
| 3 | Dream mode and verification; first app window | 🔄 first test evaluation (default@1, 6 untouched worlds): recall 54%, precision 56%, 0 decoys, predictions 14/14. default@2 vs default@1 running on test-007…009 | below; `results/phase3/` |
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
- **Development hygiene:** prompt and heuristic work uses `val` worlds; reported numbers come from test worlds never tuned on (`test-001` is spent, `test-002` was used for the Phase 2 final). Since then: test-000 and 002–006 were the first Phase 3 evaluation, and test-007…009 the recipe comparison. **test-010…039 (`test-v2`, 122 patterns) are the evaluation set for every reported comparison from now on** (PLAN.md, Evaluation tracks).
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
- **App** (`app/`), opened read-only over the brain. What a person learns from is called a **memory** on screen (your choice, 2026-11-23; the code and API still say `skill`). Where things live:
  - **Sidebar:** Library, Settings, About. Nothing else.
  - **Library:** every memory as a card or a list row. New memory…, Import… (mindpack, with preview and diff), and per memory: Open, Export (.mindpack), Findings for Obsidian, Remove… (moves it to the library's `.trash/` folder; refused while the engine works on it). Tick two or more to Combine (a fusion), Compare, or Ask across them.
  - **Inside a memory** (‹ goes back): **Overview** (number cards, strongest findings, start here, open questions), **Explore** (Timeline, Graph, List: the old Connections), **Questions** (Its questions: the question bank; Ask your own: "is A connected to B?" and word search, across ticked memories), **Compare**, **Journal**.
  - On narrow screens the five sections move from the top to the bottom bar (never both), the sidebar becomes a drawer, and every card grid reflows.
  - **Preferences** is a fixed-size GNOME-style window (Appearance, Learning, Machine, Recipes, Privacy, Storage). Themes are one block of variables each in `app/web/themes.css`.
  - **Adwaita colours follow libadwaita's published values:** surfaces exactly, and text uses its "standalone" colours, so every coloured label is at least 4.4:1 (before: 2.5–3.8:1 for the yellow, green, blue and faint text). There's a high-contrast variant for GNOME's High Contrast setting (`prefers-contrast: more`).
  - Not-yet-working parts carry a "preview" tag.

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

**First test evaluation (default@1; untouched test-000 and 002–006; `results/phase3/summary_test-final.json`):**
- Recall 14/26 (54%); precision 56%; 0 decoys believed; predictions 14/14.
- By type: periodic 8/10, cause→effect 4/10, chain 2/6.
- Baselines: B1 (every proposal, untested) 65% recall at 4.6% precision; B0 (ask the model directly) 0%; B2 (keyword script) 0%.
- Recall against dream time saturates by 60 s, so the "8 h vs 1 h" criterion can't be tested on worlds this size, and precision is below the 70% target.
- False claims were 7 chain pieces, 2 chance cycles, 1 harmonic and 1 junk chain.

**default@2** fixes these: chains assembled from two consistent links, a family-wise shuffle threshold, and consistent candidates kept over broader ones. It's being compared with default@1 on three fresh test worlds (test-007…009) by `experiments/phase3_v2.sh` (log: `results/runs/phase3/v2.log`).

**Question bank** (`learner/questions.py`; API `/api/questions` and `/api/connection`; CLI `python -m learner.questions BRAIN [--ask A B]`):
- Built from the theories each time: answered; needs material (gaps in a cycle, a cause without its effect, a return due after the last record); being tested; ruled out; consequence; and across findings (many paths to one outcome, a place in several findings, findings across memories).
- "Is A connected to B?" answers, best first: a finding, a chain of findings (certainties multiply), a shared cause or effect, still testing, tested and failed; otherwise "I don't know", with untested co-occurrence.

**Is the score calibrated?** (`experiments/phase3_calibration.py`, on the first test runs): **no, not yet.** It ranks well, but it isn't a probability. Findings scored 80–100 were true 57% of the time; 40–60, 5%. ECE 0.225, Brier 0.078. The app calls it a score. A calibration stage (isotonic or Platt, fitted on validation worlds only) is planned; see REFERENCES.md.

**Phase 3 quality check** ([results/phase3/QA.md](results/phase3/QA.md)). A UI audit (`app/web/qa/ui_audit.js`) covers 31 views at 5 widths, light and dark. It found:
- the page's fallback variables were silently overriding the whole Adwaita theme (grey cards, white entries);
- the library's card grid was nested inside another grid (cards stacked, the fusion card stretched);
- the Ask entries were unstyled;
- 379 contrast failures;
- phone toggles were too small.

All are fixed, and the audit now reports 0 issues. A regression test guards the theme bug. Ask your own was redesigned around one composer. **Every phase now ends with this check** (PLAN.md).

**Recipe comparison on fresh test worlds** (test-007…009): default@1 found 9/12 with 9/11 claims true; default@2 found 8/12 with 8/10 true; 0 decoys for both. That's a tie: 12 patterns can't separate recipes, so the evaluation set needs to grow.

**Model bake-off** ([results/bakeoff/README.md](results/bakeoff/README.md)). Seven candidates were tested on validation worlds.
- **Ouro-1.4B at 4 loops is still the best checker that fits:** 1% false yes, the fewest figures of speech fooling it, 94% of links found.
- **At 2 loops it gets clearly worse,** which is evidence for H3.
- **Ouro-2.6B is an even better checker,** but it doesn't fit 6 GB for reading or training.
- **Qwen3.5-2B-Base reads about 6× faster** (98% events, 88% links), but it's a poor checker (falls for all 22 traps).
- **End to end** (val-003 and val-004, 10 patterns; whole worlds read and dreamt):
  - Qwen3-1.7B found 7 (7/8 claims true) in 44 minutes; Ouro found 6 (6/7) in 67; Qwen3.5 found 5 (5/5) in 43.
  - That's a tie at this size. The careful checker didn't turn into more findings, and the fast readers dream longer because their labels are less consistent.
  - The split design (fast reader, Ouro checking) did worst (4 of 10): the checker re-sorted a locust record into the reader's vocabulary and broke a cycle. So: one model per memory, and a robustness fix for the cycle finder.
- **LFM3-1.7B** (added at your request) can't follow the shared note format: 0 of 10 found, 41% false yes.
- **Next:** the deciding comparison on test-v2, Ouro against Qwen3-1.7B. **Running since 2026-11-23 23:2x** (`experiments/eval_v2.sh`, log `results/runs/eval_v2/log.txt`; report `experiments/eval_v2_report.py`). Qwen3 reads and dreams first (about 10 h), then Ouro (about 14 h). It resumes from where it stopped if interrupted.
- The 8B LittleBit model at 0.55 bits wasn't tested: its own card reports ARC-Challenge at chance, and it's CC BY-NC.

**Next (in order):**
1. When `phase3_v2.sh` finishes: compare default@1 and default@2 on test-007…009 (recall, precision, curves, calibration).
2. Try BH-FDR across candidates as `default@3` (REFERENCES.md, "To try").
3. Write `results/phase3/README.md` (the honest Phase 3 verdict) and decide GO or NO-GO for Phase 4.

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
  - Phase 3 first test evaluation on 6 untouched worlds: recall 54%, precision 56%, 0 decoys. default@2 written; clean comparison on test-007…009 started.
  - Question bank and "is A connected to B?" (answers only from findings; otherwise "I don't know"). Calibration check: the score ranks but isn't a probability yet (ECE 0.225). REFERENCES.md added.
  - App: "skills" became **memories** on screen; sidebar has Library, Settings, About only; each memory has five sections (Overview, Explore, Questions, Compare, Journal), at the top on wide screens and at the bottom on narrow ones; cards reflow; Remove moves to the trash; libadwaita's text colours and a high-contrast mode.
  - Phase 3 quality check (results/phase3/QA.md): the UI audit found the theme override bug, nested library grids, unstyled entries, 379 contrast failures and small phone toggles; all fixed (0 issues at 5 widths, light and dark). Ask your own redesigned around one composer. A quality check now ends every phase.
  - Recipe comparison on test-007…009: a tie (9/12 and 9/11 vs 8/12 and 8/10). Model bake-off: Ouro-1.4B at 4 loops is still the best checker; Qwen3.5-2B the fastest good reader; models are now swappable (`--model`). End-to-end comparison running.
  - Bigger evaluation set: 30 new test worlds (test-010…039, 122 patterns, 8,957 records). The generator gained `--start`, which never overwrites a world. Checked: the generator reproduces old worlds byte for byte, and the keyword baseline finds 11.5%. Not read yet: that waits for the model verdict, since the reader decides whether it takes 9 hours or 1.5.
  - End-to-end model test: Qwen3-1.7B 7/10, Ouro 6/10, Qwen3.5 5/10, split designs 4 and 1, LFM3 0. A tie at this size; fast readers are about 1.5× faster overall. Verdict in results/bakeoff/README.md; the deciding comparison goes on test-v2.
  - Cycle diagnosis (results/diagnosis/README.md, error analysis on Qwen3's test-v2 run; nothing tuned). Of 27 missed cycles: 11 were comets rejected by the chance test because they hide in a crowded kind (fix to try: group by description, the Halley idea); 6 are unfindable by the shape rule; 5 failed the prediction test because the predicted return was never written down (fix to try: silence counts as untested); 5 were reading splits. Fixes go to validation worlds first, then fresh test worlds.
