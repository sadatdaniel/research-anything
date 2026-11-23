# Sandbox Learner: a small looped model that learns how to learn

## Context

We want a small model that runs in a local sandbox and keeps learning from whatever it's given. It should find connections in the material that nobody spelled out, and explain them with evidence when asked. The target example is Halley's comet: records spread over centuries describe "the same thing" about every 76 years.

The design combines three ideas from the research we reviewed:
- **Learning-science study habits:** focused reading, self-quizzing, a background "diffuse" mode that hunts for links, sleep consolidation and spaced repetition.
- **A looped language model, [Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B).** It's small enough for the laptop and can think longer by running its layers more times.
- **[STARS](https://github.com/njuyxw/STARS),** which keeps a looped model stable when it thinks longer than it was trained to.

End goal: a tool anyone can install, point at their own content, leave running, and then question. It's built to be shared publicly from day one.

The closest project in spirit is [Omi](https://omi.ekayan.tech/), an on-device Android assistant that "thinks while the phone rests". Omi is a personal assistant: it surfaces what matters and does jobs across apps. This project is about discovery: it finds connections nobody stated, checks them, and trains them into the model's own weights.

Hardware: RTX 4050 laptop (6 GB VRAM), 14 GB RAM, 49 GB free disk, Python 3.12, `uv`. There's no code yet, so this is a new project.

## What we are trying to prove

Four hypotheses, in priority order:

| # | Hypothesis | In plain words |
|---|---|---|
| **H1** | **Time produces discoveries** | Given a fixed set of material, the number of *true, unstated* connections found keeps rising with running time, while false claims stay low. |
| **H2** | **It learns how to learn** | After sleeping on its own confirmed findings, it finds connections **faster in brand-new material** it has never seen. The skill transfers, not just the facts. *This is the core claim.* |
| H3 | Thinking deeper helps | More loops per comparison find more true connections, with STARS keeping deep loops stable. We measure the trade between depth (loops) and breadth (hours of searching). |
| H4 | It remembers without forgetting | After sleep it can answer about what it learned without opening its notebook, and its general ability hasn't dropped. |

**Not trying to prove:** that it beats big models, or that it discovers new things about real history. Real history is a stretch goal only.

## The test: fake chronicles with planted patterns

Real history is useless for scoring, because the model already knows about Halley's comet. Instead we generate made-up worlds where we know every hidden pattern.

- Each world has 200–400 short records spread over about 1,000 years. The records come from several invented chroniclers who write in different styles, so the same kind of event is described in different words.
- **Planted patterns:**
  1. **Periodic return.** The same kind of event every *P* years, give or take a few. This is the Halley type.
  2. **Cause, then effect after a delay.** For example, a flood followed by famine two years later.
  3. **Hidden identity.** The same person appears under different names or titles.
  4. **Chains.** A leads to B, which leads to C.
- **Decoys:** random noise events, plus planted coincidences that look like patterns but predict nothing (Lincoln–Kennedy style).
- Every world uses fresh invented names, so the model can't have seen any connection before.
- The last 20% of each world's timeline is held back to test predictions.
- **Splits:** training worlds (used for sleep), validation worlds, and test worlds that are never trained on. There's also a *transfer* test set containing a pattern type that never appeared in the training worlds.

## How the system works

```
 DAY: records ─► FOCUSED (few loops): extract each record into notebook entries, quiz itself
                    │
                    ▼
            NOTEBOOK (SQLite): events and claimed links, each with source record ids
              ▲          │
 NIGHT, DREAM (many loops): pick candidate groups of distant entries,
   propose a pattern, then TEST it:
     (a) go back to the source: do its quotes exist word for word in the files, and do they support it?
     (b) does it predict the held-back records?
     (c) does it disappear when the dates are shuffled?
   keep only the patterns that pass, each with a confidence score
                    │
 NIGHT, DEEP SLEEP (after each dream): LoRA training on
     (1) confirmed findings
     (2) its own successful reasoning
     (3) which kinds of comparisons paid off   ← this is the "learning how to learn"
   with spaced-repetition review and the STARS regularizer
                    │
 ASK: "from year X to Y, any patterns?" → patterns + record ids + predictions
```

The model can't check every pair in a world (about 20k–80k pairs), so it has to learn **which comparisons are worth making**. That is where H2 shows up.

## Checking its own theories

Like a student checking their notes against the textbook, the learner never trusts its own notes or memory alone. It always goes back to the original material.

**Every theory moves through these stages:**

`proposed → source-checked → prediction-tested → confirmed`

At any stage it can instead become `rejected (reason)`.

1. **Source check.**
   - It re-opens the original files, not the notebook summary.
   - *Mechanical part:* every quote used as evidence must exist **word for word** in the file, at the location given. This is a plain string match: cheap, and 100% certain.
   - *Reading part:* with the original passages in front of it, the model judges whether they really support the claim. This catches mistakes made while taking notes and invented notebook entries, before they pile up.
2. **Prediction test:** it must predict held-back records it hasn't looked at.
3. **Chance test:** the pattern must disappear when the dates are shuffled.

**Memory audit after every deep sleep**
- Sleep changes the weights, so its memory can drift.
- After each deep sleep it quizzes itself without the notebook on a sample of learned facts, and grades the answers against the source files.
- Wrong answers are flagged, queued for spaced-repetition review, and corrected in the next sleep.
- The memory error rate is logged after every night, which gives us the H4 measurement.

Rejected theories are kept with their reason, so it doesn't propose the same wrong idea again.

## Cycle engine (resource-aware)

From moment to moment, the engine decides what the learner does based on what the machine can spare. Think of a person who works when rested, sleeps when tired, and leaves when someone else needs the room.

| State | When | GPU load | What happens |
|---|---|---|---|
| DAY: focused | machine in use but has headroom | light (inference, few loops) | read new material, take notes, quiz itself |
| NIGHT: dream | machine idle, on AC power, inside night hours | medium to high (inference, many loops) | hunt for connections between distant notes and test them |
| NIGHT: deep sleep | after each dream stage | highest (training) | train confirmed findings into the LoRA adapter |
| REST | you need the machine, or a hard stop | none (model unloaded, VRAM freed) | wait; the computer belongs to you |

- **A night is several cycles.** Each one is dream (find links) → deep sleep (train them in), like the 4–6 sleep cycles in a human night. Night hours default to 23:00–07:00 and can be changed.
- **Naps.** If the machine sits idle on AC power during the day, it can run a short cycle. When the system can handle it, it learns longer.
- **Morning report.** It ends the night by writing a short summary of what it found, with evidence, so it's ready when you look.

This split matches the 2026 sleep research: Google's framework has consolidation and dreaming stages, and SCM's "REM dreaming" step generates new connections. It also matches [Omi](https://omi.ekayan.tech/)'s "it thinks while the phone rests" approach. The sleep framing isn't ours: see [RELATED_WORK.md](RELATED_WORK.md) for what is and isn't already published.

Two drives decide when to switch. They're borrowed from Borbély's two-process model of human sleep:
- **Sleep pressure** rises with every confirmed finding that hasn't been consolidated yet, and falls during sleep. When pressure is high, the learner goes to sleep as soon as the machine can afford training.
- **Machine budget** is what the hardware can spare right now: free VRAM, GPU temperature, other programs using the GPU, free RAM, AC power versus battery, and the hours you allow.

Rules:
- **Hard stops send it to REST immediately.** It finishes or abandons the current step, saves, and unloads the model. Triggers:
  - another program needs the GPU;
  - GPU temperature reaches 83 °C (it resumes below 75 °C);
  - free RAM drops below 2 GB;
  - the laptop is on battery (unless allowed);
  - it's outside the allowed hours.
- **The budget scales the work.** With plenty of headroom it runs longer awake sessions, more loops and bigger batches. When things are tight it runs shorter sessions with fewer loops.
- **It sleeps only when the budget is high,** because training needs the most memory. Otherwise pressure keeps building while it stays awake or rests.
- **It doesn't flip back and forth.** Switching uses a gap between the start and stop thresholds, and each state has a minimum time, because loading the model costs about 10–30 s.
- **Every step ends at a checkpoint:**
  - a SQLite transaction;
  - a LoRA checkpoint;
  - the saved work queue.

  It can be paused or killed at any moment and picks up where it left off.
- **It keeps a timeline log** of each state, its duration and the reason (for example, "REST 14:02–14:40: another program using 4.1 GB VRAM"). `status` shows this.

**Sensors:**
- `nvidia-ml-py` (NVML) for GPU memory, load, temperature and other GPU processes;
- `psutil` for CPU, RAM and battery.

Keyboard and mouse idle detection is optional. It may not be available on Wayland desktops such as COSMIC, so GPU and CPU signals are the main inputs. Every threshold lives in the user's config file.

*Optional experiment (H5):* for the same compute, does sleeping when pressure is high beat sleeping on a fixed timer?

## Evaluation tracks (the real content will mostly be ordinary books)

1. **Fake chronicles** (now): exact answer keys, for building and debugging.
2. **Public pattern benchmarks:** checks against tests we didn't write. Candidates (license checked before use):
   - HypoBench's synthetic hypothesis tasks;
   - temporal-relation sets that use Allen's interval relations;
   - contradiction sets from natural-language inference.
3. **Ordinary books, the real target:** public-domain books from Project Gutenberg, mostly undated prose.
   - **Planted:** insert known changes into a real book and keep the answer key. Examples:
     - a contradiction;
     - a character in two far-apart places at once;
     - a misattributed quote;
     - an unreliable chapter;
     - a theme that recurs every N chapters.
   - **Unplanted:**
     - compare two authors, or two translations of the same book;
     - ask questions the book does and doesn't answer, to test "I don't know".
   - Undated prose has no year on every line, so the checks use the book's own order (chapters, pages) as its timeline, and say so.
4. **Rediscovery: real records where humans really found the pattern, so the answer is known.**
   - Each set is disguised before the learner sees it: names replaced and years shifted by a secret offset, so the model can't just recall the answer.
   - Licences are checked before any data is used or redistributed.

   | Discovery | Data (to verify) | What the learner must find | Needs |
   |---|---|---|---|
   | Halley's comet | Historical comet records: Williams' list of 372 comets from 613 BC to 1621 AD; Chinese records from 240 BC; NASA's 1986 report on historical apparitions | The ~76-year return among hundreds of unrelated comets | periodic (exists) |
   | Eclipse cycles (the Babylonian Saros) | NASA Five Millennium catalog, 11,898 solar eclipses (licence unconfirmed; a Hugging Face copy says CC-BY-4.0) | The ~18-year recurrence in "an eclipse was seen at X" records | periodic (exists) |
   | Semmelweis and handwashing | Monthly deaths by clinic, 1840s (published 1861) | One clinic differs, and its deaths fall after a change in 1847 | change point; contrast between groups (new) |
   | John Snow's cholera map | Deaths by street and pump, 1854 | Deaths cluster around one pump | place cluster (new) |
   | Later | Graunt's Bills of Mortality, sunspot counts, cicada broods | Seasonal and periodic patterns | data and licences to check |

   This track is the strongest evidence the project can produce: rediscovering Halley from disguised real records is the original goal.

## Baselines
- **B0, just ask:** Ouro reads the world once and is asked directly. No notebook, no search.
- **B1, search without testing:** diffuse mode keeps every proposed pattern. This shows what verification adds.
- **B2, a dumb script:** keyword clustering plus a period finder, no ML. If it solves the worlds, the test is too easy and must be made harder.
- *(Optional)* **B3, non-looped model:** the same pipeline with a normal small model (Qwen3-1.7B). This shows what looping adds.
- **B4, written notes instead of sleep** (added after the [related-work check](RELATED_WORK.md)). Before a new world, the learner gets a written list in its prompt of the kinds of comparisons that paid off on the training worlds, with no weight update. This is how HypoForge, ReasoningBank and Auto-Dreamer carry skill forward. If B4 does as well as sleep, the speed-up doesn't need the weights.
- **H2 ablations:**
  - sleep on facts only vs facts plus search outcomes (which comparisons paid off);
  - sleep on unchecked findings (B1's output) vs checked ones.

## What counts as success

All results are on test worlds the system never trained on. "Hours" always means **active compute hours** logged by the cycle engine, not clock time, because the engine pauses.

- **Minimum success (H1, the idea works):**
  - recall (planted patterns found) at 8 h is at least 20 points higher than at 1 h;
  - precision (claims that are real) stays at least 70% throughout;
  - every confirmed connection passes the source check, meaning 100% of its quotes exist word for word in the files;
  - recall at 8 h is at least 2× B0's;
  - B2 scores clearly lower, which shows the test needs understanding.
- **Strong success (H2, it learns how to learn):**
  - after sleeping on training worlds, it reaches 50% recall on a *new* world in **half the time or less** compared with the version that hasn't slept;
  - precision doesn't drop;
  - general-ability benchmarks drop by at most 2 points;
  - the speed-up also appears, at least partly, on the transfer set with an unseen pattern type;
  - **to call it new:** the slept version beats B4 (written notes instead of sleep), and sleeping on search outcomes beats sleeping on facts only. Otherwise the result repeats published work ([RELATED_WORK.md](RELATED_WORK.md)).
- **Bonus:** it rediscovers the roughly 76-year period in real comet records with names removed and years shifted by a secret offset.

**Failure signals (each one is a useful result too):**
- Recall flattens after the first hour.
- Precision falls the longer it runs, meaning invented connections pile up.
- Sleep improves known worlds but not new ones, meaning it memorized instead of learning to learn.
- Loops beyond 4 don't help, even with STARS.

**My honest expectation:** H1 is likely, because search plus testing on generated worlds is plausible. H2 is about a coin flip, and it's the interesting one. H3 probably gives modest gains. H4 is likely achievable with LoRA plus review.

## Phases (each ends with a go/no-go check)

**Phase 0: setup and feasibility (about 1 day)** — ✅ done, verdict GO (`results/phase0/README.md`)
- `uv` project with PyTorch (CUDA) and transformers. *(Done: the card's "<4.56" advice is out of date; 4.57.1 works. See `results/phase0/README.md`.)*
- Download Ouro-1.4B (about 3 GB). With 49 GB free, we can also keep the B3 baseline (Qwen3-1.7B) and the 730M looped fallback on disk from the start, about 12 GB for all models together.
- Measure:
  - text speed at 4 and 8 loops, with transformers and with vLLM (single-request generation is reported buggy, so check whether vLLM batching works);
  - VRAM use at 4 and 8 loops;
  - whether one LoRA training step fits (512 tokens, batch 1, gradient checkpointing).
- **Go/no-go:** if training doesn't fit, fall back to a 4-bit base with LoRA, or to the 730M looped checkpoint from `ifm-ai/xllm-loop`. The measured speed sets the world sizes.

**Phase 1: chronicle generator and scorer (pure Python, no ML)** — ✅ built; B2 finds 16% (`results/phase1/README.md`); waiting on your read of the sample world
- Worlds, planted patterns, decoys, ground truth as JSON.
- A scorer that matches the system's structured claims (pattern type, record ids, period or delay, prediction) against the ground truth.
- **Go/no-go:** you can find the patterns by reading a small world, and B2 can't solve the full ones.

**Phase 1b: cycle engine (pure Python; can be built alongside Phase 1)** — ✅ done, GPU handed back in 0.9 s
- Sensors, the four states, sleep pressure, hard stops, checkpoint and resume, and the timeline log.
- It's tested with a dummy GPU workload before the real learner plugs in.
- **Go/no-go:**
  - when you start a GPU-heavy program, the engine reaches REST and frees VRAM within about 10 s;
  - after you `kill -9` it, it resumes with no lost or duplicated work.

**Phase 2: notebook and focused mode**
- Pull records into structured notebook entries; add self-quiz.
- Measure extraction accuracy against the ground truth.

**Phase 3: diffuse mode and verification**
- Pick candidates, propose patterns, run the source check, then the prediction and shuffle tests, and assign confidence.
- Track every theory's status and keep the rejected ones.
- Produce the **H1 curves** (recall and precision against hours) alongside B0, B1 and B2.
- First version of the app window (timeline and found connections) running on the fake worlds. It doubles as our tool for watching experiments: planted patterns and found patterns appear on the same timeline.

**Phase 4: sleep**
- LoRA consolidation, a spaced-repetition review scheduler (using perplexity as the "recall" signal, as in SRT), and the optional STARS regularizer.
- The memory audit after every deep sleep, graded against the source files.
- Test **H2** (time to 50% recall on new worlds, before and after sleep) and **H4** (questions answered without the notebook; general benchmark before and after).
- **Compare sleep strategies from the papers** on the same worlds and the same scorer (see [RELATED_WORK.md](RELATED_WORK.md)):
  - our sleep (LoRA on checked findings and search outcomes);
  - B4, written notes in the prompt instead of training (in the style of HypoForge, ReasoningBank and Auto-Dreamer);
  - SEAL-style self-edits (the model writes its own training data);
  - Google-style sleep (consolidation, then "dreamed" practice data).

  Report which gives the most accuracy, and after how many dreams. Check each paper's code license before reusing any of it.

**Phase 5: depth versus breadth (H3)**
- Sweep 2, 4, 6 and 8 loops, with and without STARS fine-tuning.
- Plot the exchange rate between loops and hours.

**Phase 6 (stretch)**
- The disguised real comet test.
- The "ask" interface, which answers with evidence: a chat window where you load one or several skills ("1, 2, 3") and ask about what they learned.
  - **If it knows, it says so, with citations; otherwise it says "I don't know."** Answers come only from confirmed connections, notes and the original passages. When nothing supports an answer well enough, it declines instead of guessing.
  - **Tested with answerable and unanswerable questions:** each fake world gets questions its material answers and questions it doesn't (about years, places or events that aren't there). We score correct answers, correct "I don't know"s, and bluffs separately. The goal is zero bluffs on unanswerable questions, at a recall we report honestly.

**Phase 6b: deeper connections (planted in the fake worlds first, then scored)**
- **Answers only from loaded skills:**
  - Retrieve only from the loaded skills; cite every sentence; check each sentence against its cited passage; cut what isn't supported; "I don't know" if nothing survives.
  - Bluff test: names in fake worlds are invented, so any answer about them that isn't in the loaded skills is a bluff by definition. With no skills loaded, every answer must be "I don't know."
- **Rule flips:** the same subject, with an opposite stance at different points on the timeline ("allowed here, forbidden here, allowed again").
- **Impossible by the timeline:** one person in two far-apart places in the same year; something described before it existed.
- **Unreliable source:**
  - The generator plants a chronicler who misdates or invents events.
  - The learner re-runs its checks leaving out one source at a time; a source whose removal makes contradictions vanish is flagged, with the evidence ("without this writer, the records agree").
- **Open questions and new material:**
  - The learner lists what it can't answer and why: a missing return, a cause with no effect yet, a person whose fate is unknown.
  - When files are added, it re-checks and reports which gaps closed, which predictions came true, and what now contradicts what (the held-back prediction test, applied to new material you add).
- **Fun mode, "write like this author":** an adapter trained on one author's works. Output is labelled as imitation, with care for living authors' rights. A 1.4B model gives modest imitations.
- **Real texts with disputed dates or authorship** (religious texts, for example): the timeline comes from the texts' own markers or a chronology the user picks, and the app says which. Findings are shown as consistency evidence, not verdicts.

**Phase 6c: unexpected patterns (a pattern grammar, not a fixed list)**
- The four planted types (periodic, cause_effect, chain, identity) are only the measurable start. The 2026-11-23 search found no settled taxonomy of pattern types, so the search has to be generic. The vocabulary to build it from:
  - sequential and periodic patterns: n-grams, recurring and approximate patterns (temporal data mining);
  - Allen's 13 interval relations (before, meets, overlaps, during, ...), for reigns, lifetimes and itineraries;
  - text relations: entailment, paraphrase, contradiction, independence.
- **Grammar:** a pattern is a relation between sets of events.
  - A set is any combination of attributes: kind, place, person, a detail word, source, a span of years.
  - Relations:
    - recurs, follows within a lag, happens together;
    - an interval relation;
    - a trend or change point ("after this year, X stops"; "without this source, things look better");
    - contradicts or flips;
    - holds only in one source;
    - two names are one thing;
    - an outlier that breaks a strong pattern;
    - an expected event that is missing.
- **Testing:** the search walks this space, and shuffle tests decide what is real (shuffle years, labels or sources), with a correction for testing many ideas at once. Findings are ranked by how surprising they are under the shuffles.
- **Model-proposed ideas:** in dreams, the model reads distant notes and writes a hypothesis in plain words. It's translated into the grammar and tested like anything else; untranslatable ones are kept as labelled guesses, never as findings.
- **The test of "unexpected":** a "mystery" pattern type is planted in some test worlds, and no detector is written for it. Finding it is the evidence, in the same spirit as the identity type being held out of the training worlds.

**Phase 7: public release** (once H1 holds; H2 isn't required)
- Finish the app: chat, morning report, and the first-run setup screen.
- **Skills** (see "Skills: learn a theme, keep it, share it" below): add a theme folder, Learn, the Skills learned list, select or delete, export and import, versions, and the "this needs about N nights" estimate.
- Build the Linux AppImage and the Windows zip (same folder layout, same code).
- Write docs.
- Test it on a fresh machine (see "Making it shareable").
- If H2 also holds, the release ships with the trained "study skill" adapter.

**Phase 8 (later): Android**
- Start with the companion app: the same UI in a WebView, connecting to a learner on your PC.
- Fully on-device comes later, as its own project.

## Making it shareable

**The goal:** anyone can say "here's my folder" and leave it running overnight. The next day they ask questions and get connections backed by evidence. Their content never leaves their machine.

**The desktop app is the main way to share it.** It's one window.

**Top bar:** a Choose folder button (with the file and dated-record count), a Learn / Pause button, the engine state (Day, Dreaming, Deep sleep or Resting) with its reason, and key numbers.

**Four tabs:**
1. **Timeline.** The records as dots on a time axis. A found connection draws arcs between its records, and a predicted next event is drawn dashed. You can zoom and filter by status. Click an arc to open that connection.
2. **Connections.** Every theory with its status (proposed, source-checked, prediction-tested, confirmed or rejected, with the reason). Each one shows its evidence quotes, and each quote opens the original file at that exact place.
3. **Learning history.** The learner's own diary:
   - an activity strip showing when it was in Day, Dreaming, Deep sleep or Resting, and why;
   - for each day and night: what it read, what it proposed, confirmed and rejected, what it trained in during sleep, and its memory-audit score;
   - a chart of connections and memory accuracy over time.

   The morning report sits at the top.
4. **Ask.** Chat. Every answer cites its records and says how sure it is.

How it's built:
- The UI is a thin layer: a small local web page served by the same Python engine the CLI uses, opened in its own window. Research runs and the app use the same code.

**One portable folder.** The app's own folder is both its home and its source material:
```
MyLearner/
  SandboxLearner.AppImage     ← Linux (SandboxLearner.exe in the Windows zip)
  ...your files...            ← PDFs, text, Markdown, Word, EPUB, HTML; subfolders are fine
  learner-data/               ← everything the app creates; it never reads this folder as material
    runtime/                  ← Python and PyTorch, downloaded on first run
    models/                   ← Ouro weights
    brain/                    ← notebook, its own LoRA adapter, learning history, logs
    config.toml
```
- **The launcher finds its own location.** The AppImage reads `$APPIMAGE` and the Windows `.exe` reads its own path. Everything goes next to the app, with nothing in the home directory or the system.
- **First run:** a setup screen downloads the runtime and model (about 6 GB) into `learner-data/`. The download itself stays small (about 100 MB: the launcher plus `uv`).
- **Moving or sharing:** move the folder and the learner moves with it. Zip it and you've shared a trained learner, its content and its memory together. Delete the folder and it's gone completely.
- **New files:** you add more files to the folder whenever you like, and the next Day phase picks them up.

**Supported material (text-based):**
- `.pdf` with a text layer, `.txt`, `.md`, `.docx`, `.epub`, `.html`.
- Scanned PDFs with no text layer are listed as "skipped: no text". OCR is a possible later addition.
- **Parser licenses:** only permissively licensed parsers (for example `pypdf`, `python-docx`, and EPUB read directly as zipped HTML). PyMuPDF is AGPL, so we avoid it to keep the release's license simple.

**Releases:**
- **Linux:** the AppImage.
- **Windows:** a zip with the same folder layout and the same code; only the launcher differs.
- **Android (later):** the same UI (see below).

**UI that also works on Android**
- **Mobile-first and responsive.** On a wide screen the tabs sit at the top and the timeline runs left to right. On a phone the same tabs become a bottom bar and the timeline runs top to bottom, scrolling through the years.
- **Touch-friendly:** tap targets of at least 48 px, nothing that only works on hover, and pinch to zoom the timeline.
- **The UI talks to the engine only through its local API,** and never touches files directly. So the same UI runs in the Linux window, the Windows window and an Android WebView.
- **Android path:**
  1. **Companion first:** the phone app shows and questions a learner running on your PC over the home network, using the same API. That's cheap, and it works as soon as the desktop version does.
  2. **Fully on the phone later:** this needs a mobile model runtime and a smaller model, and it's unlikely to train its weights on the phone at first. That's a separate project once the desktop version proves the idea.

The command line stays available for power users and for our experiments:
```
uv tool install sandbox-learner          # or: docker run --gpus all ...
sandbox-learner init ./my-library        # point it at a folder of txt/md/pdf/epub/html
sandbox-learner learn                    # runs in the background, adapts to the machine, resumes if stopped
sandbox-learner status                   # progress, plus what the engine is doing and why
sandbox-learner report                   # the morning report: what it found last night, with evidence
sandbox-learner ask "any recurring pattern between 1400 and 1700?"
sandbox-learner ui                       # optional local web page for asking questions
```

**What we ship and what the user brings**
- **We ship:**
  - the code;
  - the base model (downloaded on first run);
  - the *study-skill* LoRA adapter trained on the fake chronicles (if H2 holds).
- **The user brings** their own content.
- **What their copy learns** stays in their folder:
  - a notebook (SQLite);
  - their own LoRA adapter (tens of MB).
- **A trained learner is portable:** zip the notebook and the adapter and send it to someone else.

**Why it can be shared**
- **Licenses:**
  - Ouro is Apache-2.0, so the model can be redistributed;
  - Qwen3, the B3 baseline, is also Apache-2.0;
  - the STARS repository has no license, so we use our own implementation of the method;
- **A real sandbox:**
  - no network access after the one-time model download;
  - no telemetry;
  - the Docker image runs with `--network none`.
- **Honest output:**
  - every answer lists its source passages, a confidence score, and whether the pattern passed the source, prediction and shuffle checks;
  - patterns that haven't been tested are labeled as hypotheses.

**Design rules from day one** (this is what makes sharing cheap later)
- Everything is driven by a config file: no hard-coded paths, and model and loop settings are configurable.
- All state lives in the user's folder, and every stage can be stopped and resumed.
- Ingestion is separate from the learner. The fake chronicles are just one more input source, so the same pipeline runs on real content.

**Honest limits to state in the README**
- **Hardware:** it needs an NVIDIA GPU with 6 GB or more. Apple Silicon and CPU-only are untested; CPU-only would be very slow.
- **Speed:** on a 6 GB GPU, an overnight run covers roughly a book or two, not a whole library. Phase 0 measures the real number.
- **Best content type:** it works best on dated material (chronicles, news archives, diaries, lab notes, incident logs), because the prediction test needs time order. On undated material, links get weaker checks, and the output says so.
- **Real content is messy:** turning real documents into clean records is the hardest part beyond the fake worlds. Phase 7 tests ingestion on public-domain chronicles, such as the Anglo-Saxon Chronicle.

## Skills: learn a theme, keep it, share it

_Added 2026-11-23 at your request. This is the modular form of "a trained learner is portable"._

**In one line: a local NotebookLM where the notebook turns into trained behavior you can load, combine and share.** It runs on your own machine and learns by itself (no labels: its own checks against held-back material are its teacher). You chat with the skills you load, and it says "I don't know" when it doesn't.

**A skill is what the learner learned from one theme**, meaning one folder of related material: a set of papers on one topic, one chronicle, one archive.
- **In the app:**
  - "Add theme" picks a folder and "Learn" starts it.
  - Each theme gets its own brain: notes, theories with their evidence and stages, learning history, and, after Phase 4, its own LoRA adapter.
  - A **Skills learned** section lists them. Select one or several to ask questions with, or delete any. Several adapters can be loaded at once (`peft` supports this), and questions run over the selected skills' notebooks.
- **Export and import:**
  - A skill exports to one `.skill` file, a zip containing:
    - `manifest.json`: name, theme description, version, base model and its hash, the source files' names and hashes, hours of day reading and dreaming, nights slept, check scores (memory audit, how many theories were confirmed or rejected), and license;
    - the theories with their evidence quotes;
    - optionally the notes;
    - the adapter as safetensors.
  - Importing checks that the base model matches, plus the file hashes and the format version.
  - Only data formats are used (JSON, SQLite, safetensors), never pickle or code, so a skill someone shares online can be downloaded and imported without running anything from it.
  - The source material itself is optional in the file. Without it, quotes are shown as quotes, and the recipient can re-check them once they have the files.
- **Learned behavior you can add and remove:**
  - The base model is never changed. A skill's behavior is its own adapter file plus its notebook.
  - Adding a skill loads its adapter next to the others; deleting it unloads it and removes its files. Nothing has to be un-trained.
  - To measure: how many adapters can be combined before they interfere.
- **Only completed chunks:**
  - Exports contain only finished units: confirmed theories, and completed sleep checkpoints, each with its version.
  - Work in progress (queued items, theories still being tested) never leaves the machine. This uses the engine's exactly-once work items and the theory stages.
- **The material travels with the skill, for transparency:**
  - Source files are included by default (with hashes), so anyone can open the exact passage behind each connection.
  - For material that can't be shared (copyrighted or private), the files can be left out; quotes and hashes stay, and the manifest says so.
- **Command line, on Linux and Windows alike** (plain Python and `pathlib`; the folder lock already handles both):
  ```
  sandbox-learner skill add ./papers-on-sleep --name "sleep research"
  sandbox-learner learn [--skill "sleep research"]   # resource-aware engine; stop and resume any time
  sandbox-learner skill list                         # versions, hours learned, confirmed / rejected, estimate
  sandbox-learner skill use "sleep research,comets"  # which skills answer questions
  sandbox-learner skill export "sleep research" [--without-material] -o sleep-research-v3.skill
  sandbox-learner skill import ./downloads/comets-v2.skill
  sandbox-learner skill remove comets
  sandbox-learner ask "..."
  ```
  The GUI's Skills section is a thin layer over the same commands' API.
- **Versioned:** every re-training gives a new version (number plus content hash), and the manifest says what changed (new files read, theories added, confirmed or rejected). Two versions can be compared.
- **"This needs about N nights":**
  - The Phase 3 and 4 curves show when a body of material stops yielding new connections.
  - From those, the learner estimates for a new folder, before and while learning, how long it should learn. The estimate is based on its size, how densely dated it is, and how many comparisons it allows.
  - It reports when more dreaming stopped paying off: "best result after 3 nights; nothing new since".
- **Explore what it found** (you often don't know what to ask):
  - **A graph view** in the spirit of Obsidian's graph and backlinks, Heptabase and Logseq. Nodes are sources, people, places and themes; edges are checked connections, each carrying its quotes. Every source shows its backlinks (everything that connects to it).
  - **Theme clusters**, and **"start here" questions** the learner proposes from its strongest findings.
  - **Compare skills:** load one poet, then another, and see the themes they share, the themes only one has, and how each treats a shared theme, quote by quote.
  - The shuffle test carries over: shuffle which work belongs to which author, and a difference that survives the shuffle is real, not chance.
- **A findings folder you can open in Obsidian:** plain Markdown with `[[wikilinks]]`: one note per connection, per source record, and per place, person and kind of event. Stage, type and scores go in front matter, and evidence quotes link to their source note. Obsidian's graph view and backlinks then work on it as they are, and it reads fine without Obsidian. It's included in every skill export.
- **Recipes: choose how to learn before teaching, and compare later.**
  - A recipe is a named, versioned settings file covering:
    - which pattern finders, checks and question prompts are used;
    - loops and thresholds;
    - the merge rule;
    - later, the sleep strategy (ours, or the paper-inspired ones in Phase 4).
  - Every skill records the recipe it was learned with, and experiments compare recipes on the same material with the same scorer.
  - New approaches are added as new recipe options, behind the same small interfaces the code already has: a pattern finder returns candidates, a check returns pass/fail with a reason, and a workload has plan/run/save.
- **Honest limit on real themes:**
  - The checks lean on dates: prediction, shuffle, and the order of events.
  - A folder of papers on one topic is mostly undated, and its connections are between ideas, not events.
  - Those links need a different check (for example, "does paper B really cite or restate A's claim?"), plus the same verbatim quote check. Until that exists, connections in undated material are labeled as weaker.

## The mindpack: a skill you can share, update and roll back

A skill travels as one **`.mindpack`** file. The name was checked against existing extensions on 2026-11-23: no known use, and it avoids `.skill` (taken by many apps) and `.slpack` (too close to Esri's `.slpk`). It's a zip of data only, never code:

```
mindpack.json       manifest:
                      format version; skill id; name; version; parent (content hash of the version it updates);
                      created; base model (name and hash); the recipe it was learned with;
                      sources (name, sha256, records); stats; license; whether the material is included;
                      contents (every file with its sha256 and size)
theories.jsonl      every theory: stage, tests, evidence quotes, confidence, prediction
history.jsonl       every stage change, with dream time
notes.jsonl         the day-reading notes
judgments.jsonl     the model's re-reading answers (what moved, what was doubted, what merged)
sources.jsonl       the material itself (optional, on by default, for transparency)
findings/           the same findings as Obsidian-ready Markdown
adapter/            (Phase 4) the skill's LoRA adapter as safetensors
```

**Failsafe by design:**
- **Import never overwrites.** It builds the new brain in a temporary file, checks it, then moves it into place in one step. Updating a skill keeps its previous version, so rollback is a click.
- **Every file is checked against its sha256 in the manifest.** Members can't escape the archive (no absolute paths, no `..`). Sizes are capped against zip bombs. The format version and the base model are checked.
- **A different base model** imports read-only: you can explore and ask, but not keep training.
- **Before anything is written,** the import preview shows what the pack is and what it would change: new connections, changed connections, connections gone, confirmed or rejected counts, its recipe, and its material.

**Upstream (optional):**
- A mindpack can name an upstream: a git repository (for example on GitHub) that follows the mindpack template, holding released packs and, optionally, the source material.
- "Check for updates" fetches the newest release, or new material to dream on, and shows the usual import preview and diff before anything changes.
- It makes network calls, so it obeys the "check for updates" setting (off by default). Every file is checked against the hashes in the release's manifest, and the formats are data only.

**Versioned:**
- Every export of a skill gets the next version number and its parent's content hash.
- Two versions can be diffed ("+3 new connections, 1 now rejected"). An update is just a newer pack with the same skill id.

## The pipeline: every stage swappable

The learner is a pipeline of stages, and a **recipe** names the implementation used at each stage plus its settings. The stages and their implementations are listed in `learner/recipe.py`.
- **A recipe is recorded** in every brain and every mindpack, so any result can be traced to exactly how it was learned.
- **Comparing two recipes** means running both on the same worlds or books with the same scorer.
- **Adding an approach** means registering a new implementation of one stage (same inputs and outputs) and naming it in a recipe.

| Stage | Now (recipe `default@1`) | Worth swapping in (papers and methods to try) |
|---|---|---|
| reader | few-shot notes at 4 loops | SEAL-style self-edits (the model writes its own study notes); EntiGraph-style "write how A relates to B" |
| names | capitalised and never lower case; stated aliases only | entity-resolution models |
| kinds | label screen, then a majority vote both ways from re-reading | embedding clusters; NLI entailment between descriptions |
| proposers | periodic, lagged, chain, identity | sequential pattern mining (PrefixSpan or SPADE); Allen interval relations; HypoGeniC-style model-proposed hypotheses; the pattern grammar (Phase 6c) |
| tests | shape, prediction on own held-back years, shuffle or binomial chance, verbatim quotes | Benjamini–Hochberg control of false discoveries across all candidates; permutation tests for every type |
| source check | "Is this record about X?" yes/no | NLI entailment; RARR-style attribution checks |
| agenda (what to check first) | fixed order | bandit (UCB) priorities as in HypoGeniC; ReasoningBank or Auto-Dreamer-style learned strategies; **what sleep learns (H2)** |
| sleep | none yet | ours (LoRA on checked findings and search outcomes); B4 written notes; SEAL; Google-style sleep; STaR; spaced repetition (SRT); STARS for deep loops |
| ask | search over loaded skills | retrieval, cite every sentence, check each sentence, abstain when unsupported |

**Keeping in touch with papers:** each phase starts with a short related-work check for its stages (like RELATED_WORK.md did for H2). Promising methods enter as recipe options, so "does paper X beat what we have?" is one comparison run, not a rewrite. The aim is to prove or disprove published ideas in this setting, and to find the combination that is both new and the most accurate.

## Project layout

Folder: `/home/daniel/Documents/claude_projects/sandbox-learner/`. A copy of this plan goes there as `PLAN.md`.

```
PLAN.md
pyproject.toml            # uv; installable package with a `sandbox-learner` command
chronicles/  generator.py  patterns.py  scorer.py
ingest/      files.py (pdf/txt/md/docx/epub/html → records; skips learner-data/)  chronicles.py (fake-world input)
learner/     model.py (Ouro loading, loop-count control)  notebook.py (SQLite)
             focused.py  diffuse.py  verify.py  sleep.py (LoRA + review scheduler)
engine/      sensors.py (GPU/CPU/RAM/battery)  policy.py (states, sleep pressure, budget)
             scheduler.py (runs stages, checkpoints, resume, timeline log)
app/         server.py (local API over the engine)  web/ (one page: folder, learn, timeline, report, chat)
             setup.py (first-run download screen)
packaging/   launcher (finds its own folder, first-run setup), Linux AppImage and Windows zip builds (Phase 7)
cli.py       init / learn / status / ask / report / ui
config.py    one config file per user folder
experiments/ configs and run scripts, each run ≤ ~8 h (overnight)
results/     logs, plots, scores
Dockerfile   (Phase 7)
```

We reuse these rather than writing from scratch:
- the Ouro loading code from its model card;
- the STARS method, **re-implemented from the paper**: its repository has no license file, so its code can't ship;
- `peft` for LoRA.

## Verification
- **Phase 0:** a script prints tokens per second, peak VRAM, and whether a LoRA step fits.
- **Phase 1:** unit tests check that every planted pattern really is in the generated text and that decoys don't predict held-back records. You also read one small world by hand.
- **Phase 3 (source check):** unit tests feed it a theory with one invented quote and one quote that exists but has the wrong location. Both must be rejected.
- **Phase 1b:** a test script runs the engine with a dummy workload and simulates each trigger: a GPU-heavy process, high temperature (faked sensor value), low RAM, battery, and a `kill -9`. For each one it checks the state change, the time to free VRAM, and a clean resume.
- **Phases 2–5:** every experiment writes scores to `results/`, computed by the same scorer on test worlds, together with a plot. The success thresholds above are checked from those files, not by eye.
- **Phase 7 (app):** on a fresh Linux user account, someone who has never seen the project opens the AppImage, picks a folder of a public-domain chronicle, clicks Learn, leaves it overnight, and the next morning sees connections on the timeline and gets answers with evidence in chat. No terminal is needed at any step.
- **Phase 7 (Windows):** the same overnight test on a fresh Windows machine, using the unzipped folder.
- **Phase 7 (portable):** copy the trained folder to another machine and open it there. It must show the same connections and history with no new training.
- **UI:** every tab is checked at phone width (375 px) and at desktop width, with touch emulation turned on.
- **Phase 7 (CLI):** a clean install in a fresh container runs `init` → `learn` (stopped and resumed once) → `ask` on a public-domain chronicle, with the network turned off after the model download. The answers must list their source passages.

## Risks
- **Speed.** A looped 1.4B model may only manage around 15–20 tokens per second without batching. That would allow only a few hundred comparisons per hour, so worlds stay small.
- **Ouro quirks** (fixed in Phase 0, see `learner/model.py`): the model card's transformers advice is out of date, and padded batches broke because of a cache/mask bug.
- **RAM:** only about 4 GB of the 14 GB is free while other apps run. Close heavy apps before overnight runs.
- **Memory during training** may force the 4-bit base or the smaller model.
- **Windows quirks:**
  - PyTorch's deep folder paths can hit Windows' 260-character path limit, so the README says to unzip to a short path such as `C:\SandboxLearner`.
  - An unsigned `.exe` triggers a SmartScreen warning until we sign it.
- **Android on-device** needs a different runtime and a smaller model, so it can't reuse the desktop engine as it is.
- **First-run download is large:** about 6 GB.
- **UI before results:** the app is only worth polishing once H1 shows it finds real connections. Until then it stays a bare viewer for experiments.
