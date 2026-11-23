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

**Phase 5: depth versus breadth (H3)**
- Sweep 2, 4, 6 and 8 loops, with and without STARS fine-tuning.
- Plot the exchange rate between loops and hours.

**Phase 6 (stretch)**
- The disguised real comet test.
- The "ask" interface, which answers with evidence.

**Phase 7: public release** (once H1 holds; H2 isn't required)
- Finish the app: chat, morning report, and the first-run setup screen.
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
