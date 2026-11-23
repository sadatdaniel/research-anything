# Phase 1 results: fake chronicles, scorer, and the keyword baseline

**Verdict: GO. Phase 1 passes.** The keyword baseline (B2) finds only 9–12% of the planted patterns, well under the 30% limit, so word matching can't solve the worlds. Two blind readers who understood the text found **7 of 7 patterns with no false claims and no decoys** (see "Blind readers" below). The worlds are hard for keyword matching and solvable by reading.

## What was built

| File | What it does |
|---|---|
| `chronicles/generator.py` | Builds worlds: `records.jsonl` (what the learner may read), `heldout.jsonl` (the last 20% of the timeline, kept for testing predictions), and `truth.json` (the answer key, including per-record ground truth for scoring note-taking in Phase 2) |
| `chronicles/phrasing.py` | How chroniclers write: each kind of event in 3 registers (plain, florid, terse), with many phrasings |
| `chronicles/scorer.py` | Scores claims: recall, precision, decoy hits, duplicates, and predictions checked against the held-out years. It also defines the **claim format** the learner will output |
| `chronicles/baseline_b2.py` | The dumb baseline: keyword groups, a period finder, and lagged keyword pairs |
| `worlds/` | 20 train, 5 val, 10 test and 10 transfer worlds (regenerate with `uv run python -m chronicles.generator splits --out worlds`) |

Each world has 200–400 visible records over 1,000 invented years, written by 5–8 chroniclers. It contains 1–2 periodic patterns, 1–2 cause-and-effect patterns and 1 chain, plus 2 decoys. Transfer worlds also contain a hidden-identity pattern and a decoy for it. That pattern type never appears in training worlds.

## What makes it a fair test

- **The same event is written many ways.** The returning comet appears as "a golden broom of fire above the sunset", "heaven showed a bearded light, yellow, where the day goes down", "a hairy star, brassy…". No single word marks it.
- **Places have two names,** and each chronicler uses their own. About 12% of records give both ("Apimbrath (called Briariam by others)"), which is how a careful reader links them.
- **Pattern words also appear in noise,** for example "a flood of pilgrims", "the star of the house rose at court", "hunger for war".
- **Random comets appear alongside the returning one.** The returning one is set apart only by its details (colour, direction, about the same number of nights), which is how Halley's comet was actually identified.
- **There are decoys,** for example a red moon three times, nine years apart, then never again; or a fire after the fair twice, while four other fairs had no fire.

## Baseline B2 results (micro-averaged over worlds)

| Split | Patterns | B2 recall | B2 precision | Decoy hits | Periodic | Cause → effect | Chain | Identity |
|---|---|---|---|---|---|---|---|---|
| test | 43 | **9%** | 4% | 1 | 2/16 | 1/17 | 1/10 | n/a |
| transfer | 49 | **12%** | 7% | 0 | see json | | | 0/10 |

Raw numbers are in `b2_test.json` and `b2_transfer.json`. (Before the location mentions described below were added, B2 scored 16% and 10%.) In its first version B2 scored 33%, because it latched onto words that never varied: every locust record said "locust", and direction words like "south/southern" were consistent. Those phrasing banks were widened until no single word covers more than about a third of the occurrences.

## Tests

`uv run pytest chronicles/tests`: 16 tests, all passing. They check that:
- the same seed gives byte-identical worlds;
- visible and held-out records are split correctly;
- every planted occurrence really is in the text: the right direction and colour phrasing, and one of the place's names;
- every cause-and-effect instance has the stated lag;
- every identity record uses one of the aliases, and the aliases are linked somewhere;
- decoys predict nothing (the red-moon run never continues; most fairs have no fire);
- noise comets never share the returning comet's direction or colour;
- the scorer gives 100% to the answer key and 0% to no claims, refuses "list every record" claims, counts decoy hits and duplicates, and only accepts predictions about the hidden future;
- **B2 recall stays under 30%** (the go/no-go check).

The tests caught one real flaw, now fixed: a hard-winter phrasing never named its river, so that planted cause couldn't be tied to its place.

## Blind readers (can someone who understands the text solve it?)

A fresh agent with no access to the answer key, the code or this conversation gets **one** file of records and is asked to find the hidden connections and write claims in the scorer's format. Its claims are then graded by `chronicles/scorer.py`. Files are in `blind/`.

| World | Records | Patterns found | False claims | Decoys fallen for | Predictions |
|---|---|---|---|---|---|
| A: the 41-record sample | 41 | **3/3** | 0 | 0 of 2 | 3/3 |
| B: a full test world (`test-000`) | 218 | **4/4** | 0 | 0 of 2 | 3/4 |

Reader A worked out the place-name pairs on its own. It rejected the red-moon run ("repeats every 9 years, then stops for good") and the fair→fire decoy ("worked twice but failed in 454, 471, 576 and 609").

Reader B (a full 218-record world) found all four patterns. Its one wrong prediction exposed a third problem, listed below.

The blind readers exposed three problems, all now fixed:
1. **The scorer's prediction tolerance was too strict.** Reader A predicted the comet for 658 (626 + 32), and the hidden record is 662. Every occurrence is shifted by up to ±2 years, so a correct "last + period" can be 4 years off. The tolerance is now `max(2 × jitter, 5% of the period)`, with a test.
2. **The map couldn't be learned.** No record ever said which city lies in which region, so "the wells of Hawoumia ran dry" couldn't be tied to "drought in Stuweisko". Now each city's first written mention names its region and its second names its river ("Hawoumia in Stuweisko", "Raewaem on the Lebren"), with occasional mentions after that. A test checks that every cause-and-effect city is placed on the map. These mentions use their own random stream, so the worlds' events and wording didn't otherwise change.
3. **A real pattern could look like a decoy.** In world B the generator happened to leave *two* comet returns in a row unrecorded (1017 and 1121), so the run seemed to stop after 909 while the chronicles carried on. Reader B rightly gave it only 55% confidence and predicted inside the visible years. Now an occurrence can go unrecorded, but never two in a row, and a test checks it.
