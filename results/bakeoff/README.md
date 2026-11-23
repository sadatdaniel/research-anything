# Model bake-off: is the looped Ouro still the right model?

_2026-11-23, at your request ("run the tests on multiple candidates to see if looped is still our preferred model"). Code: [experiments/model_bakeoff.py](../../experiments/model_bakeoff.py); raw results: `*.json` here; table: `uv run python experiments/model_bakeoff_report.py`._

## Setup

- **Candidates:** every one is open, Apache-2.0, and small enough to try on the RTX 4050 (6 GB).
  - looped: Ouro-1.4B at 4 and 2 loops, Ouro-2.6B;
  - ordinary: MiniCPM5-1B and -2B (September 2026), Qwen3-1.7B-Base, Qwen3.5-2B-Base (a hybrid of linear and full attention; it needs transformers 5, so it ran in a separate environment).
- **Data:** validation worlds the prompts were never tuned on. Note-taking used the first 100 records of val-003. The yes/no checks used val-001 and val-002, about 1,000 questions each, identical for every model: they were built from one shared notebook, so only the answers differ.
- **Prompts:** the same for all, and they were written with Ouro, which may favour it a little.

## Results

| Model | Looped | Params | Events | Places | Links | Records/min | "About?" AUC | Yes if true / yes if false | Traps said yes | "Same kind?" AUC | LoRA step fits (peak MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Ouro-1.4B, 4 loops** (current) | yes | 1.43B | 96% | 88% | **94%** | 17 | **0.994** | 97% / **1%** | **7/22** | 0.891 | yes (3,614) |
| Ouro-1.4B, 2 loops | yes | 1.43B | 95% | 93% | 59% | 41 | 0.987 | 97% / 8% | 13/22 | 0.827 | not tried |
| Ouro-2.6B, 4 loops | yes | 2.67B | out of memory | | | | **0.998** | 98% / 2% | **7/22** | **0.939** | no, out of memory |
| Qwen3.5-2B-Base | no | 1.88B | **98%** | **96%** | 88% | **99** | 0.990 | 100% / 11% | 22/22 | 0.833 | yes (5,221) |
| Qwen3-1.7B-Base | no | 1.72B | **98%** | 91% | 71% | 82 | 0.988 | 96% / 4% | 11/22 | 0.934 | yes (4,448) |
| MiniCPM5-2B | no | 2.52B | 95% | 86% | 71% | 68 | 0.988 | 98% / 15% | 21/22 | 0.881 | no, out of memory |
| MiniCPM5-1B | no | 1.08B | 68% | 86% | 0% | 151 | 0.918 | 77% / 6% | 11/22 | 0.811 | yes (3,010) |
| LFM3-1.7B-Base (Oct 2026) | no | 1.73B | 71% | 0% | 0% | 240 | 0.930 | 94% / **41%** | 12/22 | 0.697 | yes (4,289) |

LFM3 (LiquidAI, LFM Open License v1.0) was added afterwards at your request:
- It's a hybrid of convolution and attention layers, and it runs its own repository code, which was read before it ran.
- With the shared prompt it writes only the event's name and then the note's end marker (`storm###`), so none of its notes are complete. This isn't our setup: it does the same alone and unpadded, and its tokenizer keeps line breaks. A prompt written for LFM3 might do better, but that would no longer be the same test.

How to read the columns:
- **Events, Places, Links:** how much of a record's note matches the answer key. Links means stated connections such as "A is in B" or "X is also called Y".
- **Records/min:** note-taking speed.
- **"About?" AUC:** dream mode's main check, "is this record about X?". It measures how well the model's yes-probability separates true from false (0.5 is chance, 1.0 is perfect).
- **Traps said yes:** "a flood of pilgrims" asked "is this about a flood?" (lower is better).
- **"Same kind?" AUC:** the label screen.

## What it says

1. **For checking and for links, the looped Ouro is still the best model that fits.** At 4 loops it says a false yes only 1% of the time; the others manage 4–15%. It falls for the fewest figures of speech (7 of 22; Qwen3.5 falls for all 22, MiniCPM5-2B for 21) and finds the most stated links (94%; next best 88%). Those are the judgments the whole pipeline rests on: a false yes in a source check becomes a false finding.
2. **Looping is what makes it careful (H3).** The same Ouro at 2 loops drops to 59% links, 8% false yes and 13 traps. That's more than twice the speed, but clearly worse on exactly the careful parts. It's the clearest evidence for H3 so far: thinking deeper helps judgment, not just speed.
3. **Bigger helps judging, but doesn't fit.** Ouro-2.6B is the best checker of all (AUC 0.998 and 0.939), and MiniCPM5-2B beats MiniCPM5-1B everywhere. But Ouro-2.6B can't read or train in 6 GB, and MiniCPM5-2B can't train. Sleep (Phase 4) needs training to fit.
4. **Ordinary models read faster, and Qwen3.5-2B reads well.** It gets 98% events, 96% places and 88% links at 99 records a minute, almost 6× Ouro. For bigger worlds, reading speed is the bottleneck.
5. **The newest isn't automatically best here.** MiniCPM5-2B has the highest general benchmark score in this size class, but it's the most easily fooled checker (15% false yes, 21 of 22 traps).

**So far:** keep Ouro-1.4B at 4 loops as the checker. The open question is whether a fast reader plus Ouro as the checker beats Ouro alone end to end. `experiments/model_e2e.sh` answers that: whole worlds (val-003 and val-004), read and dreamt by each model, plus Qwen3.5's notes checked by Ouro. Results get added below.

**Caveats:**
- 100 records and about 2,000 questions per model on validation worlds only.
- The prompts were tuned with Ouro.
- The test-world numbers come later, once one design is chosen.

## End to end: whole worlds, read and dreamt

`experiments/model_e2e.sh` and `experiments/model_lfm3.sh` ran on val-003 and val-004, 10 planted patterns in all, recipe `default@2`, at most 30 minutes of dreaming per world. Each model read both worlds completely and then dreamt on its own notes. Two split pairings had Ouro dream on a faster model's notes. Summary: `e2e_summary.json`.

| Reader → checker | Found (of 10) | Claims that were true | Decoys | Reading | Dreaming | Total | Questions asked |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base → itself | **7** | 7/8 | 0 | 9 min | 35 min | **44 min** | 7,914 |
| Ouro-1.4B (4 loops) → itself | 6 | 6/7 | 0 | 46 min | 20 min | 67 min | 3,222 |
| Qwen3.5-2B-Base → itself | 5 | **5/5** | 0 | 7 min | 36 min | 43 min | 9,150 |
| Qwen3.5-2B-Base → Ouro | 4 | 4/6 | 0 | 7 min | 29 min | 36 min | n/a |
| LFM3-1.7B-Base → Ouro | 1 | 1/1 | 0 | 2.5 min | 29 min | 32 min | n/a |
| LFM3-1.7B-Base → itself | 0 | – | 0 | 2.5 min | 38 min | 41 min | 7,295 |

What it shows:
- **On recall, nothing clearly beats anything at this size.** With 10 patterns, one pattern is 10 points, inside the noise. Two patterns no run found (val-003's p1 cycle and p5 chain).
- **Shared false claim:** the "drought at Rarkdiahox → locusts" claim that three runs share is a piece of val-004's planted chain p5, the known chain-piece scoring issue from Phase 3, not a model mistake.
- **Fast readers save time overall, but dream longer.** Qwen3-1.7B and Qwen3.5-2B read 5–6× faster than Ouro. They answer each question faster too (3.8–4.2 a second vs 2.6). But their notes raise 2.5–2.8× as many questions, because their labels are less consistent and dreaming has more kinds to sort out. End to end they took about 44 minutes against Ouro's 67.
- **The careful checker didn't turn into more findings.** In the bake-off Ouro was by far the best yes/no checker, but end to end the ordinary models' precision was just as good (5/5, 7/8 against 6/7). The statistical tests catch most of what a looser checker lets through.
- **The split design (fast reader, Ouro checking) did worse, and the reason is instructive.** In val-004, Ouro moved one of the 11 locust records into another kind, "silver swarm", a label Qwen3.5 had written for a different record. That left a gap in the 67-year cycle, and the cycle was never proposed, although Qwen3.5 alone and Ouro alone both confirmed it from 13 records.
  - A reader and a checker that use different words interact badly.
  - **The cycle finder is fragile:** one misfiled record can hide a whole cycle. A robustness fix (allowing one move to be undone, or testing the cycle with and without each questionable record) belongs on the recipe list.
- **LFM3 can't be used with the shared prompt:** 0 of 10.

## Verdict

1. **Looped Ouro is still the most careful model, but not clearly the best learner.** Its checking advantage (1% false yes) didn't show up as more findings end to end. On 10 patterns, Qwen3-1.7B (7) and Ouro (6) are a tie.
2. **For scale, an ordinary reader is about 1.5× faster overall** (5–6× faster at reading, slower at dreaming): 44 vs 67 minutes for two worlds.
3. **Don't mix models within one memory.** The split design did worst, because the checker re-sorted records into the reader's vocabulary.
4. **Phase 4 can use either:** both Ouro (3.6 GB) and Qwen3-1.7B (4.4 GB) can train in 6 GB.
5. **The deciding comparison has to be on test-v2** (122 patterns, about ±8 points): Ouro against Qwen3-1.7B, each reading and dreaming alone. Qwen3-1.7B takes about 2 hours of reading for the 30 worlds, Ouro about 9.

## Not tested: LittleBit Qwen3-8B at 0.55 bits per weight

`LittleBitLLM/littlebit-qwen3-8b-0.55bpw` is an 8B model compressed with [LittleBit-2](https://arxiv.org/abs/2603.00042) (ICML 2026). It wasn't included, for three reasons:
- **Its own card reports heavy damage.** ARC-Challenge 19.5% (chance is 25%), ARC-Easy 44 vs 84 for the uncompressed model, and WikiText-2 perplexity 24.5 vs 9.7. It's an early community checkpoint: half an epoch of distillation, where the paper uses five.
- **License:** CC BY-NC 4.0 (non-commercial), which conflicts with releasing Learn Anything openly.
- **Tooling:** it needs a custom loader (SamsungLabs/LittleBit), and LoRA sleep on its binarised factors isn't supported.

A bigger model is still worth trying when the compression keeps its quality: for example a 4B model at standard 4-bit (about 2.5 GB).
