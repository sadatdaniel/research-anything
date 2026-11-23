# Phase 0 results: setup and feasibility

**Verdict: GO.** Ouro-1.4B runs, generates in batches and trains with LoRA on the RTX 4050 laptop GPU (6 GB), with no 4-bit fallback needed. The cycle engine hands the GPU back in under a second.

Machine: RTX 4050 Laptop (6 GB), 14 GB RAM, Python 3.12, torch 2.8.0+cu128, transformers 4.57.1, peft 0.17.1.
Raw numbers: [`feasibility.json`](feasibility.json). Reproduce: `uv run python experiments/phase0_feasibility.py`.

## Model on the GPU

| What | Result |
|---|---|
| Weights in GPU memory | 2.7 GB (2.8 GB including the CUDA context) |
| Load time | about 4 s (files already cached) |
| Sanity check | "The capital of France is" → "Paris." · "A comet that returns every 76 years was first predicted by" → "Halley in 1682." |

## Generation speed (212-token prompt, 128 new tokens)

| Loops | Batch | Tokens/s | Peak GPU memory |
|---|---|---|---|
| 2 | 1 | 27.7 | 2.9 GB |
| **4** (Ouro's default) | 1 | 13.9 | 3.0 GB |
| 8 | 1 | 7.1 | 3.3 GB |
| **4** | **4** | **41.7** (total) | 3.5 GB |

Speed falls in exact proportion to the loop count, as expected: every loop reruns all 24 layers. A batch of 4 is 3× faster than one prompt at a time. Rough capacity at 4 loops and batch 4 is about 150k generated tokens per active hour, or roughly 500–1,000 dream comparisons per hour. That's better than the "few hundred" the plan feared. Batch 8 and vLLM have not been measured yet.

## LoRA training step (rank 16 on all linear layers, 30 M trainable parameters, batch 1)

| Loops | Tokens | Fits | Seconds per step | GPU memory (whole process) |
|---|---|---|---|---|
| 4 | 512 | yes | 1.7 | 4.1 GB |
| 4 | 1024 | yes | 3.6 | 5.0 GB |
| 8 | 512 | yes | 3.1 | 4.5 GB |

Deep sleep can train at 1024 tokens within 6 GB. The engine's GPU-memory settings now use these measured numbers: 4.0 GB to read or dream, 5.2 GB to sleep.

## Cycle engine (Phase 1b), real GPU check

`uv run python -m engine.check_yield`: the learner holds 1.6 GB, then a "game" grabs 2 GB.

| What | Result | Target |
|---|---|---|
| GPU fully handed back after the game starts | **0.9 s** | ≤ 10 s |
| Back at work after the game ends | 3.1 s | n/a |

All 24 engine tests pass (`uv run pytest engine/tests`). They cover the policy rules, exactly-once work, recovery from a `kill -9` (nothing lost, nothing done twice, and the downtime is recorded in the history), the night cycle (dream → deep sleep → dream), and a crashing worker being restarted and then held in REST after repeated crashes.

## Problems found and fixed

1. **The model card's transformers advice is out of date.** It says "use <4.56 (4.54.1)". The bundled code now includes a cache fix written for ≥4.56, and on 4.54.1 it fails with "property 'key_cache' … has no setter". We pinned **4.57.1**.
2. **Batched generation gave garbage for padded prompts** (the bug users report). Root cause: Ouro's cache never fills `layers`, so transformers' `Cache.get_mask_sizes` falls back to "query length". During cached decoding, the attention mask then covered only the newest token instead of the whole history. Fixed in [`learner/model.py`](../../learner/model.py) by giving the cache a correct `get_mask_sizes`. PyTorch's fast attention kernel also needed the mask expanded to a contiguous tensor. Neither fix edits the model's own files. After the fixes, batched outputs are coherent and match batch-1 outputs closely; small greedy splits remain from bf16 rounding.
3. **GPU memory isn't freed in-process.** PyTorch keeps its CUDA context (about 100–400 MB) until the process exits. So the model runs in a separate worker process, and resting means ending that process.

## Licenses

| Component | License | Consequence |
|---|---|---|
| Ouro-1.4B | Apache-2.0 | can be redistributed |
| `ifm-ai/xllm-loop` (730M looped fallback) | Apache-2.0 | usable |
| `njuyxw/STARS` | **no license file** | can't copy its code into a public release. **We write our own STARS regularizer from the paper** (Jacobian spectral-radius penalty with one power-iteration step, plus a random loop count per batch) |

## Not done in Phase 0
- vLLM wasn't tried. Batched transformers generation now works, so vLLM is an optional speed-up that would need its own environment.
- Batch 8+ throughput wasn't measured.
