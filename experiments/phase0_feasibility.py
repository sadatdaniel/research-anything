"""Phase 0: can Ouro-1.4B do the job on this laptop?

Measures, and writes to results/phase0/:
  1. load time and GPU memory
  2. a sanity check that the model writes sensible text
  3. generation speed at 2, 4 and 8 loops (batch 1), with peak GPU memory
  4. batched generation (batch 4): speed, and whether outputs match batch-1 outputs (a bug is reported)
  5. whether one LoRA training step fits (512 and 1024 tokens; 4 and 8 loops), with step time and peak memory

Run:  uv run python experiments/phase0_feasibility.py
"""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

import pynvml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from learner.model import DEFAULT_MODEL_DIR as MODEL_DIR, load_ouro  # noqa: E402
OUT_DIR = ROOT / "results" / "phase0"
MB = 1024 * 1024

CHRONICLE = (
    "In the ninth year of Queen Avelind, a star with a long pale tail stood over the river Dorn for "
    "eleven nights, and the fishermen of Vesk would not put out their boats. The monks of Hollow Abbey "
    "wrote that the same sign had been seen in the days of their founder, and that it came before a "
    "hard winter. In the spring the river rose and drowned the lower fields of Marrow; two summers later "
    "the granaries of Vesk stood empty and the people ate bark. "
)
PROMPTS = [
    CHRONICLE * 2 + "\n\nWhat pattern connects these events? Answer:",
    "The capital of France is",
    "A comet that returns every 76 years was first predicted by",
    "Chronicle of the year 1066:",
]


def gpu_process_mb() -> float:
    """GPU memory held by this process, including the CUDA context PyTorch doesn't report."""
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
        if p.pid == os.getpid():
            return (p.usedGpuMemory or 0) / MB
    return 0.0


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / MB


def generate(lm, prompts: list[str], new_tokens: int) -> tuple[list[str], float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    texts = lm.generate(prompts, max_new_tokens=new_tokens, min_new_tokens=new_tokens)
    torch.cuda.synchronize()
    return texts, time.perf_counter() - start


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pynvml.nvmlInit()
    results: dict = {
        "machine": {
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / MB,
            "torch": torch.__version__,
            "python": platform.python_version(),
        }
    }

    # 1. load ------------------------------------------------------------------------------------
    t = time.perf_counter()
    lm = load_ouro(MODEL_DIR, loops=4)
    model, tok = lm.model, lm.tokenizer
    results["load"] = {
        "seconds": round(time.perf_counter() - t, 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "allocated_mb": round(torch.cuda.memory_allocated() / MB),
        "process_mb_incl_cuda_context": round(gpu_process_mb()),
    }
    print("load:", results["load"], flush=True)

    # 2. sanity ----------------------------------------------------------------------------------
    texts, _ = generate(lm, PROMPTS[1:3], 24)
    results["sanity"] = [{"prompt": p, "continuation": c} for p, c in zip(PROMPTS[1:3], texts)]
    for row in results["sanity"]:
        print(f"sanity: {row['prompt']!r} -> {row['continuation']!r}", flush=True)

    # 3. speed by loop count (batch 1) -----------------------------------------------------------
    prompt_tokens = len(tok(PROMPTS[0])["input_ids"])
    generate(lm, PROMPTS[:1], 8)  # warm-up
    results["speed"] = []
    for loops in (2, 4, 8):
        lm.set_loops(loops)
        reset_peak()
        _, secs = generate(lm, PROMPTS[:1], 128)
        row = {"loops": loops, "prompt_tokens": prompt_tokens, "new_tokens": 128,
               "seconds": round(secs, 2), "tokens_per_s": round(128 / secs, 1),
               "peak_allocated_mb": round(peak_mb())}
        results["speed"].append(row)
        print("speed:", row, flush=True)
    lm.set_loops(4)

    # 4. batch 4 vs batch 1 ----------------------------------------------------------------------
    singles = [generate(lm, [p], 48)[0][0] for p in PROMPTS]
    reset_peak()
    batched, secs = generate(lm, PROMPTS, 48)
    matches = [s.strip() == b.strip() for s, b in zip(singles, batched)]
    results["batch4"] = {
        "loops": 4, "new_tokens_each": 48, "seconds": round(secs, 2),
        "total_tokens_per_s": round(4 * 48 / secs, 1), "peak_allocated_mb": round(peak_mb()),
        "outputs_match_batch1": matches,
        "examples": [{"single": s[:160], "batched": b[:160]} for s, b in zip(singles, batched)],
    }
    print("batch4:", {k: v for k, v in results["batch4"].items() if k != "examples"}, flush=True)

    # 5. LoRA training step ----------------------------------------------------------------------
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    optim = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-4)
    long_ids = tok(CHRONICLE * 40, return_tensors="pt")["input_ids"][0]
    results["lora"] = {"trainable_params_m": round(trainable / 1e6, 2), "runs": []}
    lora.train()
    for loops, seq in ((4, 512), (4, 1024), (8, 512)):
        lm.set_loops(loops)
        reset_peak()
        ids = long_ids[:seq].unsqueeze(0).to("cuda")
        row: dict = {"loops": loops, "seq_len": seq, "batch": 1}
        try:
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
            row.update(fits=True, step_seconds=round(sum(times[1:]) / 2, 2), loss=round(loss.item(), 3),
                       peak_allocated_mb=round(peak_mb()),
                       process_mb_incl_cuda_context=round(gpu_process_mb()))
        except torch.OutOfMemoryError as exc:
            row.update(fits=False, error=str(exc).splitlines()[0][:200])
            optim.zero_grad(set_to_none=True)
        results["lora"]["runs"].append(row)
        print("lora:", row, flush=True)

    (OUT_DIR / "feasibility.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'feasibility.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
