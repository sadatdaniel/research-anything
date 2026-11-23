"""Loading Ouro (the looped model) and controlling how many times it loops.

Two compatibility fixes live here so nothing else has to know about them:

1. transformers version. Ouro's model card says "transformers<4.56", but its bundled code now carries
   a cache fix written for >=4.56 and fails on 4.54.1. We pin 4.57.1 (see pyproject.toml).

2. Batched generation (padded prompts). Two separate problems, both fixed at load time without
   modifying the model's own files:
   a. Ouro's UniversalTransformerCache never fills `self.layers`, so transformers' Cache.get_mask_sizes
      falls back to "query length" and, during cached decoding, builds an attention mask covering only
      the newest token instead of the whole history. Padded rows then attend to the wrong positions and
      produce garbage (the "batch_size > 1 gives incorrect outputs" bug users report). We give the cache
      a correct get_mask_sizes: past length + new tokens.
   b. PyTorch's memory-efficient attention kernel rejects the head-broadcast boolean mask
      ("(*bias): last dimension must be contiguous") when the length isn't a multiple of 8. We expand the
      mask to a full, contiguous (batch, heads, queries, keys) tensor before calling SDPA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Less fragmentation of GPU memory (must be set before PyTorch's first CUDA allocation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "learner-data" / "models" / "Ouro-1.4B"

_patched = False


def _install_attention_fix() -> None:
    global _patched
    if _patched:
        return
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def sdpa_full_mask(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 4:
            b, h, q, k = query.shape[0], query.shape[1], query.shape[2], key.shape[2]
            attention_mask = attention_mask[..., :k].expand(b, h, q, k).contiguous()
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_full_mask
    _patched = True


def _fix_cache_mask_sizes(model: torch.nn.Module) -> None:
    import sys

    cache_cls = getattr(sys.modules[type(model).__module__], "UniversalTransformerCache", None)
    if cache_cls is None or getattr(cache_cls, "_sandbox_mask_fix", False):
        return

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0) -> tuple[int, int]:
        # All loops and layers see the same token history, so layer 0's length is the history length.
        return self.get_seq_length(0) + cache_position.shape[0], 0

    cache_cls.get_mask_sizes = get_mask_sizes
    cache_cls._sandbox_mask_fix = True


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object

    @property
    def looped(self) -> bool:
        return hasattr(self.model.config, "total_ut_steps")

    @property
    def loops(self) -> int:
        return int(getattr(self.model.config, "total_ut_steps", 1))

    def set_loops(self, n: int) -> None:
        """How many times the model runs its 24 layers per token (Ouro was trained with 4). An ordinary
        model always runs its layers once, so only 1 is allowed for it."""
        if n < 1:
            raise ValueError("loops must be >= 1")
        if not self.looped:
            if n != 1:
                raise ValueError("this model doesn't loop; loops must be 1")
            return
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        base.config.total_ut_steps = n  # sizes the key/value cache
        base.model.total_ut_steps = n  # the loop itself

    def cache_bytes_per_token(self) -> int:
        """Key/value cache cost per token per prompt. A looped model stores one entry per layer per loop:
        at 4 loops Ouro-1.4B needs 96 entries, about 0.8 MB per token (4x a normal model). Hybrid models
        (linear + full attention, like Qwen3.5) only keep a growing cache in their full-attention layers."""
        cfg = getattr(self.model.config, "text_config", None) or self.model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        layer_types = getattr(cfg, "layer_types", None)
        layers = sum(t == "full_attention" for t in layer_types) if layer_types else cfg.num_hidden_layers
        return self.loops * layers * 2 * cfg.num_key_value_heads * head_dim * 2  # bf16

    def safe_batch(self, tokens_per_prompt: int, cap: int = 8, margin_mb: int = 600) -> int:
        """How many prompts of this length fit in the GPU memory that's free right now."""
        free, _ = torch.cuda.mem_get_info()
        per_prompt = tokens_per_prompt * self.cache_bytes_per_token() * 1.35  # + masks, concat copies, activations
        return max(1, min(cap, int((free - margin_mb * 2**20) // per_prompt)))

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int, min_new_tokens: int = 0,
                 do_sample: bool = False, **kwargs) -> list[str]:
        tok = self.tokenizer
        enc = tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        enc.pop("token_type_ids", None)  # some tokenizers (MiniCPM5) add these; decoder-only models don't take them
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                  do_sample=do_sample, pad_token_id=tok.pad_token_id, **kwargs)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    @torch.inference_mode()
    def p_yes(self, prompts: list[str], batch: int = 8) -> list[float]:
        """For prompts ending in 'Answer:' (or similar), the model's probability that the next word is
        'yes' rather than 'no'. One forward pass, no generation and no key/value cache."""
        tok = self.tokenizer
        yes = {tok.encode(w, add_special_tokens=False)[0] for w in (" yes", " Yes")}
        no = {tok.encode(w, add_special_tokens=False)[0] for w in (" no", " No")}
        out: list[float] = []
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(self.model.device)
            enc.pop("token_type_ids", None)
            logits = self.model(**enc, use_cache=False).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            py = probs[:, list(yes)].sum(-1)
            pn = probs[:, list(no)].sum(-1)
            out += (py / (py + pn).clamp_min(1e-9)).tolist()
        return out


def load_ouro(model_dir: str | Path = DEFAULT_MODEL_DIR, loops: int = 4,
              device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _install_attention_fix()
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"  # decoder-only generation needs left padding
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
                                                 dtype=dtype, device_map=device)
    model.eval()
    _fix_cache_mask_sizes(model)
    loaded = LoadedModel(model, tok)
    loaded.set_loops(loops)
    return loaded


# Models the learner can use (the model is a swappable stage, like the rest of the pipeline).
# name: (Hugging Face id, folder in learner-data/models). All are Apache-2.0 except LFM3 (LFM Open License
# v1.0: free, but not licensed for commercial use by organisations with $10M+ annual revenue).
MODELS = {
    "ouro-1.4b": ("ByteDance/Ouro-1.4B", "Ouro-1.4B"),
    "ouro-2.6b": ("ByteDance/Ouro-2.6B", "Ouro-2.6B"),
    "minicpm5-1b": ("openbmb/MiniCPM5-1B", "MiniCPM5-1B"),
    "minicpm5-2b": ("openbmb/MiniCPM5-2B", "MiniCPM5-2B"),
    "qwen3-1.7b-base": ("Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base"),
    "qwen3.5-2b-base": ("Qwen/Qwen3.5-2B-Base", "Qwen3.5-2B-Base"),   # needs transformers >= 5
    "lfm3-1.7b-base": ("LiquidAI/LFM3-1.7B-Base", "LFM3-1.7B-Base"),  # needs transformers >= 5
}
MODELS_DIR = DEFAULT_MODEL_DIR.parent
# Models whose architecture isn't in transformers, so their repository's own Python code runs. Each was read
# before being added (2026-11-23: imports only torch, transformers and triton; no network, file or shell
# calls). The code runs from the local folder, never fetched at load time.
REMOTE_CODE = {"lfm3-1.7b-base"}   # Ouro too, through load_ouro


def load_model(name: str = "ouro-1.4b", loops: int = 4, device: str = "cuda",
               dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    """Load any model in MODELS by name. Ouro gets its fixes and its loop count; the others run once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _, folder = MODELS[name]
    model_dir = MODELS_DIR / folder
    if name.startswith("ouro"):
        return load_ouro(model_dir, loops=loops, device=device, dtype=dtype)
    remote = name in REMOTE_CODE
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=remote)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device, trust_remote_code=remote)
    model.eval()
    return LoadedModel(model, tok)
