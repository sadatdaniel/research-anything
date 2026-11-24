"""Vectors for notes: what each record describes, as a point in space, so that the same thing told in
different words lands close together ("chalky light ... eight nights" near "a white star ... nine nights").

First use: group the records of a crowded kind by description before searching it for cycles. The cycle
diagnosis (results/diagnosis/README.md) found that most missed comet cycles hide among unrelated comets.
Grouping by description is blind to the years, so the chance test that follows stays honest. Later uses:
REM-style dreaming (associations between distant notes) and a direct readout of what sleep changed.

Embedders are swappable, like every stage:
  bge-small@1    BAAI/bge-small-en-v1.5 (33M parameters, MIT): a small dedicated sentence embedder, CPU-friendly
  own-hidden@1   the learner model's own hidden state (planned: needs the GPU; the readout for sleep)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

EMBEDDERS = {
    "bge-small@1": "BAAI/bge-small-en-v1.5: CLS vector, L2-normalised (33M parameters, MIT)",
    "own-hidden@1": "the learner model's own last hidden state (planned)",
}
BGE_DIR = Path(__file__).resolve().parents[1] / "learner-data" / "models" / "bge-small-en-v1.5"
_NUM = re.compile(r"\b\d+\b")
_CAP = re.compile(r"\b[A-Z][a-z'-]+\b")


def description(entry) -> str:
    """What the record says the thing looked like: the note's details if it has any, otherwise the record's
    text without its years and names. Years and places say when and where, which is the cycle test's job,
    so they're kept out of the vector on purpose."""
    details = [str(d) for d in (entry.details or []) if str(d).strip()]
    text = "; ".join(details) if details else entry.text
    return _CAP.sub("", _NUM.sub("", text)).strip(" ,;.") or entry.kind


class BgeEmbedder:
    def __init__(self, model_dir: Path = BGE_DIR, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        self.device = device

    def embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=128,
                               return_tensors="pt").to(self.device)
                cls = self.model(**enc).last_hidden_state[:, 0]
                out.append(self.torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


def cluster(vectors: np.ndarray, threshold: float) -> list[int]:
    """Average-linkage agglomerative clustering on cosine similarity: keep merging the two most similar
    clusters while their average similarity is at least `threshold`. Returns a cluster number per vector.
    Small groups only (a kind's records, tens of them), so the simple O(n^3) loop is fine."""
    n = len(vectors)
    if n == 0:
        return []
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-9)
    sim = v @ v.T
    clusters = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = -2.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[a], clusters[b])].mean())
                if s > best:
                    best, pair = s, (a, b)
        if best < threshold:
            break
        a, b = pair
        clusters[a] += clusters.pop(b)
    labels = [0] * n
    for c, members in enumerate(clusters):
        for i in members:
            labels[i] = c
    return labels
