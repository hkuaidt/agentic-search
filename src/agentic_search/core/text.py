"""Dependency-free text utilities: CJK-aware tokenisation, BM25, and a hashing embedder.

These keep every pipeline runnable on CPU (tests, demos, CI). Production swaps in real components
through the same interfaces: a dense embedder (e.g. BGE) for :class:`Embedder`, the community search
API for BM25.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

_CJK = r"㐀-䶿一-鿿豈-﫿"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-z0-9]+(?:[.\-'][a-z0-9]+)*")


def normalize_text(text: str) -> str:
    """NFKC + lowercase + collapse whitespace (used for exact-match dedup and cache keys)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Latin words as-is; CJK runs as unigrams + bigrams (a standard no-segmenter baseline)."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(normalize_text(text)):
        if re.match(rf"[{_CJK}]", tok):
            out.extend(tok)
            out.extend(tok[i : i + 2] for i in range(len(tok) - 1))
        else:
            out.append(tok)
    return out


class BM25:
    """Okapi BM25 over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: Iterable[str]) -> np.ndarray:
        q = list(query)
        out = np.zeros(len(self.tfs))
        for i, tf in enumerate(self.tfs):
            denom_base = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avgdl or 1.0))
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + denom_base)
            out[i] = s
        return out


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised ``[N, D]`` float32 embeddings."""
        ...


class HashingEmbedder:
    """Character n-gram feature hashing. Deterministic, no model download; good enough to cluster
    near-duplicate / paraphrased short queries in tests and demos."""

    def __init__(self, dim: int = 512, ngram: tuple[int, ...] = (1, 2, 3)):
        self.dim, self.ngram = dim, ngram

    def _bucket(self, gram: str) -> int:
        return int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little") % self.dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            s = re.sub(r"\s+", "", normalize_text(text))
            for n in self.ngram:
                for j in range(max(len(s) - n + 1, 0)):
                    mat[i, self._bucket(s[j : j + n])] += 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-12)
