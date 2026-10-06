"""Dependency-free deterministic local embedding fallback.

This is not a neural model. It is a hashing (feature-hashing) bag-of-words
embedder over word unigrams/bigrams and character trigrams, L2-normalized.
Its purpose is to keep the whole Day 21 pipeline runnable and testable with
zero external services while remaining a genuine *local* embedding that
supports cosine similarity.

When Ollama (preferred) or ``sentence-transformers`` is available, the factory
selects it instead; this provider is the offline safety net.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter

from app.services.rag.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[\w\u0400-\u04ff]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _features(text: str) -> list[str]:
    tokens = _tokenize(text)
    features: list[str] = list(tokens)
    features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
    compact = _WHITESPACE_RE.sub(" ", text.lower()).strip()
    features.extend(f"#{compact[i:i + 3]}" for i in range(max(0, len(compact) - 2)))
    return features


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline, dependency-free embedding provider."""

    name = "hashing"

    def __init__(self, dimension: int = 384, model: str = "hashing-ngrams") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts = Counter(_features(text or ""))
        if not counts:
            return vector
        for feature, count in counts.items():
            index = zlib.crc32(feature.encode("utf-8")) % self.dimension
            # Sub-linear term frequency weighting.
            vector[index] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector
