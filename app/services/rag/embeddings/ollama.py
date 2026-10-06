"""Ollama embedding provider (preferred local model).

Talks to a local Ollama server over HTTP. No paid cloud API is involved.
The dimension is discovered from the first response, so it works with any
Ollama embedding model (``bge-m3``, ``nomic-embed-text``, ...).
"""
from __future__ import annotations

import logging

import httpx

from app.services.rag.embeddings.base import EmbeddingError, EmbeddingProvider

logger = logging.getLogger("app.services.rag.embeddings.ollama")


def ollama_model_available(
    base_url: str,
    model: str,
    *,
    timeout: float = 3.0,
) -> bool:
    """Return True when the Ollama server is reachable and has ``model``."""
    if not base_url:
        return False
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception:  # noqa: BLE001 - absent server is a normal fallback path
        return False
    names = [str(item.get("name", "")) for item in payload.get("models", [])]
    target = model.strip()
    return any(name == target or name.split(":")[0] == target for name in names)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embed texts through a local Ollama ``/api/embed`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        *,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client
        self.dimension = 0

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        if self._client is not None:
            response = self._client.post(url, json=payload, timeout=self.timeout)
        else:
            response = httpx.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            payload = self._post("/api/embed", {"model": self.model, "input": texts})
            embeddings = payload.get("embeddings")
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code in (404, 400):
                # Older Ollama versions expose /api/embeddings per text.
                embeddings = self._legacy_embed(texts)
            else:
                raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Cannot reach Ollama at {self.base_url}: {exc}"
            ) from exc
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected embedding payload")
        vectors = [[float(value) for value in vector] for vector in embeddings]
        if vectors and vectors[0]:
            self.dimension = len(vectors[0])
        return vectors

    def _legacy_embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            payload = self._post("/api/embeddings", {"model": self.model, "prompt": text})
            vector = payload.get("embedding")
            if not vector:
                raise EmbeddingError("Ollama returned no embedding")
            vectors.append([float(value) for value in vector])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
