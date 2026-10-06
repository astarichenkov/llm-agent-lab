"""Optional ``sentence-transformers`` embedding provider.

Only used when the package is installed; imported lazily so the core pipeline
never depends on it. Never falls back to a paid API.
"""
from __future__ import annotations

from app.services.rag.embeddings.base import EmbeddingError, EmbeddingProvider


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    name = "sentence_transformers"

    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2", *, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # noqa: BLE001 - optional dependency
            raise EmbeddingError(
                "sentence-transformers is not installed"
            ) from exc
        self.model = model
        self._model = SentenceTransformer(model, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        self.dimension = int(dimension or 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
