"""Persistent local vector index implementations."""
from app.services.rag.index.base import VectorIndex, VectorIndexError
from app.services.rag.index.sqlite_index import SqliteVectorIndex

__all__ = ["SqliteVectorIndex", "VectorIndex", "VectorIndexError"]
