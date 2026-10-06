"""Generation (LLM) providers used by the Day 22 RAG pipeline.

``bge-m3`` is an embedding model and is intentionally NOT re-exported here:
generation and embedding are separate concerns.
"""
from app.services.rag.generation.base import (
    GenerationError,
    GenerationProvider,
    GenerationResult,
)
from app.services.rag.generation.deepseek import DeepSeekGenerationProvider
from app.services.rag.generation.factory import build_generation_provider
from app.services.rag.generation.ollama import OllamaGenerationProvider

__all__ = [
    "GenerationError",
    "GenerationProvider",
    "GenerationResult",
    "DeepSeekGenerationProvider",
    "OllamaGenerationProvider",
    "build_generation_provider",
]
