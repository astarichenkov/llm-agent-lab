"""LLM-backed query rewriter.

Reuses the SAME generation provider as the answer pipeline (no separate model
config): rewriting and answering are two calls to the configured generation
model, with different prompts and temperatures. A failure never breaks the
request — the original question is used as the retrieval query and the error
is surfaced in the result metadata (``rewrite.applied == False``).
"""
from __future__ import annotations

import logging

from app.services.rag.generation.base import GenerationError, GenerationProvider
from app.services.rag.prompts import build_rewrite_messages
from app.services.rag.rewriting.base import RewriteResult

logger = logging.getLogger("app.services.rag.rewriting")

# Hard cap on a rewritten query; the knowledge base is small, so a very long
# "query" is almost certainly the model ignoring the instructions.
MAX_REWRITTEN_QUERY_CHARS = 600
MAX_REWRITTEN_QUERY_WORDS = 80


def _clean_rewrite(text: str) -> str:
    """Extract the search query from a completion.

    Handles the common formatting noise (code fences, surrounding quotes,
    a leading ``Query:`` label, multi-line answers) without trying to be
    clever: only the first meaningful line is kept.
    """
    if not text:
        return ""
    cleaned = text.strip()
    # Drop markdown code fences if the model wrapped the query.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            first, rest = cleaned.split("\n", 1)
            cleaned = rest if first.lower().startswith(("text", "sql", "query")) else cleaned
        cleaned = cleaned.strip()
    for line in cleaned.splitlines():
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        lowered = line.lower()
        for prefix in ("search query:", "query:", "запрос:", "поисковый запрос:"):
            if lowered.startswith(prefix):
                line = line[len(prefix):].strip()
        if line:
            return line
    return ""


def _truncate(query: str) -> str:
    words = query.split()
    if len(words) > MAX_REWRITTEN_QUERY_WORDS:
        query = " ".join(words[:MAX_REWRITTEN_QUERY_WORDS])
    if len(query) > MAX_REWRITTEN_QUERY_CHARS:
        query = query[:MAX_REWRITTEN_QUERY_CHARS].rstrip()
    return query


class LLMQueryRewriter:
    """Turn a question into a retrieval query with the generation model."""

    name = "llm"

    def __init__(
        self,
        generation: GenerationProvider,
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> None:
        self.generation = generation
        self.max_tokens = max_tokens
        self.temperature = temperature

    @property
    def model(self) -> str:
        return self.generation.model

    async def rewrite(self, question: str) -> RewriteResult:
        original = (question or "").strip()
        if not original:
            return RewriteResult(
                original_query=question,
                rewritten_query=question,
                applied=False,
                error="empty question",
                provider=self.generation.name,
                model=self.model,
            )
        try:
            result = await self.generation.generate(
                build_rewrite_messages(original),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except GenerationError as exc:
            logger.warning("Query rewrite failed: %s", exc.message)
            return RewriteResult(
                original_query=original,
                rewritten_query=original,
                applied=False,
                error=exc.message,
                provider=self.generation.name,
                model=self.model,
            )
        except Exception as exc:  # noqa: BLE001 - demo must never crash here
            logger.exception("Unexpected query-rewrite failure")
            return RewriteResult(
                original_query=original,
                rewritten_query=original,
                applied=False,
                error=str(exc),
                provider=self.generation.name,
                model=self.model,
            )

        cleaned = _truncate(_clean_rewrite(result.content))
        if not cleaned:
            return RewriteResult(
                original_query=original,
                rewritten_query=original,
                applied=False,
                error="empty rewrite",
                provider=self.generation.name,
                model=self.model,
            )
        return RewriteResult(
            original_query=original,
            rewritten_query=cleaned,
            applied=True,
            provider=self.generation.name,
            model=self.model,
        )
