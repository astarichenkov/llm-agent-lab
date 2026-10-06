"""Citation validation for Day 24.

The validator is the *hard* anti-hallucination layer. It receives the exact
chunks that were placed in the LLM context and the evidence list the model
returned, then:

1. rejects any ``chunk_id`` that was not in the context;
2. distinguishes an invented id from a chunk that exists but was not shown;
3. rejects an empty quote;
4. rejects a quote that is not literally contained in the referenced chunk
   (after whitespace-only normalization).

Crucially, the citation's source metadata (file, page, section, message ids)
is copied from the retrieved chunk here, in the backend — never from the LLM
response.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

from app.schemas.day24 import (
    ERR_CHUNK_ID_NOT_FOUND,
    ERR_CITATION_NOT_IN_CONTEXT,
    ERR_EMPTY_QUOTE,
    ERR_QUOTE_NOT_IN_CHUNK,
    GroundedCitation,
)
from app.services.rag.grounding.models import ValidatedCitations
from app.services.rag.normalization.whitespace import normalize_whitespace

__all__ = ["CitationValidator", "normalize_whitespace"]


def _metadata_of(chunk) -> dict[str, Any]:
    meta = dict(getattr(chunk, "metadata", None) or {})
    # RetrievedChunk keeps source_type/source as first-class fields; make sure
    # the raw chunk object (e.g. a hand-built fake) still contributes them.
    meta.setdefault("source_type", getattr(chunk, "source_type", ""))
    meta.setdefault("source", getattr(chunk, "source", ""))
    return meta


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _message_ids(meta: dict[str, Any]) -> list[int]:
    values = meta.get("message_ids")
    if values is None:
        values = meta.get("message_id")
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        values = [values]
    result: list[int] = []
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None and parsed not in result:
            result.append(parsed)
    return result


def _to_citation(
    chunk_id: str,
    quote: str,
    chunk,
    quote_valid: bool,
    errors: list[str],
) -> GroundedCitation:
    if chunk is None:
        return GroundedCitation(
            chunk_id=chunk_id,
            quote=quote,
            quote_valid=quote_valid,
            validation_errors=list(errors),
        )
    meta = _metadata_of(chunk)
    return GroundedCitation(
        chunk_id=chunk_id,
        quote=quote,
        quote_valid=quote_valid,
        validation_errors=list(errors),
        source_type=str(meta.get("source_type", "") or ""),
        source=str(meta.get("source", "") or ""),
        title=str(meta.get("title", "") or ""),
        section=str(meta.get("section", "") or ""),
        page=_int_or_none(meta.get("page")),
        page_from=_int_or_none(meta.get("page_from")),
        page_to=_int_or_none(meta.get("page_to")),
        message_ids=_message_ids(meta),
        date_from=str(meta.get("date_from", "") or ""),
        date_to=str(meta.get("date_to", "") or ""),
        chat_name=str(meta.get("chat_name", "") or ""),
        topic_id=_int_or_none(meta.get("topic_id")),
        chunk_text=str(getattr(chunk, "text", "") or ""),
    )


class CitationValidator:
    """Validate an LLM evidence list against the retrieved context chunks."""

    def __init__(
        self,
        context_chunks: Iterable[Any],
        *,
        known_chunk_ids: set[str] | Callable[[str], bool] | None = None,
        resolve_quote_chunk: bool = False,
    ) -> None:
        self._chunks: dict[str, Any] = {
            str(chunk.chunk_id): chunk for chunk in context_chunks
        }
        self._normalized = {
            chunk_id: normalize_whitespace(str(getattr(chunk, "text", "") or ""))
            for chunk_id, chunk in self._chunks.items()
        }
        self._known = known_chunk_ids
        # Day 25 chat only: the LLM sometimes attributes a VERBATIM quote to
        # the wrong chunk_id. When enabled, a quote that exists in exactly one
        # context chunk is re-attributed to that chunk. This never accepts a
        # paraphrase: the quote must still be a verbatim substring of the
        # provided context.
        self._resolve_quote_chunk = resolve_quote_chunk

    def _resolve_chunk_for_quote(self, quote_norm: str):
        if not quote_norm:
            return None
        matches = [
            self._chunks[chunk_id]
            for chunk_id, text in self._normalized.items()
            if quote_norm in text
        ]
        return matches[0] if len(matches) == 1 else None

    def is_known(self, chunk_id: str) -> bool | None:
        """Return True/False when a known-set was supplied, else None."""
        if self._known is None:
            return None
        if callable(self._known):
            return bool(self._known(chunk_id))
        return chunk_id in self._known

    def validate(self, raw_evidence: Any) -> ValidatedCitations:
        citations: list[GroundedCitation] = []
        errors: list[str] = []
        for item in list(raw_evidence or []):
            if not isinstance(item, dict):
                errors.append(f"{ERR_CHUNK_ID_NOT_FOUND}: malformed evidence item")
                continue
            chunk_id = str(item.get("chunk_id") or "").strip()
            quote = str(item.get("quote") or "")
            errors_here: list[str] = []

            effective_chunk_id = chunk_id
            chunk = self._chunks.get(chunk_id)
            if not chunk_id:
                errors_here.append(ERR_CHUNK_ID_NOT_FOUND)
            elif chunk is None:
                known = self.is_known(chunk_id)
                if known is False:
                    errors_here.append(ERR_CHUNK_ID_NOT_FOUND)
                else:
                    # Either explicitly known-but-not-shown or we cannot
                    # distinguish: from the LLM's context it is out of scope.
                    errors_here.append(ERR_CITATION_NOT_IN_CONTEXT)
            else:
                if not quote.strip():
                    errors_here.append(ERR_EMPTY_QUOTE)
                elif normalize_whitespace(quote) not in self._normalized[chunk_id]:
                    resolved = None
                    if self._resolve_quote_chunk:
                        resolved = self._resolve_chunk_for_quote(
                            normalize_whitespace(quote)
                        )
                    if resolved is not None:
                        chunk = resolved
                        effective_chunk_id = str(resolved.chunk_id)
                    else:
                        errors_here.append(ERR_QUOTE_NOT_IN_CHUNK)

            quote_valid = not errors_here
            citation = _to_citation(
                effective_chunk_id, quote, chunk, quote_valid, errors_here
            )
            citations.append(citation)
            for error in errors_here:
                errors.append(f"{chunk_id or '?'}: {error}")

        return ValidatedCitations(citations=citations, errors=errors)
