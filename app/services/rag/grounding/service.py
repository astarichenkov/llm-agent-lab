"""Day 24 grounded RAG service: gate -> grounded LLM -> citation validation.

This service reuses the Day 23 improved retrieval (query rewrite + wider
candidate search + similarity filter) untouched. On top of it it adds three
anti-hallucination layers:

    Layer 1  retrieval filter   (Day 23)  weak chunks never enter the context
    Layer 2  grounding gate      (here)   no accepted context -> NO LLM call
    Layer 3  citation validator  (here)   the LLM's chunk_id + quote must be
                                          backed by a real retrieved chunk

The result is always structured: an answer is only marked ``answered`` when
every citation was verified. Otherwise the service returns
``insufficient_context`` or ``grounding_failed`` — it never fabricates a
source and never lets the LLM supply source metadata.
"""
from __future__ import annotations

import json
import logging

from app.config import Settings
from app.schemas.day23 import RewriteInfo
from app.schemas.day24 import (
    ERR_ANSWER_WITHOUT_EVIDENCE,
    ERR_INVALID_MODEL_OUTPUT,
    GATE_BELOW_ANSWER_THRESHOLD,
    GATE_NO_ACCEPTED_CONTEXT,
    GATE_PASSED,
    GROUNDING_FAILED_MESSAGE,
    INSUFFICIENT_CLARIFICATION,
    INSUFFICIENT_MESSAGE,
    STATUS_ANSWERED,
    STATUS_GROUNDING_FAILED,
    STATUS_INSUFFICIENT_CONTEXT,
    GroundedMessageLink,
    GroundedRAGResult,
    GroundedRetrieval,
)
from app.services.rag.answer_service import ImprovedRetrieval, RagAnswerService
from app.services.rag.grounding.models import GroundingGateDecision
from app.services.rag.grounding.prompt import RESPONSE_FORMAT, build_grounded_messages
from app.services.rag.grounding.validator import CitationValidator
from app.services.rag.sources.links import SourceLinkResolver

logger = logging.getLogger("app.services.rag.grounding.service")


def extract_json_object(content: str | None) -> dict | None:
    """Best-effort extraction of ONE JSON object from an LLM response.

    Accepts raw JSON or JSON wrapped in a `````json`` code fence. Returns None
    when no object can be parsed — the caller then reports
    ``invalid_model_output`` instead of guessing.
    """
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        # Drop a leading language tag (json / JSON).
        newline = text.find("\n")
        if newline != -1:
            first = text[:newline].strip().lower()
            if first in ("json", "json_object", ""):
                text = text[newline + 1 :]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


class GroundedRagService:
    """Orchestrate the Day 24 grounded pipeline over the Day 21 index."""

    def __init__(
        self,
        settings: Settings,
        *,
        answer_service: RagAnswerService | None = None,
        source_link_resolver: SourceLinkResolver | None = None,
    ) -> None:
        self.settings = settings
        self.answer_service = answer_service or RagAnswerService(settings)
        self.source_links = source_link_resolver or SourceLinkResolver(settings)

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def generation(self):
        return self.answer_service.generation

    def status(self) -> dict:
        base = self.answer_service.status_day23()
        return {
            **base,
            "answer_threshold": self.answer_service.default_similarity_threshold,
            "demo_question_manual": getattr(
                self.settings, "rag_day24_demo_question_manual", ""
            ),
            "demo_question_telegram": getattr(
                self.settings, "rag_day24_demo_question_telegram", ""
            ),
            "demo_question_insufficient": getattr(
                self.settings, "rag_day24_demo_question_insufficient", ""
            ),
        }

    def _common(self) -> dict:
        common = self.answer_service._common_kwargs()
        return {
            "model": common["model"],
            "generation_provider": common["generation_provider"],
            "temperature": common["temperature"],
            "embedding_model": common["embedding_model"],
            "index_path": common["index_path"],
            "chunking": common["chunking"],
        }

    # ------------------------------------------------------------------
    # gate + trace
    # ------------------------------------------------------------------
    @staticmethod
    def _gate(prepared: ImprovedRetrieval) -> GroundingGateDecision:
        """Decide whether the retrieved context can justify an answer.

        Uses the Day 23 similarity threshold as the answer threshold
        (Option A): a chunk that was not relevant enough for retrieval must
        not be relevant enough to justify an answer.
        """
        threshold = float(prepared.similarity_threshold)
        context_count = len(prepared.context_chunks)
        accepted_count = len(prepared.accepted_candidates)
        # Best score among ALL retrieved candidates so the refusal UI can show
        # "best score vs required threshold" even when nothing was accepted.
        scores = [float(c.similarity) for c in prepared.retrieved_candidates]
        best = max(scores) if scores else None

        if context_count == 0 or accepted_count == 0:
            return GroundingGateDecision(
                passed=False,
                reason=GATE_NO_ACCEPTED_CONTEXT,
                best_score=best,
                required=threshold,
                accepted_count=accepted_count,
                context_count=context_count,
            )
        if best is None or best < threshold:
            return GroundingGateDecision(
                passed=False,
                reason=GATE_BELOW_ANSWER_THRESHOLD,
                best_score=best,
                required=threshold,
                accepted_count=accepted_count,
                context_count=context_count,
            )
        return GroundingGateDecision(
            passed=True,
            reason=GATE_PASSED,
            best_score=best,
            required=threshold,
            accepted_count=accepted_count,
            context_count=context_count,
        )

    def _trace(
        self, prepared: ImprovedRetrieval, gate: GroundingGateDecision
    ) -> GroundedRetrieval:
        rewrite = prepared.rewrite
        return GroundedRetrieval(
            rewritten_query=prepared.rewritten_query,
            rewrite=RewriteInfo(
                original_query=rewrite.original_query,
                rewritten_query=rewrite.rewritten_query,
                applied=rewrite.applied,
                error=rewrite.error,
                provider=rewrite.provider,
                model=rewrite.model,
            ),
            retrieval_top_k=prepared.retrieval_top_k,
            final_top_k=prepared.final_top_k,
            similarity_threshold=prepared.similarity_threshold,
            answer_threshold=gate.required,
            gate_passed=gate.passed,
            gate_reason=gate.reason,
            best_score=gate.best_score,
            accepted_count=gate.accepted_count,
            rejected_count=prepared.rejected_count,
            retrieved_count=len(prepared.retrieved_candidates),
            context_count=gate.context_count,
            retrieved_candidates=prepared.retrieved_candidates,
            accepted_candidates=prepared.accepted_candidates,
            rejected_candidates=prepared.rejected_candidates,
            context_chunks=prepared.context_chunks,
        )

    # ------------------------------------------------------------------
    # result builders
    # ------------------------------------------------------------------
    def _insufficient(
        self,
        question: str,
        retrieval: GroundedRetrieval,
        common: dict,
        *,
        llm_called: bool,
        message: str | None = None,
        pipeline: list[str] | None = None,
    ) -> GroundedRAGResult:
        return GroundedRAGResult(
            status=STATUS_INSUFFICIENT_CONTEXT,
            question=question,
            answer=None,
            message=message or INSUFFICIENT_MESSAGE,
            clarification_request=INSUFFICIENT_CLARIFICATION,
            grounding_valid=False,
            sources=[],
            citations=[],
            validation_errors=[],
            llm_called=llm_called,
            retrieval=retrieval,
            pipeline=pipeline
            or [
                "Question",
                "Day 23 Improved Retrieval",
                "Grounding Gate",
                f"insufficient_context ({retrieval.gate_reason}) — LLM not called"
                if not llm_called
                else "insufficient_context (LLM refused)",
            ],
            **common,
        )

    def _enrich(self, citation):
        """Attach a backend-resolved source link to a VALID citation.

        Invalid citations never receive a trusted link. The link is built
        only from the validated chunk metadata + configuration, never from
        the LLM output.
        """
        if not getattr(citation, "quote_valid", False):
            return citation
        link = self.source_links.resolve(citation)
        context_links = [
            GroundedMessageLink(
                message_id=item.message_id,
                url=item.url,
                is_cited=item.is_cited,
            )
            for item in (link.context_links or [])
        ]
        return citation.model_copy(
            update={
                "url": link.url,
                "cited_message_id": link.cited_message_id,
                "topic_id": link.topic_id,
                "link_kind": link.link_kind,
                "link_note": link.note,
                "context_links": context_links,
            }
        )

    @staticmethod
    def _sources(citations) -> list:
        """Deduplicate sources by (source_type, chunk_id, cited_message_id)."""
        seen: set = set()
        sources: list = []
        for citation in citations:
            if not getattr(citation, "quote_valid", False):
                continue
            key = (
                citation.source_type,
                citation.chunk_id,
                citation.cited_message_id,
            )
            if key in seen:
                continue
            seen.add(key)
            sources.append(citation.as_source())
        return sources

    def _failed(
        self,
        question: str,
        retrieval: GroundedRetrieval,
        common: dict,
        *,
        answer: str | None,
        citations,
        errors: list[str],
        finish_reason: str | None,
        pipeline: list[str],
    ) -> GroundedRAGResult:
        valid_sources = self._sources(citations)
        return GroundedRAGResult(
            status=STATUS_GROUNDING_FAILED,
            question=question,
            answer=answer,
            message=GROUNDING_FAILED_MESSAGE,
            clarification_request=INSUFFICIENT_CLARIFICATION,
            grounding_valid=False,
            sources=valid_sources,
            citations=list(citations),
            validation_errors=list(errors),
            llm_called=True,
            finish_reason=finish_reason,
            retrieval=retrieval,
            pipeline=pipeline,
            **common,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def ask(
        self,
        question: str,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
        retrieval_query: str | None = None,
        task_state_text: str = "",
        recent_messages: list | None = None,
        expand_context: bool = False,
        resolve_quote_chunk: bool = False,
        allow_partial_citations: bool = False,
    ) -> GroundedRAGResult:
        """Run the grounded pipeline for one question.

        Day 25 additions (all optional and backward compatible):

        * ``retrieval_query`` — the contextual query used ONLY for retrieval;
          the generation model still answers the ORIGINAL ``question``;
        * ``task_state_text`` / ``recent_messages`` — extra grounded-prompt
          context; the citation contract is unchanged.
        """
        prepared = await self.answer_service.prepare_improved_retrieval(
            retrieval_query or question,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
            # A lossy rewrite must not hide an on-topic chunk from the
            # grounding step: union the query's candidates in.
            merge_original_query=True,
            expand_context=expand_context,
        )
        gate = self._gate(prepared)
        retrieval = self._trace(prepared, gate)
        common = self._common()

        # Layer 2 — deterministic refusal BEFORE any generation call.
        if not gate.passed:
            logger.info(
                "grounded gate refused question (reason=%s best=%s required=%s)",
                gate.reason,
                gate.best_score,
                gate.required,
            )
            return self._insufficient(question, retrieval, common, llm_called=False)

        # Layer 3 — grounded generation + validation.
        messages = build_grounded_messages(
            question,
            prepared.context_chunks,
            task_state_text=task_state_text,
            recent_messages=recent_messages,
        )
        result = await self.generation.generate(
            messages,
            temperature=self.answer_service.temperature,
            max_tokens=int(getattr(self.settings, "rag_generation_max_tokens", 1024)),
            response_format=RESPONSE_FORMAT,
        )
        pipeline = [
            "Question",
            "Day 23 Improved Retrieval",
            "Retrieval union: rewritten query + original query",
            f"Similarity filter >= {prepared.similarity_threshold}",
            f"{len(prepared.accepted_candidates)} accepted → "
            f"{len(prepared.context_chunks)} context chunks",
            f"Grounding Gate: passed (best {gate.best_score:.4f} "
            f">= {gate.required:.4f})",
            "Grounded Prompt (JSON)",
            "Generation LLM",
            "Citation Validator",
        ]

        payload = extract_json_object(result.content)
        if payload is None:
            logger.warning("grounded LLM returned unparseable JSON")
            return self._failed(
                question,
                retrieval,
                common,
                answer=None,
                citations=[],
                errors=[ERR_INVALID_MODEL_OUTPUT],
                finish_reason=result.finish_reason,
                pipeline=pipeline + [f"grounding_failed ({ERR_INVALID_MODEL_OUTPUT})"],
            )

        status = str(payload.get("status") or "").strip().lower()
        answer = payload.get("answer")
        evidence = payload.get("evidence") or []

        # The model itself may (correctly) refuse on borderline context.
        if status == STATUS_INSUFFICIENT_CONTEXT or (
            not status and not answer and not evidence
        ):
            return self._insufficient(
                question,
                retrieval,
                common,
                llm_called=True,
                message=str(answer) if answer else None,
                pipeline=pipeline + ["insufficient_context (LLM refused)"],
            )

        validator = CitationValidator(
            prepared.context_chunks, resolve_quote_chunk=resolve_quote_chunk
        )
        validated = validator.validate(evidence)
        # Resolve clickable source links only AFTER validation passes for a
        # citation (invalid evidence never gets a trusted URL).
        validated.citations = [self._enrich(c) for c in validated.citations]

        if not validated.citations:
            errors = [ERR_ANSWER_WITHOUT_EVIDENCE] + list(validated.errors)
            return self._failed(
                question,
                retrieval,
                common,
                answer=str(answer) if answer else None,
                citations=[],
                errors=errors,
                finish_reason=result.finish_reason,
                pipeline=pipeline + [f"grounding_failed ({ERR_ANSWER_WITHOUT_EVIDENCE})"],
            )

        if not validated.all_valid:
            valid_citations = [c for c in validated.citations if c.quote_valid]
            if allow_partial_citations and valid_citations:
                # Partial grounded answer (Day 25 chat): keep the VERIFIED
                # evidence and drop the unverified citations. This is what
                # lets the assistant answer the part it can prove instead of
                # refusing the whole question. Day 24 standalone keeps the
                # strict all-valid rule (allow_partial_citations=False).
                return GroundedRAGResult(
                    status=STATUS_ANSWERED,
                    question=question,
                    answer=str(answer) if answer else "",
                    message=None,
                    clarification_request=None,
                    grounding_valid=True,
                    sources=self._sources(valid_citations),
                    citations=valid_citations,
                    validation_errors=list(validated.errors),
                    llm_called=True,
                    finish_reason=result.finish_reason,
                    retrieval=retrieval,
                    pipeline=pipeline
                    + ["answered (partial evidence verified)"],
                    **common,
                )
            return self._failed(
                question,
                retrieval,
                common,
                answer=str(answer) if answer else None,
                citations=validated.citations,
                errors=list(validated.errors),
                finish_reason=result.finish_reason,
                pipeline=pipeline + ["grounding_failed (invalid citation)"],
            )

        sources = self._sources(validated.citations)
        return GroundedRAGResult(
            status=STATUS_ANSWERED,
            question=question,
            answer=str(answer) if answer else "",
            message=None,
            clarification_request=None,
            grounding_valid=True,
            sources=sources,
            citations=validated.citations,
            validation_errors=[],
            llm_called=True,
            finish_reason=result.finish_reason,
            retrieval=retrieval,
            pipeline=pipeline + ["answered (evidence verified)"],
            **common,
        )
