"""Day 22 application service: the first end-to-end RAG request.

This is the business-logic layer used by the web API. It owns the two
pipelines and guarantees they share ONE generation model/configuration:

* No-RAG::

      question -> generation LLM -> answer

* RAG::

      question -> bge-m3 embedding -> structural SQLite index
               -> top-K chunks -> context builder
               -> generation LLM -> answer

Retrieval reuses :class:`~app.services.rag.service.RagService` from Day 21;
no SQL, no vector math and no embeddings are reimplemented here. The result is
always a structured :class:`RAGAnswer` / :class:`RAGComparison`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.schemas.day22 import (
    DEFAULT_TOP_K,
    MODE_NO_RAG,
    MODE_RAG,
    RAGAnswer,
    RAGComparison,
    RetrievedChunk,
)
from app.schemas.day23 import (
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    STATUS_BELOW_THRESHOLD,
    STATUS_RELEVANT_NOT_USED,
    STATUS_USED,
    Day23Comparison,
    ImprovedCandidate,
    ImprovedRAGResult,
    RewriteInfo,
)
from app.services.rag.context import ContextBuilder
from app.services.rag.embeddings.base import EmbeddingError
from app.services.rag.filtering.similarity_filter import SimilarityFilter
from app.services.rag.generation import (
    GenerationProvider,
    build_generation_provider,
)
from app.services.rag.index.base import VectorIndexError
from app.services.rag.prompts import build_messages
from app.services.rag.rewriting import LLMQueryRewriter, QueryRewriter
from app.services.rag.rewriting.base import RewriteResult
from app.services.rag.service import RagService

logger = logging.getLogger("app.services.rag.answer_service")


@dataclass
class ImprovedRetrieval:
    """Retrieval-only result of the Day 23 improved pipeline.

    Extracted from :meth:`RagAnswerService.answer_with_improved_rag` so the
    Day 24 grounding gate can inspect the accepted context *before* any
    generation call is made. Contains no answer and no prompt.
    """

    rewrite: RewriteResult
    rewritten_query: str
    retrieval_top_k: int
    final_top_k: int
    similarity_threshold: float
    retrieved_candidates: list = field(default_factory=list)
    accepted_candidates: list = field(default_factory=list)
    rejected_candidates: list = field(default_factory=list)
    context_chunks: list = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_candidates)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_candidates)

    @property
    def best_score(self) -> float | None:
        """Best similarity among ACCEPTED (>= threshold) candidates."""
        if not self.accepted_candidates:
            return None
        return max(float(c.similarity) for c in self.accepted_candidates)


class RagRetrievalError(Exception):
    """Retrieval/embedding failure with a browser-safe message + status code."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RagAnswerService:
    """Orchestrate No-RAG / RAG / compare over the Day 21 index."""

    def __init__(
        self,
        settings: Settings,
        *,
        rag_service: RagService | None = None,
        generation: GenerationProvider | None = None,
        context_builder: ContextBuilder | None = None,
        rewriter: QueryRewriter | None = None,
    ) -> None:
        self.settings = settings
        self.rag = rag_service or RagService(settings)
        self._generation = generation
        self._rewriter = rewriter
        self.context_builder = context_builder or ContextBuilder(
            max_chars=settings.rag_context_max_chars
        )

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------
    @property
    def generation(self) -> GenerationProvider:
        """Lazily build the configured provider (never contacts the network)."""
        if self._generation is None:
            self._generation = build_generation_provider(self.settings)
        return self._generation

    @property
    def rewriter(self) -> QueryRewriter:
        """Lazily build the query rewriter (uses the generation model)."""
        if self._rewriter is None:
            self._rewriter = LLMQueryRewriter(
                self.generation,
                max_tokens=int(getattr(self.settings, "rag_rewrite_max_tokens", 200)),
            )
        return self._rewriter

    @property
    def temperature(self) -> float:
        return float(self.settings.rag_generation_temperature)

    @property
    def default_top_k(self) -> int:
        return int(self.settings.rag_top_k)

    @property
    def default_retrieval_top_k(self) -> int:
        return int(getattr(self.settings, "rag_retrieval_top_k", DEFAULT_RETRIEVAL_TOP_K))

    @property
    def default_final_top_k(self) -> int:
        return int(getattr(self.settings, "rag_final_top_k", DEFAULT_FINAL_TOP_K))

    @property
    def default_similarity_threshold(self) -> float:
        return float(
            getattr(
                self.settings,
                "rag_similarity_threshold",
                DEFAULT_SIMILARITY_THRESHOLD,
            )
        )

    def _index_stats(self) -> dict:
        try:
            return self.rag.stats()
        except VectorIndexError as exc:  # pragma: no cover - corrupt index
            logger.exception("Cannot read RAG index stats")
            raise RagRetrievalError(str(exc.message)) from exc

    def status(self) -> dict:
        """Technical description shown in the UI (models + index)."""
        stats = self._index_stats()
        return {
            "generation_provider": self.generation.name,
            "generation_model": self.generation.model,
            "temperature": self.temperature,
            "embedding_model": stats.get("embedding_model")
            or self.settings.rag_embedding_model,
            "index_path": stats.get("index_path", str(self.rag.index_path)),
            "chunking": stats.get("chunking", ""),
            "default_top_k": self.default_top_k,
            "top_k_options": [1, 3, 5, 10],
            "index_exists": bool(stats.get("exists", False)),
            "chunks": int(stats.get("chunks", 0) or 0),
            "chunks_by_source_type": stats.get("chunks_by_source_type", {}),
        }

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    def _retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        if not self.rag.index_exists():
            raise RagRetrievalError(
                "Локальный индекс RAG не найден. Сначала постройте индекс Day 21 "
                "(python -m app.services.rag.cli index --chunking structural)."
            )
        try:
            hits = self.rag.search(question, top_k=top_k)
        except EmbeddingError as exc:
            logger.exception("Query embedding failed")
            base_url = self.settings.rag_ollama_base_url
            model = self.settings.rag_embedding_model
            raise RagRetrievalError(
                "Не удалось вычислить embedding запроса: "
                f"Ollama/{model} недоступна по адресу {base_url}. "
                "Проверьте, что Ollama запущена и модель установлена. "
                "В Docker используйте RAG_OLLAMA_BASE_URL="
                "http://host.docker.internal:11434 (localhost внутри контейнера — "
                "это сам контейнер приложения)."
            ) from exc
        except VectorIndexError as exc:
            logger.exception("Vector search failed")
            raise RagRetrievalError(f"Ошибка поиска по индексу: {exc.message}") from exc

        return [
            RetrievedChunk(
                rank=rank,
                similarity=float(hit.score),
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                source_type=str(hit.metadata.get("source_type", "")),
                source=str(hit.metadata.get("source", "")),
                text=hit.text,
                metadata=dict(hit.metadata),
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    async def _generate(self, messages: list[dict[str, str]]):
        return await self.generation.generate(
            messages,
            temperature=self.temperature,
            max_tokens=self.settings.rag_generation_max_tokens,
        )

    def _common_kwargs(self) -> dict:
        stats = self._index_stats()
        return {
            "model": self.generation.model,
            "generation_provider": self.generation.name,
            "temperature": self.temperature,
            "embedding_model": stats.get("embedding_model")
            or self.settings.rag_embedding_model,
            "index_path": stats.get("index_path", str(self.rag.index_path)),
            "chunking": stats.get("chunking", ""),
        }

    # ------------------------------------------------------------------
    # public pipelines
    # ------------------------------------------------------------------
    async def answer_without_rag(
        self, question: str, *, top_k: int = DEFAULT_TOP_K
    ) -> RAGAnswer:
        messages = build_messages(question)
        result = await self._generate(messages)
        return RAGAnswer(
            question=question,
            mode=MODE_NO_RAG,
            answer=result.content,
            model=self.generation.model,
            generation_provider=self.generation.name,
            temperature=self.temperature,
            top_k=top_k,
            retrieval_performed=False,
            chunks=[],
            embedding_model=self.settings.rag_embedding_model,
            index_path=str(self.rag.index_path),
            finish_reason=result.finish_reason,
            pipeline=["Question", "Generation LLM", "Answer"],
        )

    async def answer_with_rag(
        self, question: str, *, top_k: int = DEFAULT_TOP_K
    ) -> RAGAnswer:
        chunks = self._retrieve(question, top_k)
        context = self.context_builder.build(chunks)
        messages = build_messages(question, context)
        result = await self._generate(messages)
        common = self._common_kwargs()
        return RAGAnswer(
            question=question,
            mode=MODE_RAG,
            answer=result.content,
            model=common["model"],
            generation_provider=common["generation_provider"],
            temperature=common["temperature"],
            top_k=top_k,
            retrieval_performed=True,
            chunks=chunks,
            embedding_model=common["embedding_model"],
            index_path=common["index_path"],
            chunking=common["chunking"],
            finish_reason=result.finish_reason,
            pipeline=[
                "Question",
                f"Embedding / {common['embedding_model']}",
                "Vector Search",
                f"Top-{top_k} chunks",
                "Context Builder",
                "Generation LLM",
                "Answer",
            ],
        )

    async def compare(
        self, question: str, *, top_k: int = DEFAULT_TOP_K
    ) -> RAGComparison:
        """Run the SAME question through both modes with the SAME model."""
        no_rag = await self.answer_without_rag(question, top_k=top_k)
        rag = await self.answer_with_rag(question, top_k=top_k)
        return RAGComparison(
            question=question,
            top_k=top_k,
            model=no_rag.model,
            temperature=no_rag.temperature,
            no_rag=no_rag,
            rag=rag,
        )

    # ------------------------------------------------------------------
    # Day 23 — improved pipeline (query rewrite + similarity filtering)
    # ------------------------------------------------------------------
    def status_day23(self) -> dict:
        """Technical description of the improved pipeline for the UI."""
        stats = self._index_stats()
        return {
            "generation_provider": self.generation.name,
            "generation_model": self.generation.model,
            "temperature": self.temperature,
            "embedding_model": stats.get("embedding_model")
            or self.settings.rag_embedding_model,
            "index_path": stats.get("index_path", str(self.rag.index_path)),
            "chunking": stats.get("chunking", ""),
            "default_top_k": self.default_top_k,
            "default_retrieval_top_k": self.default_retrieval_top_k,
            "default_final_top_k": self.default_final_top_k,
            "default_similarity_threshold": self.default_similarity_threshold,
            "min_similarity_threshold": 0.0,
            "max_similarity_threshold": 1.0,
            "retrieval_top_k_max": 50,
            "final_top_k_max": 20,
            "rewrite_provider": self.rewriter.name,
            "rewrite_model": self.rewriter.model,
            "recommended_demo_question": getattr(
                self.settings, "rag_day23_demo_question", ""
            ),
            "index_exists": bool(stats.get("exists", False)),
            "chunks": int(stats.get("chunks", 0) or 0),
            "chunks_by_source_type": stats.get("chunks_by_source_type", {}),
        }

    @staticmethod
    def _to_improved_candidate(
        chunk: RetrievedChunk,
        *,
        accepted: bool,
        used: bool,
        reason: str | None,
    ) -> ImprovedCandidate:
        status = (
            STATUS_BELOW_THRESHOLD
            if not accepted
            else (STATUS_USED if used else STATUS_RELEVANT_NOT_USED)
        )
        return ImprovedCandidate(
            rank=chunk.rank,
            similarity=chunk.similarity,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source_type=chunk.source_type,
            source=chunk.source,
            text=chunk.text,
            metadata=dict(chunk.metadata),
            accepted=accepted,
            used=used,
            reason=reason,
            status=status,
        )

    @staticmethod
    def _merge_candidates(
        primary: list[RetrievedChunk],
        secondary: list[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Union two retrieval lists by ``chunk_id`` keeping the best score.

        The rewrite can degrade retrieval for some questions (it is only a
        reformulation). Merging the rewritten-query candidates with the
        original-question candidates keeps the on-topic chunks from both
        while never lowering a chunk's score. The merged list is re-ranked
        by similarity and capped at ``top_k``.
        """
        best: dict[str, RetrievedChunk] = {}
        for chunk in list(primary) + list(secondary):
            existing = best.get(chunk.chunk_id)
            if existing is None or float(chunk.similarity) > float(existing.similarity):
                best[chunk.chunk_id] = chunk
        ordered = sorted(
            best.values(), key=lambda c: float(c.similarity), reverse=True
        )[:top_k]
        return [c.model_copy(update={"rank": index}) for index, c in enumerate(ordered, 1)]

    # ------------------------------------------------------------------
    # controlled context expansion (Day 25) — manuals/PDFs only
    # ------------------------------------------------------------------
    @property
    def expansion_enabled(self) -> bool:
        return bool(getattr(self.settings, "rag_context_expansion_enabled", True))

    def _neighbor_candidate(self, anchor: ImprovedCandidate, chunk) -> ImprovedCandidate:
        metadata = dict(getattr(chunk, "metadata", None) or {})
        base = RetrievedChunk(
            rank=anchor.rank,
            similarity=float(anchor.similarity),
            chunk_id=chunk.chunk_id,
            doc_id=getattr(chunk, "doc_id", "") or anchor.doc_id,
            source_type=str(metadata.get("source_type", anchor.source_type) or ""),
            source=str(metadata.get("source", anchor.source) or ""),
            text=getattr(chunk, "text", "") or "",
            metadata=metadata,
        )
        return self._to_improved_candidate(
            base, accepted=True, used=True, reason="doc_context_expansion"
        )

    def _select_context(
        self,
        accepted: list[ImprovedCandidate],
        final_top_k: int,
        *,
        expand_context: bool,
    ) -> list[ImprovedCandidate]:
        """Pick the final context, optionally inserting same-document neighbours.

        A strong MANUAL hit (title/spec page) often has its fasteners/scheme on
        the adjacent page. Expansion only ever adds REAL indexed chunks from
        the same document; it never invents content and never crosses sources.
        """
        selected: list[ImprovedCandidate] = []
        seen: set[str] = set()
        expanded: set[str] = set()
        added_neighbors = 0
        neighbor_fn = getattr(self.rag, "neighbors", None)
        if not expand_context or not self.expansion_enabled or not callable(neighbor_fn):
            neighbor_fn = None
        min_score = float(
            getattr(self.settings, "rag_context_expansion_min_score", 0.55)
        )
        radius = int(getattr(self.settings, "rag_context_expansion_radius", 1))
        max_neighbors = int(getattr(self.settings, "rag_context_expansion_max", 4))

        for candidate in accepted:
            if len(selected) >= final_top_k:
                break
            if candidate.chunk_id not in seen:
                selected.append(candidate)
                seen.add(candidate.chunk_id)
            if neighbor_fn is None:
                continue
            if candidate.source_type != "manual":
                continue
            if float(candidate.similarity) < min_score:
                continue
            if candidate.chunk_id in expanded:
                continue
            expanded.add(candidate.chunk_id)
            for chunk in neighbor_fn(candidate.chunk_id, radius=radius):
                if len(selected) >= final_top_k or added_neighbors >= max_neighbors:
                    break
                if chunk.chunk_id in seen:
                    continue
                neighbor = self._neighbor_candidate(candidate, chunk)
                selected.append(neighbor)
                seen.add(neighbor.chunk_id)
                added_neighbors += 1
        return selected

    async def prepare_improved_retrieval(
        self,
        question: str,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
        merge_original_query: bool = False,
        expand_context: bool = False,
    ) -> ImprovedRetrieval:
        """Run ONLY the Day 23 retrieval half of the pipeline.

        Separated from generation so Day 24 can run a grounding gate on the
        accepted context before deciding whether to call the LLM at all.
        Behaviour is identical to the retrieval half of
        :meth:`answer_with_improved_rag` unless ``merge_original_query`` is
        set, in which case the original question's retrieval candidates are
        unioned in (grounded mode uses this so a lossy rewrite cannot hide a
        relevant chunk).
        """
        retrieval_top_k = retrieval_top_k or self.default_retrieval_top_k
        final_top_k = final_top_k or self.default_final_top_k
        threshold = (
            self.default_similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )
        if final_top_k > retrieval_top_k:
            final_top_k = retrieval_top_k

        rewrite = await self.rewriter.rewrite(question)
        rewritten = rewrite.rewritten_query or question
        candidates = self._retrieve(rewritten, retrieval_top_k)
        if merge_original_query and rewritten != question:
            original_candidates = self._retrieve(question, retrieval_top_k)
            candidates = self._merge_candidates(
                candidates, original_candidates, top_k=retrieval_top_k
            )

        filtered = SimilarityFilter(threshold).filter(candidates)
        retrieved_candidates = [
            self._to_improved_candidate(
                item.candidate,
                accepted=item.accepted,
                used=False,
                reason=item.reason,
            )
            for item in filtered
        ]
        accepted_candidates = [c for c in retrieved_candidates if c.accepted]

        selected = self._select_context(
            accepted_candidates, final_top_k, expand_context=expand_context
        )
        used_ids = {candidate.chunk_id for candidate in selected}

        # Append expansion neighbours (if any) to the trace so the UI can
        # explain exactly which chunks came from document neighbours.
        known_ids = {candidate.chunk_id for candidate in retrieved_candidates}
        for candidate in selected:
            if candidate.chunk_id not in known_ids:
                retrieved_candidates.append(candidate)
                accepted_candidates.append(candidate)
                known_ids.add(candidate.chunk_id)

        for candidate in retrieved_candidates:
            candidate.used = candidate.chunk_id in used_ids
            candidate.status = (
                STATUS_USED
                if candidate.used
                else (
                    STATUS_RELEVANT_NOT_USED
                    if candidate.accepted
                    else STATUS_BELOW_THRESHOLD
                )
            )

        rejected_candidates = [c for c in retrieved_candidates if not c.accepted]
        context_chunks = [c for c in accepted_candidates if c.used]

        return ImprovedRetrieval(
            rewrite=rewrite,
            rewritten_query=rewritten,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=threshold,
            retrieved_candidates=retrieved_candidates,
            accepted_candidates=accepted_candidates,
            rejected_candidates=rejected_candidates,
            context_chunks=context_chunks,
        )

    async def answer_with_improved_rag(
        self,
        question: str,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
        expand_context: bool = False,
    ) -> ImprovedRAGResult:
        """Rewrite the query, retrieve widely, filter, then answer the ORIGINAL question.

        The rewritten query is used ONLY to compute the retrieval embedding.
        The generation prompt always receives the original question.
        """
        prepared = await self.prepare_improved_retrieval(
            question,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
            expand_context=expand_context,
        )
        threshold = prepared.similarity_threshold
        final_top_k = prepared.final_top_k
        retrieval_top_k = prepared.retrieval_top_k
        rewritten = prepared.rewritten_query
        accepted_candidates = prepared.accepted_candidates
        context_chunks = prepared.context_chunks

        context = self.context_builder.build(
            [c for c in context_chunks]  # ImprovedCandidate is a RetrievedChunk
        )
        messages = build_messages(question, context)
        result = await self._generate(messages)
        common = self._common_kwargs()
        rewrite = prepared.rewrite

        return ImprovedRAGResult(
            original_question=question,
            rewritten_query=rewritten,
            rewrite=RewriteInfo(
                original_query=rewrite.original_query,
                rewritten_query=rewrite.rewritten_query,
                applied=rewrite.applied,
                error=rewrite.error,
                provider=rewrite.provider,
                model=rewrite.model,
            ),
            retrieval_top_k=retrieval_top_k,
            similarity_threshold=threshold,
            final_top_k=final_top_k,
            retrieved_candidates=prepared.retrieved_candidates,
            accepted_candidates=accepted_candidates,
            rejected_candidates=prepared.rejected_candidates,
            context_chunks=context_chunks,
            answer=result.content,
            no_relevant_context=len(context_chunks) == 0,
            model=common["model"],
            generation_provider=common["generation_provider"],
            temperature=common["temperature"],
            embedding_model=common["embedding_model"],
            index_path=common["index_path"],
            chunking=common["chunking"],
            finish_reason=result.finish_reason,
            pipeline=[
                "Question",
                "Query Rewrite (retrieval only)",
                f"Retrieval query: {rewritten}",
                f"Embedding / {common['embedding_model']}",
                "Vector Search",
                f"Retrieval Top-{retrieval_top_k} candidates",
                f"Similarity filter >= {threshold}",
                f"{len(accepted_candidates)} passed → Final Top-{final_top_k}",
                f"{len(context_chunks)} chunks → Context Builder",
                "Generation LLM (original question)",
                "Answer",
            ],
        )

    async def compare_baseline_vs_improved(
        self,
        question: str,
        *,
        baseline_top_k: int = DEFAULT_TOP_K,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day23Comparison:
        """Run the SAME question through the Day 22 baseline and Day 23 improved."""
        baseline = await self.answer_with_rag(question, top_k=baseline_top_k)
        improved = await self.answer_with_improved_rag(
            question,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
        )
        return Day23Comparison(
            question=question,
            model=baseline.model,
            temperature=baseline.temperature,
            baseline_top_k=baseline_top_k,
            retrieval_top_k=improved.retrieval_top_k,
            final_top_k=improved.final_top_k,
            similarity_threshold=improved.similarity_threshold,
            baseline=baseline,
            improved=improved,
        )
