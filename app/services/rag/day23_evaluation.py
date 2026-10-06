"""Day 23 evaluation: baseline vs improved on the SAME 10 control questions.

Nothing is re-invented here:

* the dataset is the Day 22 ``evaluation_questions.json`` (loaded by
  :func:`app.services.rag.evaluation.load_questions`);
* retrieval reuses ``RagAnswerService`` / ``RagService`` from the Day 21 index;
* expected-source detection reuses :mod:`.source_matching`.

The service adds a similarity *score report* whose only purpose is to justify
the default threshold with real numbers instead of a guess.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from app.config import Settings
from app.schemas.day22 import EvaluationQuestion
from app.schemas.day23 import (
    Day23EvaluationGradeRecord,
    Day23EvaluationResultsResponse,
    Day23EvaluationRunAllResponse,
    Day23EvaluationRunResponse,
    Day23EvaluationSummary,
    Day23ScoreReport,
    Day23ScoreRow,
)
from app.services.rag.answer_service import RagAnswerService
from app.services.rag.evaluation import EvaluationError, load_questions
from app.services.rag.source_matching import (
    all_expected_sources_retrieved,
    matched_expected_indices,
)

logger = logging.getLogger("app.services.rag.day23_evaluation")

GRADE_VALUES = ("pass", "partial", "fail")


class Day23EvaluationResultStore:
    """Persist Day 23 manual grades in a small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Day23EvaluationGradeRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable Day 23 results at %s", self.path)
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, Day23EvaluationGradeRecord] = {}
        for qid, value in raw.items():
            try:
                records[str(qid)] = Day23EvaluationGradeRecord.model_validate(value)
            except Exception:  # noqa: BLE001 - skip malformed entries
                logger.warning("Skipping malformed Day 23 grade for %s", qid)
        return records

    def save(self, records: dict[str, Day23EvaluationGradeRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {qid: rec.model_dump() for qid, rec in records.items()}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(
        self, qid: str, record: Day23EvaluationGradeRecord
    ) -> dict[str, Day23EvaluationGradeRecord]:
        records = self.load()
        records[qid] = record
        self.save(records)
        return records


class Day23EvaluationService:
    """Run baseline vs improved and manage their manual grades."""

    def __init__(
        self,
        settings: Settings,
        *,
        answer_service: RagAnswerService | None = None,
        dataset_path: str | None = None,
        results_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.answer_service = answer_service or RagAnswerService(settings)
        self.dataset_path = dataset_path or settings.rag_day22_evaluation_path
        self.store = Day23EvaluationResultStore(
            results_path or settings.rag_day23_results_path
        )

    # ------------------------------------------------------------------
    # dataset
    # ------------------------------------------------------------------
    def questions(self) -> list[EvaluationQuestion]:
        return load_questions(self.dataset_path)

    def get(self, question_id: str) -> EvaluationQuestion:
        for question in self.questions():
            if question.id == question_id:
                return question
        raise EvaluationError(
            f"Контрольный вопрос '{question_id}' не найден.", status_code=404
        )

    # ------------------------------------------------------------------
    # runs
    # ------------------------------------------------------------------
    async def run(
        self,
        question_id: str,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
        baseline_top_k: int | None = None,
    ) -> Day23EvaluationRunResponse:
        question = self.get(question_id)
        comparison = await self.answer_service.compare_baseline_vs_improved(
            question.question,
            baseline_top_k=baseline_top_k or self.answer_service.default_top_k,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
        )
        return Day23EvaluationRunResponse(
            question=question,
            comparison=comparison,
            baseline_expected_source_retrieved=all_expected_sources_retrieved(
                comparison.baseline.chunks, question.expected_sources
            ),
            improved_expected_source_retrieved=all_expected_sources_retrieved(
                comparison.improved.context_chunks, question.expected_sources
            ),
        )

    async def run_all(
        self,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
        baseline_top_k: int | None = None,
    ) -> Day23EvaluationRunAllResponse:
        runs: list[Day23EvaluationRunResponse] = []
        for question in self.questions():
            runs.append(
                await self.run(
                    question.id,
                    retrieval_top_k=retrieval_top_k,
                    final_top_k=final_top_k,
                    similarity_threshold=similarity_threshold,
                    baseline_top_k=baseline_top_k,
                )
            )
        return Day23EvaluationRunAllResponse(runs=runs)

    # ------------------------------------------------------------------
    # threshold tuning / score report
    # ------------------------------------------------------------------
    def score_distribution(
        self,
        *,
        retrieval_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day23ScoreReport:
        """Similarity statistics of expected vs noise chunks per question.

        Purely technical: uses the original (baseline) queries and the Day 21
        index to show WHY a threshold is safe or unsafe.
        """
        top_k = retrieval_top_k or self.answer_service.default_retrieval_top_k
        threshold = (
            self.answer_service.default_similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )
        rows: list[Day23ScoreRow] = []
        best_expected: list[float] = []
        expected_kept = expected_total = noise_removed = best_kept = 0

        for question in self.questions():
            hits = self.answer_service.rag.search(question.question, top_k=top_k)
            scores = [h.score for h in hits]
            matched_rows: list[tuple[int, float]] = []
            expected_chunk_count = 0
            noise_count = 0
            for index, hit in enumerate(hits):
                matched = matched_expected_indices([hit], question.expected_sources)
                if matched:
                    expected_chunk_count += 1
                    expected_total += 1
                    if hit.score >= threshold:
                        expected_kept += 1
                    matched_rows.append((index + 1, hit.score))
                else:
                    noise_count += 1
                    if hit.score < threshold:
                        noise_removed += 1

            best_row = matched_rows[0] if matched_rows else None
            if best_row is not None:
                best_expected.append(best_row[1])
                if best_row[1] >= threshold:
                    best_kept += 1

            rows.append(
                Day23ScoreRow(
                    id=question.id,
                    question=question.question,
                    category=question.category,
                    top1=scores[0] if scores else None,
                    top5_min=min(scores[:5]) if scores else None,
                    top_retrieval_min=min(scores) if scores else None,
                    best_expected_score=best_row[1] if best_row else None,
                    best_expected_rank=best_row[0] if best_row else None,
                    expected_chunk_count=expected_chunk_count,
                    matched_at_threshold=sum(
                        1 for _, score in matched_rows if score >= threshold
                    ),
                    noise_count=noise_count,
                )
            )

        return Day23ScoreReport(
            retrieval_top_k=top_k,
            similarity_threshold=threshold,
            rows=rows,
            best_expected_min=min(best_expected) if best_expected else None,
            best_expected_avg=(
                round(statistics.fmean(best_expected), 4) if best_expected else None
            ),
            expected_kept=expected_kept,
            expected_total=expected_total,
            noise_removed=noise_removed,
            question_best_expected_kept=best_kept,
            question_count=len(rows),
        )

    # ------------------------------------------------------------------
    # manual grading
    # ------------------------------------------------------------------
    def _summary(
        self,
        questions: list[EvaluationQuestion],
        records: dict[str, Day23EvaluationGradeRecord],
    ) -> Day23EvaluationSummary:
        baseline = {grade: 0 for grade in GRADE_VALUES}
        improved = {grade: 0 for grade in GRADE_VALUES}
        baseline_hits = improved_hits = evaluated = 0
        for question in questions:
            record = records.get(question.id)
            if record is None:
                continue
            touched = False
            if record.baseline in baseline:
                baseline[record.baseline] += 1
                touched = True
            if record.improved in improved:
                improved[record.improved] += 1
                touched = True
            if record.baseline_expected_source_retrieved is not None:
                if record.baseline_expected_source_retrieved:
                    baseline_hits += 1
                touched = True
            if record.improved_expected_source_retrieved is not None:
                if record.improved_expected_source_retrieved:
                    improved_hits += 1
                touched = True
            if touched:
                evaluated += 1
        return Day23EvaluationSummary(
            total_questions=len(questions),
            evaluated=evaluated,
            baseline=baseline,
            improved=improved,
            baseline_expected_source_retrieved=baseline_hits,
            improved_expected_source_retrieved=improved_hits,
        )

    def results(self) -> Day23EvaluationResultsResponse:
        questions = self.questions()
        records = self.store.load()
        return Day23EvaluationResultsResponse(
            results=records, summary=self._summary(questions, records)
        )

    def update_result(
        self, question_id: str, record: Day23EvaluationGradeRecord
    ) -> Day23EvaluationResultsResponse:
        self.get(question_id)
        self.store.update(question_id, record)
        return self.results()
