"""Day 24 evaluation: grounded answers, automatic evidence checks, refusals.

Reuses the EXACT same 10 control questions as Day 22/23 (no expected facts
were changed) and adds:

* automatic checks per question (sources present / quotes present / quotes
  valid / grounding valid);
* a manual semantic grade (PASS/PARTIAL/FAIL) answering "is the answer
  supported by the evidence?";
* a separate negative dataset of realistic automotive questions the
  knowledge base does NOT cover, where the only correct outcome is a
  deterministic ``insufficient_context`` refusal.

The negative dataset is real: every entry was checked against the Day 21
index during authoring (see the ``note`` field and docs/week5/day24.md).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import Settings
from app.schemas.day22 import EvaluationQuestion
from app.schemas.day24 import (
    STATUS_ANSWERED,
    STATUS_GROUNDING_FAILED,
    STATUS_INSUFFICIENT_CONTEXT,
    Day24EvaluationChecks,
    Day24EvaluationGradeRecord,
    Day24EvaluationResultsResponse,
    Day24EvaluationRunAllResponse,
    Day24EvaluationRunResponse,
    Day24EvaluationSummary,
    Day24NegativeQuestion,
    Day24NegativeReport,
    Day24NegativeRunAllResponse,
    Day24NegativeRunResponse,
)
from app.services.rag.answer_service import RagAnswerService
from app.services.rag.evaluation import EvaluationError, load_questions
from app.services.rag.grounding.service import GroundedRagService

logger = logging.getLogger("app.services.rag.day24_evaluation")

GRADE_VALUES = ("pass", "partial", "fail")


class Day24EvaluationResultStore:
    """Persist Day 24 evaluation rows (auto checks + manual grade)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Day24EvaluationGradeRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable Day 24 results at %s", self.path)
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, Day24EvaluationGradeRecord] = {}
        for qid, value in raw.items():
            try:
                records[str(qid)] = Day24EvaluationGradeRecord.model_validate(value)
            except Exception:  # noqa: BLE001 - skip malformed entries
                logger.warning("Skipping malformed Day 24 record for %s", qid)
        return records

    def save(self, records: dict[str, Day24EvaluationGradeRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {qid: rec.model_dump() for qid, rec in records.items()}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(
        self, qid: str, record: Day24EvaluationGradeRecord
    ) -> dict[str, Day24EvaluationGradeRecord]:
        records = self.load()
        records[qid] = record
        self.save(records)
        return records


def load_negative_questions(path: str | Path) -> list[Day24NegativeQuestion]:
    file_path = Path(path)
    if not file_path.exists():
        raise EvaluationError(
            f"Файл с negative-вопросами не найден: {file_path}", status_code=404
        )
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"Не удалось прочитать negative dataset: {exc}"
        ) from exc
    if not isinstance(raw, list):
        raise EvaluationError("Negative dataset должен быть JSON-массивом.")
    questions: list[Day24NegativeQuestion] = []
    for index, item in enumerate(raw):
        try:
            questions.append(Day24NegativeQuestion.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - report the offending item
            raise EvaluationError(
                f"Некорректный negative-вопрос #{index + 1}: {exc}"
            ) from exc
    return questions


class Day24EvaluationService:
    """Bind the datasets to the grounded service and the result store."""

    def __init__(
        self,
        settings: Settings,
        *,
        answer_service: RagAnswerService | None = None,
        grounded_service: GroundedRagService | None = None,
        dataset_path: str | None = None,
        results_path: str | None = None,
        negative_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.answer_service = answer_service or RagAnswerService(settings)
        self.grounded = grounded_service or GroundedRagService(
            settings, answer_service=self.answer_service
        )
        self.dataset_path = dataset_path or settings.rag_day22_evaluation_path
        self.negative_path = negative_path or settings.rag_day24_negative_path
        self.store = Day24EvaluationResultStore(
            results_path or settings.rag_day24_results_path
        )

    # ------------------------------------------------------------------
    # datasets
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

    def negative_questions(self) -> list[Day24NegativeQuestion]:
        return load_negative_questions(self.negative_path)

    def get_negative(self, question_id: str) -> Day24NegativeQuestion:
        for question in self.negative_questions():
            if question.id == question_id:
                return question
        raise EvaluationError(
            f"Negative-вопрос '{question_id}' не найден.", status_code=404
        )

    # ------------------------------------------------------------------
    # automatic checks
    # ------------------------------------------------------------------
    @staticmethod
    def checks_for(result) -> Day24EvaluationChecks:
        citations = list(result.citations or [])
        return Day24EvaluationChecks(
            status=result.status,
            has_sources=len(result.sources or []) > 0,
            has_quotes=any((c.quote or "").strip() for c in citations),
            all_quotes_valid=bool(citations)
            and all(bool(c.quote_valid) for c in citations),
            grounding_valid=bool(result.grounding_valid),
        )

    def _persist_run(self, question_id: str, checks: Day24EvaluationChecks) -> None:
        records = self.store.load()
        existing = records.get(question_id) or Day24EvaluationGradeRecord()
        record = existing.model_copy(
            update={
                "status": checks.status,
                "has_sources": checks.has_sources,
                "has_quotes": checks.has_quotes,
                "all_quotes_valid": checks.all_quotes_valid,
                "grounding_valid": checks.grounding_valid,
            }
        )
        self.store.update(question_id, record)

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
    ) -> Day24EvaluationRunResponse:
        question = self.get(question_id)
        result = await self.grounded.ask(
            question.question,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
        )
        checks = self.checks_for(result)
        self._persist_run(question_id, checks)
        return Day24EvaluationRunResponse(
            question=question, result=result, checks=checks
        )

    async def run_all(
        self,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day24EvaluationRunAllResponse:
        runs: list[Day24EvaluationRunResponse] = []
        for question in self.questions():
            runs.append(
                await self.run(
                    question.id,
                    retrieval_top_k=retrieval_top_k,
                    final_top_k=final_top_k,
                    similarity_threshold=similarity_threshold,
                )
            )
        return Day24EvaluationRunAllResponse(
            runs=runs, summary=self.results().summary
        )

    # ------------------------------------------------------------------
    # summary / manual grading
    # ------------------------------------------------------------------
    def _summary(
        self,
        questions: list[EvaluationQuestion],
        records: dict[str, Day24EvaluationGradeRecord],
    ) -> Day24EvaluationSummary:
        supported = {grade: 0 for grade in GRADE_VALUES}
        status_counts = {
            STATUS_ANSWERED: 0,
            STATUS_INSUFFICIENT_CONTEXT: 0,
            STATUS_GROUNDING_FAILED: 0,
        }
        sources_present = quotes_present = quotes_valid = grounding_valid = 0
        evaluated = 0
        for question in questions:
            record = records.get(question.id)
            if record is None:
                continue
            touched = False
            if record.has_sources is not None:
                sources_present += int(bool(record.has_sources))
                touched = True
            if record.has_quotes is not None:
                quotes_present += int(bool(record.has_quotes))
                touched = True
            if record.all_quotes_valid is not None:
                quotes_valid += int(bool(record.all_quotes_valid))
                touched = True
            if record.grounding_valid is not None:
                grounding_valid += int(bool(record.grounding_valid))
                touched = True
            if record.status in status_counts:
                status_counts[record.status] += 1
                touched = True
            if record.answer_supported_by_evidence in supported:
                supported[record.answer_supported_by_evidence] += 1
                touched = True
            if touched:
                evaluated += 1
        return Day24EvaluationSummary(
            total_questions=len(questions),
            evaluated=evaluated,
            sources_present=sources_present,
            quotes_present=quotes_present,
            quotes_valid=quotes_valid,
            grounding_valid=grounding_valid,
            answered=status_counts[STATUS_ANSWERED],
            insufficient_context=status_counts[STATUS_INSUFFICIENT_CONTEXT],
            grounding_failed=status_counts[STATUS_GROUNDING_FAILED],
            supported=supported,
        )

    def results(self) -> Day24EvaluationResultsResponse:
        questions = self.questions()
        records = self.store.load()
        return Day24EvaluationResultsResponse(
            results=records, summary=self._summary(questions, records)
        )

    def update_result(
        self, question_id: str, record: Day24EvaluationGradeRecord
    ) -> Day24EvaluationResultsResponse:
        """Persist the manual semantic grade WITHOUT touching auto fields."""
        self.get(question_id)
        existing = self.store.load().get(question_id) or Day24EvaluationGradeRecord()
        merged = existing.model_copy(
            update={
                "answer_supported_by_evidence": record.answer_supported_by_evidence
            }
        )
        self.store.update(question_id, merged)
        return self.results()

    # ------------------------------------------------------------------
    # negative evaluation
    # ------------------------------------------------------------------
    async def run_negative(
        self,
        question_id: str,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day24NegativeRunResponse:
        question = self.get_negative(question_id)
        result = await self.grounded.ask(
            question.question,
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
        )
        return Day24NegativeRunResponse(
            question=question,
            expected_status=question.expected_status,
            actual_status=result.status,
            llm_called=bool(result.llm_called),
            passed=result.status == question.expected_status,
            best_score=result.retrieval.best_score,
            answer_threshold=result.retrieval.answer_threshold,
        )

    async def run_all_negative(
        self,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day24NegativeRunAllResponse:
        runs: list[Day24NegativeRunResponse] = []
        for question in self.negative_questions():
            runs.append(
                await self.run_negative(
                    question.id,
                    retrieval_top_k=retrieval_top_k,
                    final_top_k=final_top_k,
                    similarity_threshold=similarity_threshold,
                )
            )
        return Day24NegativeRunAllResponse(runs=runs)

    async def negative_report(
        self,
        *,
        retrieval_top_k: int | None = None,
        final_top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> Day24NegativeReport:
        response = await self.run_all_negative(
            retrieval_top_k=retrieval_top_k,
            final_top_k=final_top_k,
            similarity_threshold=similarity_threshold,
        )
        runs = response.runs
        return Day24NegativeReport(
            questions=len(runs),
            correctly_refused=sum(1 for run in runs if run.passed),
            incorrectly_answered=sum(1 for run in runs if not run.passed),
            llm_invoked=sum(1 for run in runs if run.llm_called),
            runs=runs,
        )
