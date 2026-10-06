"""Day 22 evaluation: real control questions + manual PASS/PARTIAL/FAIL grades.

The dataset is a small JSON file whose questions were derived from the ACTUAL
Day 21 index (manual + Telegram), not invented. The result store is a separate
JSON file so generated answers never mix with the human grades. Both paths are
configurable and live under the git-ignored ``data/`` directory.

No automatic LLM-as-a-judge is implemented on Day 22: the summary counts only
grades the user actually assigned, and an ungraded question is never counted
as a failure.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import Settings
from app.schemas.day22 import (
    DEFAULT_TOP_K,
    GRADES,
    EvaluationGradeRecord,
    EvaluationQuestion,
    EvaluationResultsResponse,
    EvaluationRunAllResponse,
    EvaluationRunResponse,
    EvaluationSummary,
)
from app.services.rag.answer_service import RagAnswerService

logger = logging.getLogger("app.services.rag.evaluation")


class EvaluationError(Exception):
    """Dataset / grade-store problem with a browser-safe message."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def load_questions(path: str | Path) -> list[EvaluationQuestion]:
    """Load and validate the control-question dataset."""
    file_path = Path(path)
    if not file_path.exists():
        raise EvaluationError(
            f"Файл с контрольными вопросами не найден: {file_path}", status_code=404
        )
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Не удалось прочитать evaluation dataset: {exc}") from exc
    if not isinstance(raw, list):
        raise EvaluationError("Evaluation dataset должен быть JSON-массивом.")
    questions: list[EvaluationQuestion] = []
    for index, item in enumerate(raw):
        try:
            questions.append(EvaluationQuestion.model_validate(item))
        except Exception as exc:  # noqa: BLE001 - report the offending item
            raise EvaluationError(
                f"Некорректный контрольный вопрос #{index + 1}: {exc}"
            ) from exc
    return questions


class EvaluationResultStore:
    """Persist manual grades in a small JSON file (``{qid: {...}}``)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, EvaluationGradeRecord]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable evaluation results at %s", self.path)
            return {}
        if not isinstance(raw, dict):
            return {}
        records: dict[str, EvaluationGradeRecord] = {}
        for qid, value in raw.items():
            try:
                records[str(qid)] = EvaluationGradeRecord.model_validate(value)
            except Exception:  # noqa: BLE001 - skip malformed entries
                logger.warning("Skipping malformed grade for %s", qid)
        return records

    def save(self, records: dict[str, EvaluationGradeRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {qid: rec.model_dump() for qid, rec in records.items()}
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(
        self, qid: str, record: EvaluationGradeRecord
    ) -> dict[str, EvaluationGradeRecord]:
        records = self.load()
        records[qid] = record
        self.save(records)
        return records


class EvaluationService:
    """Bind the dataset to the RAG answer service and the grade store."""

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
        self.store = EvaluationResultStore(
            results_path or settings.rag_day22_results_path
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
        raise EvaluationError(f"Контрольный вопрос '{question_id}' не найден.", status_code=404)

    # ------------------------------------------------------------------
    # runs (generated answers are returned but NOT persisted)
    # ------------------------------------------------------------------
    async def run(self, question_id: str, *, top_k: int = DEFAULT_TOP_K) -> EvaluationRunResponse:
        question = self.get(question_id)
        comparison = await self.answer_service.compare(question.question, top_k=top_k)
        return EvaluationRunResponse(question=question, comparison=comparison)

    async def run_all(
        self, *, top_k: int = DEFAULT_TOP_K
    ) -> EvaluationRunAllResponse:
        runs: list[EvaluationRunResponse] = []
        for question in self.questions():
            comparison = await self.answer_service.compare(question.question, top_k=top_k)
            runs.append(EvaluationRunResponse(question=question, comparison=comparison))
        return EvaluationRunAllResponse(runs=runs)

    # ------------------------------------------------------------------
    # manual grading
    # ------------------------------------------------------------------
    def _summary(
        self, questions: list[EvaluationQuestion], records: dict[str, EvaluationGradeRecord]
    ) -> EvaluationSummary:
        no_rag = {grade: 0 for grade in GRADES}
        rag = {grade: 0 for grade in GRADES}
        source_hits = 0
        evaluated = 0
        for question in questions:
            record = records.get(question.id)
            if record is None:
                continue
            touched = False
            if record.no_rag in no_rag:
                no_rag[record.no_rag] += 1
                touched = True
            if record.rag in rag:
                rag[record.rag] += 1
                touched = True
            if record.expected_source_retrieved is not None:
                if record.expected_source_retrieved:
                    source_hits += 1
                touched = True
            if touched:
                evaluated += 1
        return EvaluationSummary(
            total_questions=len(questions),
            evaluated=evaluated,
            no_rag=no_rag,
            rag=rag,
            expected_source_retrieved=source_hits,
        )

    def results(self) -> EvaluationResultsResponse:
        questions = self.questions()
        records = self.store.load()
        return EvaluationResultsResponse(
            results=records, summary=self._summary(questions, records)
        )

    def update_result(
        self, question_id: str, record: EvaluationGradeRecord
    ) -> EvaluationResultsResponse:
        # Validate the id against the real dataset before writing anything.
        self.get(question_id)
        self.store.update(question_id, record)
        return self.results()
