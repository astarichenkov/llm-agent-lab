"""Day 25 scenario evaluation.

Two reproducible multi-turn scenarios (diagnostics + maintenance) exercise
the chat memory end to end:

* every user turn updates the task state;
* deliberately incomplete follow-ups are made self-contained by the
  Contextual Query Builder;
* corrections must SUPERSEDE the old active value;
* every grounded technical answer must carry validated evidence.

Automatic checks are deterministic: they inspect the persisted task state,
the per-turn trace (contextual query) and the stored message evidence. Answer
QUALITY (goal retention, semantic support) remains a human judgment; the
runner only reports the raw state it observed. No LLM judge is introduced.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings
from app.schemas.day24 import STATUS_ANSWERED, STATUS_INSUFFICIENT_CONTEXT
from app.schemas.day25 import (
    MSG_STATUS_ANSWERED,
    MSG_STATUS_INSUFFICIENT,
    Scenario,
    ScenarioMetrics,
    ScenarioRunResult,
    ScenarioTurn,
    ScenarioTurnResult,
    TaskState,
)
from app.services.day25.chat_service import Day25ChatService
from app.services.day25.task_state import state_contains

logger = logging.getLogger("app.services.day25.evaluation")


class ScenarioError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def load_scenarios(path: str | Path) -> list[Scenario]:
    root = Path(path)
    if not root.exists():
        return []
    scenarios: list[Scenario] = []
    for file_path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            scenarios.append(Scenario.model_validate(raw))
        except Exception as exc:  # noqa: BLE001 - report the offending file
            raise ScenarioError(
                f"Некорректный файл сценария {file_path.name}: {exc}"
            ) from exc
    return scenarios


class Day25EvaluationService:
    """Run the two scenarios through the real chat pipeline."""

    def __init__(
        self,
        settings: Settings,
        *,
        chat_service: Day25ChatService | None = None,
        scenarios_path: str | None = None,
        results_path: str | None = None,
    ) -> None:
        self.settings = settings
        self.chat = chat_service or Day25ChatService(settings)
        self.scenarios_path = scenarios_path or settings.day25_scenarios_path
        self.results_path = Path(
            results_path or settings.day25_eval_results_path
        )

    # ------------------------------------------------------------------
    # datasets
    # ------------------------------------------------------------------
    def scenarios(self) -> list[Scenario]:
        return load_scenarios(self.scenarios_path)

    def get_scenario(self, scenario_id: str) -> Scenario:
        for scenario in self.scenarios():
            if scenario.id == scenario_id:
                return scenario
        raise ScenarioError(
            f"Сценарий '{scenario_id}' не найден.", status_code=404
        )

    # ------------------------------------------------------------------
    # persisted results
    # ------------------------------------------------------------------
    def results(self) -> dict:
        if not self.results_path.exists():
            return {"runs": {}, "updated_at": ""}
        try:
            return json.loads(self.results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable Day 25 results at %s", self.results_path)
            return {"runs": {}, "updated_at": ""}

    def _save_result(self, result: ScenarioRunResult) -> None:
        payload = self.results()
        runs = payload.get("runs") or {}
        runs[result.scenario.id] = result.model_dump()
        payload = {
            "runs": runs,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    async def run_scenario(self, scenario_id: str) -> ScenarioRunResult:
        scenario = self.get_scenario(scenario_id)
        session = self.chat.repository.create_session(
            title=f"[eval] {scenario.title}"
        )
        self.chat.repository.save_state(
            session.id, TaskState()
        )
        turn_results: list[ScenarioTurnResult] = []
        error: str | None = None
        completed = 0
        for index, turn in enumerate(scenario.turns, start=1):
            try:
                response = await self.chat.send_message(session.id, turn.user)
            except Exception as exc:  # noqa: BLE001 - record and stop the run
                error = str(exc)
                logger.warning("Scenario %s stopped at turn %s: %s", scenario_id, index, exc)
                break
            completed += 1
            trace = response.assistant_message.trace or {}
            evidence = response.assistant_message.evidence or []
            checks = {entry.chunk_id for entry in evidence}
            turn_results.append(
                ScenarioTurnResult(
                    index=index,
                    user=turn.user,
                    kind=turn.kind,
                    assistant_status=response.assistant_message.status,
                    contextual_query=str(trace.get("contextual_query", "")),
                    current_message=str(trace.get("current_message", "")),
                    rewritten_query=str(trace.get("rewritten_query", "")),
                    sources=len(checks),
                    quotes_valid=bool(evidence)
                    and all(entry.quote_valid for entry in evidence),
                    state_contains=[
                        value
                        for value in turn.expected_state_contains
                        if state_contains(response.state, value)
                    ],
                )
            )

        detail = self.chat.get_detail(session.id)
        final_state = detail.state
        metrics = self._compute_metrics(scenario, turn_results, final_state, completed)
        result = ScenarioRunResult(
            scenario=scenario,
            session_id=session.id,
            metrics=metrics,
            turns=turn_results,
            final_state=final_state,
            error=error,
        )
        if error is None:
            self._save_result(result)
        return result

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(
        scenario: Scenario,
        turn_results: list[ScenarioTurnResult],
        final_state: TaskState,
        completed: int,
    ) -> ScenarioMetrics:
        superseded: set[str] = set()
        expected_facts: list[str] = []
        for turn in scenario.turns:
            superseded.update(turn.supersedes)
            for value in turn.expected_state_contains:
                if value not in expected_facts:
                    expected_facts.append(value)
        active_facts = [value for value in expected_facts if value not in superseded]
        retained = [value for value in active_facts if state_contains(final_state, value)]

        corrections_expected = 0
        corrections_applied = 0
        for turn in scenario.turns:
            if not turn.correction:
                continue
            corrections_expected += 1
            if all(
                state_contains(final_state, value)
                for value in turn.expect_active.values()
            ) and all(
                not state_contains(final_state, value) for value in turn.supersedes
            ):
                corrections_applied += 1

        followups = [turn for turn in scenario.turns if turn.expect_contextual_terms]
        followups_resolved = 0
        for turn, result in zip(scenario.turns, turn_results):
            if not turn.expect_contextual_terms:
                continue
            query = result.contextual_query.lower()
            if all(term.lower() in query for term in turn.expect_contextual_terms):
                followups_resolved += 1

        grounded_answers = sum(
            1 for result in turn_results if result.assistant_status == MSG_STATUS_ANSWERED
        )
        answers_with_sources = sum(
            1
            for result in turn_results
            if result.assistant_status == MSG_STATUS_ANSWERED and result.sources > 0
        )
        quotes_valid = sum(
            1
            for result in turn_results
            if result.assistant_status == MSG_STATUS_ANSWERED and result.quotes_valid
        )
        insufficient = sum(
            1 for result in turn_results if result.assistant_status == MSG_STATUS_INSUFFICIENT
        )
        correct_refusals = sum(
            1
            for turn, result in zip(scenario.turns, turn_results)
            if turn.expect_refusal and result.assistant_status == MSG_STATUS_INSUFFICIENT
        )

        return ScenarioMetrics(
            turns_total=len(scenario.turns),
            turns_completed=completed,
            goal_retained=bool(final_state.goal),
            facts_expected=len(active_facts),
            facts_retained=len(retained),
            memory_retention=f"{len(retained)}/{len(active_facts)}",
            corrections_expected=corrections_expected,
            corrections_applied=corrections_applied,
            contextual_followups=len(followups),
            contextual_followups_resolved=followups_resolved,
            grounded_answers=grounded_answers,
            answers_with_sources=answers_with_sources,
            quotes_valid=quotes_valid,
            insufficient_context_turns=insufficient,
            correct_refusals=correct_refusals,
        )
