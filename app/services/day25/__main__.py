"""Day 25 scenario runner CLI.

Usage (from the project root)::

    python -m app.services.day25 list
    python -m app.services.day25 run scenario_01_diagnostics
    python -m app.services.day25 run-all

The runner uses the REAL Day 21 index and the configured generation provider
(``RAG_GENERATION_PROVIDER``). Unit tests never use this entry point; they
inject deterministic fakes instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from app.config import get_settings
from app.services.day25 import Day25EvaluationService


def _print_metrics(title: str, metrics) -> None:
    print(f"\n=== {title}")
    print(f"Turns:                {metrics.turns_completed} / {metrics.turns_total}")
    print(f"Goal retained:        {'PASS' if metrics.goal_retained else 'FAIL'}")
    print(f"Facts retained:       {metrics.memory_retention}")
    print(
        "Corrections applied:  "
        f"{metrics.corrections_applied} / {metrics.corrections_expected}"
    )
    print(
        "Contextual follow-ups: "
        f"{metrics.contextual_followups_resolved} / {metrics.contextual_followups}"
    )
    print(f"Grounded answers:     {metrics.grounded_answers}")
    print(
        "Answers with sources: "
        f"{metrics.answers_with_sources} / {metrics.grounded_answers}"
    )
    print(f"Quotes valid:         {metrics.quotes_valid} / {metrics.grounded_answers}")
    print(f"Insufficient context: {metrics.insufficient_context_turns}")
    print(f"Correct refusals:     {metrics.correct_refusals}")


async def _run(service: Day25EvaluationService, scenario_id: str) -> None:
    result = await service.run_scenario(scenario_id)
    _print_metrics(result.scenario.title, result.metrics)
    if result.error:
        print(f"ERROR: {result.error}")
    print("\nPer-turn:")
    for turn in result.turns:
        print(
            f"  {turn.index:>2}. [{turn.assistant_status}] "
            f"sources={turn.sources} quotes_valid={turn.quotes_valid}"
        )
        print(f"      user:       {turn.user}")
        print(f"      contextual: {turn.contextual_query}")


async def _main(args: argparse.Namespace) -> None:
    settings = get_settings()
    service = Day25EvaluationService(settings)
    if args.command == "list":
        for scenario in service.scenarios():
            print(f"{scenario.id}: {scenario.title} ({len(scenario.turns)} turns)")
        return
    if args.command == "run":
        await _run(service, args.scenario_id)
        return
    if args.command == "run-all":
        for scenario in service.scenarios():
            await _run(service, scenario.id)
        return
    if args.command == "results":
        print(json.dumps(service.results(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 25 scenario runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("scenario_id")
    sub.add_parser("run-all")
    sub.add_parser("results")
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
