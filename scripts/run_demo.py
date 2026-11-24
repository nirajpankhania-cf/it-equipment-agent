"""Run the evaluation cases against the real model and compare expected with actual.

    uv run python scripts/run_demo.py            # all five cases
    uv run python scripts/run_demo.py --case 3   # one case by number or id prefix

Each case is a separate request under one run, so the log file holds every trace. The
checks follow plan section 8: an approval needs an ELIGIBLE result and no flag call; a
denial needs no flag call and the next eligible date in the draft; an escalation needs
exactly one flag call, the expected reason code, and the ticket ID in the draft.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from equipment.agent import ESCALATE, Agent, AgentResult, Request, server_params, to_anthropic_tools
from equipment.llm import LLMClient, load_environment
from equipment.observability import RunLogger

CASES_PATH = Path(__file__).resolve().parents[1] / "evals" / "cases.json"
PROMISE_WORDS = ("approved", "will be delivered", "will arrive", "has been ordered")


@dataclass
class Outcome:
    case_id: str
    expected: dict[str, Any]
    result: AgentResult
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def load_cases(selector: str | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]
    if selector is None:
        return cases
    chosen = [c for c in cases if c["id"] == selector or c["id"].startswith(f"{selector}-")]
    if not chosen:
        sys.exit(f"no case matching '{selector}'; ids: {[c['id'] for c in cases]}")
    return chosen


def check(expected: dict[str, Any], result: AgentResult) -> list[str]:
    problems: list[str] = []
    if result.decision != expected["decision"]:
        problems.append(f"decision {result.decision} != {expected['decision']}")
    flags = result.flag_calls
    draft = result.draft_response.lower()

    if expected["decision"] == ESCALATE:
        if expected.get("reason_code") and result.reason_code != expected["reason_code"]:
            problems.append(f"reason_code {result.reason_code} != {expected['reason_code']}")
        if len(flags) != 1:
            problems.append(f"{len(flags)} flag calls, expected 1")
        if not result.ticket_id or result.ticket_id not in result.draft_response:
            problems.append("ticket ID missing from draft")
        if any(word in draft for word in PROMISE_WORDS):
            problems.append("draft promises an outcome")
    else:
        if flags:
            problems.append(f"{len(flags)} flag calls, expected 0")
        if result.guardrail:
            problems.append(f"guardrail {result.guardrail}")
        next_date = expected.get("next_eligible_date")
        if next_date and next_date not in result.draft_response:
            problems.append(f"next eligible date {next_date} missing from draft")
    return problems


async def run_cases(cases: list[dict[str, Any]]) -> list[Outcome]:
    load_environment()
    outcomes: list[Outcome] = []
    with RunLogger() as run:
        print(f"run: {run.run_id}")
        async with stdio_client(server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = to_anthropic_tools((await session.list_tools()).tools)
                for case in cases:
                    print(f"\n{'=' * 70}\ncase: {case['id']}")
                    logger = run.for_request()
                    llm = LLMClient(logger)
                    request = Request(
                        employee_id=case["request"]["employee_id"],
                        item=case["request"]["item"],
                        reason=case["request"]["reason"],
                        submitted_on=date.fromisoformat(case["request"]["submitted_on"]),
                    )
                    started = time.perf_counter()
                    result = await Agent(logger, llm, session, tools).run(request)
                    outcomes.append(
                        Outcome(
                            case_id=case["id"],
                            expected=case["expected"],
                            result=result,
                            calls=llm.calls,
                            input_tokens=llm.total_input_tokens,
                            output_tokens=llm.total_output_tokens,
                            cost_usd=llm.total_cost_usd,
                            latency_s=round(time.perf_counter() - started, 1),
                            problems=check(case["expected"], result),
                        )
                    )
        print(f"\nlog: {run.path}")
    return outcomes


def print_summary(outcomes: list[Outcome]) -> None:
    print(f"\n{'=' * 70}\nresults")
    header = (
        f"{'case':<30} {'expected':<9} {'actual':<9} {'reason code':<22} {'steps':>5} {'ok':>3}"
    )
    print(header)
    print("-" * len(header))
    for o in outcomes:
        print(
            f"{o.case_id:<30} {o.expected['decision']:<9} {o.result.decision:<9} "
            f"{o.result.reason_code:<22} {o.result.steps:>5} {'yes' if o.ok else 'NO':>3}"
        )
        for problem in o.problems:
            print(f"    - {problem}")

    print("\ncost")
    header = f"{'case':<30} {'calls':>5} {'in tok':>8} {'out tok':>8} {'cost $':>8} {'secs':>6}"
    print(header)
    print("-" * len(header))
    for o in outcomes:
        print(
            f"{o.case_id:<30} {o.calls:>5} {o.input_tokens:>8} {o.output_tokens:>8} "
            f"{o.cost_usd:>8.4f} {o.latency_s:>6.1f}"
        )
    print("-" * len(header))
    print(
        f"{'total':<30} {sum(o.calls for o in outcomes):>5} "
        f"{sum(o.input_tokens for o in outcomes):>8} {sum(o.output_tokens for o in outcomes):>8} "
        f"{sum(o.cost_usd for o in outcomes):>8.4f} {sum(o.latency_s for o in outcomes):>6.1f}"
    )
    passed = sum(o.ok for o in outcomes)
    print(f"\n{passed} of {len(outcomes)} cases reached the expected outcome")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", help="case id or leading number, e.g. 3")
    args = parser.parse_args()
    outcomes = asyncio.run(run_cases(load_cases(args.case)))
    print_summary(outcomes)
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
