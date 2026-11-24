"""Run reflection on a planted draft against the real model and show the catch.

    uv run python scripts/reflection_demo.py

The observations are real: the script calls get_employee_info and
check_request_eligibility on the server for employee E001's laptop. The draft is planted:
an approval that promises "your laptop will arrive Friday". Reflection must reject it,
revise it once, and confirm the revision.
"""

from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from equipment.agent import observation_from, server_params
from equipment.decisions import APPROVE, Draft, Observation, Request
from equipment.llm import LLMClient, load_environment
from equipment.observability import RunLogger
from equipment.reflection import Reflection

REQUEST = Request(
    employee_id="E001",
    item="laptop",
    reason="My laptop is over four years old and has become very slow.",
    submitted_on=date(2026, 11, 24),
)
PLANTED = Draft(
    decision=APPROVE,
    reason_code="REFRESH_DUE",
    ticket_id=None,
    draft_response=(
        "Good news: your laptop replacement is approved. Your laptop will arrive Friday, "
        "and IT will set it up for you the same day."
    ),
)


async def gather_observations(session: ClientSession) -> list[Observation]:
    calls: list[tuple[str, dict[str, Any]]] = [
        ("get_employee_info", {"employee_id": REQUEST.employee_id}),
        (
            "check_request_eligibility",
            {
                "employee_id": REQUEST.employee_id,
                "item": REQUEST.item,
                "as_of": REQUEST.submitted_on.isoformat(),
            },
        ),
    ]
    observations: list[Observation] = []
    for step, (tool, arguments) in enumerate(calls, start=1):
        result = await session.call_tool(tool, arguments)
        observations.append(
            Observation(step, tool, arguments, observation_from(result), result.is_error)
        )
    return observations


async def main() -> None:
    load_environment()
    with RunLogger() as run:
        logger = run.for_request()
        llm = LLMClient(logger)
        async with stdio_client(server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                observations = await gather_observations(session)

        print(f"run: {run.run_id} | request: {logger.request_id} | model: {llm.model}")
        print(f"\n{REQUEST.as_text()}")
        print(f"\nPlanted draft ({PLANTED.decision}, {PLANTED.reason_code}):")
        print(PLANTED.draft_response)

        outcome = await Reflection(llm, logger).run(REQUEST, PLANTED, observations)

        print(f"\noutcome: {outcome.verdict}")
        print(
            f"model calls: {llm.calls} | input tokens: {llm.total_input_tokens} | "
            f"output tokens: {llm.total_output_tokens} | cost: ${llm.total_cost_usd:.4f}"
        )
        print(f"log: {run.path}")


if __name__ == "__main__":
    asyncio.run(main())
