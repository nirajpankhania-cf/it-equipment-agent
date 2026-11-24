import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp import types as mcp_types

from equipment.agent import Request, run_one_step, to_anthropic_tools
from equipment.llm import LLMClient, ModelPrice, Pricing
from equipment.observability import RequestLogger, RunLogger
from tests.agent.fake_model import FakeModel, text_message, tool_use_message

PRICING = Pricing({"claude-fable-5-2": ModelPrice(8.0, 40.0)}, max_cost_per_run_usd=5.0)
REQUEST = Request("E001", "laptop", "It is slow.", date(2026, 11, 24))


class FakeSession:
    """Stands in for the MCP ClientSession: records calls and returns a scripted result."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
        self.calls.append((name, arguments))
        return mcp_types.CallToolResult(content=[], structured_content=self.result)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def logger(tmp_path: Path) -> RequestLogger:
    return RunLogger(run_id="r1", logs_dir=tmp_path).for_request("req-1")


def test_request_text_has_all_four_fields() -> None:
    text = REQUEST.as_text()
    assert "Employee ID: E001" in text
    assert "Item requested: laptop" in text
    assert "Reason given: It is slow." in text
    assert "Submitted on: 2026-11-24" in text


def test_to_anthropic_tools_converts_name_description_and_schema() -> None:
    schema = {"type": "object", "properties": {"employee_id": {"type": "string"}}}
    tools = to_anthropic_tools(
        [
            mcp_types.Tool(name="get_employee_info", description="Look up", inputSchema=schema),
            mcp_types.Tool(name="bare", inputSchema={"type": "object"}),
        ]
    )
    assert tools == [
        {"name": "get_employee_info", "description": "Look up", "input_schema": schema},
        {"name": "bare", "description": "", "input_schema": {"type": "object"}},
    ]


async def test_one_step_executes_the_tool_and_logs_the_trace(
    logger: RequestLogger, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeModel(
        [
            tool_use_message(
                "get_employee_info",
                {"employee_id": "E001"},
                thought="I will look up the employee first.",
            )
        ]
    )
    llm = LLMClient(logger, model="claude-fable-5-2", pricing=PRICING, create=fake.create)
    session = FakeSession({"found": True, "role": "standard"})
    tools: list[Any] = [{"name": "get_employee_info", "description": "", "input_schema": {}}]

    observation = await run_one_step(REQUEST, logger, llm, session, tools)  # type: ignore[arg-type]

    assert observation == {"found": True, "role": "standard"}
    assert session.calls == [("get_employee_info", {"employee_id": "E001"})]
    assert fake.calls[0]["tools"] == tools
    assert fake.calls[0]["messages"][0]["content"] == REQUEST.as_text()

    out = capsys.readouterr().out
    assert "Thought: I will look up the employee first." in out
    assert 'Action: get_employee_info({"employee_id": "E001"})' in out
    assert 'Observation: {\n  "found": true' in out

    lines = read(logger.run.path)
    events = [line["event"] for line in lines]
    assert events == ["model_call", "span", "react_step", "span", "observation"]
    assert all(line["request_id"] == "req-1" for line in lines)
    spans = {line["name"]: line for line in lines if line["event"] == "span"}
    assert spans["tool_call"]["parent_span_id"] == spans["model_call"]["span_id"]
    assert spans["tool_call"]["tool"] == "get_employee_info"


async def test_one_step_with_no_tool_call_warns(
    logger: RequestLogger, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeModel([text_message("I cannot help.")])
    llm = LLMClient(logger, model="claude-fable-5-2", pricing=PRICING, create=fake.create)
    session = FakeSession({})

    observation = await run_one_step(REQUEST, logger, llm, session, [])  # type: ignore[arg-type]

    assert observation is None
    assert session.calls == []
    assert "No tool call made" in capsys.readouterr().out
    assert any(line["event"] == "no_tool_call" for line in read(logger.run.path))


async def test_missing_thought_is_logged(logger: RequestLogger) -> None:
    fake = FakeModel([tool_use_message("get_employee_info", {"employee_id": "E001"}, thought=None)])
    llm = LLMClient(logger, model="claude-fable-5-2", pricing=PRICING, create=fake.create)
    session = FakeSession({"found": True})

    await run_one_step(REQUEST, logger, llm, session, [])  # type: ignore[arg-type]

    assert any(line["event"] == "missing_thought" for line in read(logger.run.path))
