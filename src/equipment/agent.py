"""The agent: an MCP client plus the model.

Milestone 4: connect to the server, list its tools, convert them to the model's tool
format, and for one hardcoded request make a single tool call and print the
Thought, Action, and Observation. No loop yet.

    uv run python -m equipment.agent
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any

from anthropic.types import MessageParam, ToolParam
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client

from equipment.llm import LLMClient, load_environment
from equipment.observability import RequestLogger, RunLogger

SYSTEM_PROMPT = """You are an IT equipment request handler. You decide whether an employee's \
equipment request should be approved, denied, or escalated to a human, using only the \
tools provided. Never guess facts about the employee or the policy.

Before every tool call, state in one or two sentences why you are calling it.

Always begin by calling get_employee_info with the employee ID from the request."""


@dataclass(frozen=True)
class Request:
    employee_id: str
    item: str
    reason: str
    submitted_on: date

    def as_text(self) -> str:
        return (
            f"Employee ID: {self.employee_id}\n"
            f"Item requested: {self.item}\n"
            f"Reason given: {self.reason}\n"
            f"Submitted on: {self.submitted_on.isoformat()}"
        )


DEMO_REQUEST = Request(
    employee_id="E001",
    item="laptop",
    reason="My laptop is over four years old and has become very slow.",
    submitted_on=date.today(),
)


def to_anthropic_tools(tools: list[mcp_types.Tool]) -> list[ToolParam]:
    """Convert MCP tool listings to the Messages API tool format."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def observation_from(result: mcp_types.CallToolResult) -> Any:
    if result.structured_content is not None:
        return result.structured_content
    return [block.text for block in result.content if block.type == "text"]


async def run_one_step(
    request: Request,
    logger: RequestLogger,
    llm: LLMClient,
    session: ClientSession,
    tools: list[ToolParam],
) -> Any:
    """Ask the model for its first action, execute that one tool call, print the trace."""
    messages: list[MessageParam] = [{"role": "user", "content": request.as_text()}]
    with logger.span("model_call", purpose="react", step=1) as model_span:
        response = await llm.complete(
            purpose="react", step=1, system=SYSTEM_PROMPT, messages=messages, tools=tools
        )
    model_span.fields["stop_reason"] = response.stop_reason

    thought = " ".join(b.text for b in response.content if b.type == "text").strip()
    tool_uses = [b for b in response.content if b.type == "tool_use"]
    if thought:
        print(f"Thought: {thought}")
    else:
        print("Thought: (none stated)")
        logger.log("missing_thought", level="WARNING", step=1)

    if not tool_uses:
        print(f"No tool call made (stop_reason={response.stop_reason}).")
        logger.log("no_tool_call", level="WARNING", step=1, stop_reason=response.stop_reason)
        return None

    call = tool_uses[0]
    arguments = dict(call.input) if isinstance(call.input, dict) else {}
    print(f"Action: {call.name}({json.dumps(arguments)})")
    logger.log("react_step", step=1, thought=thought, tool=call.name, arguments=arguments)

    with logger.span(
        "tool_call", parent_span_id=model_span.span_id, tool=call.name, arguments=arguments
    ) as tool_span:
        result = await session.call_tool(call.name, arguments)
    observation = observation_from(result)
    tool_span.fields["is_error"] = result.is_error
    print(f"Observation: {json.dumps(observation, indent=2)}")
    logger.log("observation", step=1, tool=call.name, is_error=result.is_error, result=observation)
    return observation


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable, args=["-m", "equipment.server"], env=dict(os.environ)
    )


async def main(request: Request = DEMO_REQUEST) -> None:
    load_environment()
    with RunLogger() as run:
        logger = run.for_request()
        logger.log(
            "request_received",
            employee_id=request.employee_id,
            item=request.item,
            request_text=request.as_text(),
        )
        llm = LLMClient(logger)

        async with stdio_client(server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = to_anthropic_tools(listed.tools)
                names = [tool["name"] for tool in tools]
                logger.log("tools_listed", tools=names)

                print(f"run: {run.run_id}")
                print(f"request: {logger.request_id}")
                print(f"model: {llm.model}")
                print(f"tools: {names}")
                print(f"\n{request.as_text()}\n")
                await run_one_step(request, logger, llm, session, tools)

        print(
            f"\nmodel calls: {llm.calls} | input tokens: {llm.total_input_tokens} | "
            f"output tokens: {llm.total_output_tokens} | cost: ${llm.total_cost_usd:.4f}"
        )
        print(f"log: {run.path}")


if __name__ == "__main__":
    asyncio.run(main())
