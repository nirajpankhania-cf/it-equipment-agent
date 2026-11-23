"""Start the real MCP server over stdio and exercise it through an MCP client."""

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def server_session() -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "equipment.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call(session: ClientSession, employee_id: str) -> dict[str, Any]:
    result = await session.call_tool("get_employee_info", {"employee_id": employee_id})
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


@pytest.mark.integration
async def test_lists_get_employee_info_tool() -> None:
    async with server_session() as session:
        tools = await session.list_tools()

    names = [tool.name for tool in tools.tools]
    assert names == ["get_employee_info"]

    tool = tools.tools[0]
    assert tool.description
    assert tool.input_schema["required"] == ["employee_id"]
    assert tool.input_schema["properties"]["employee_id"]["type"] == "string"


@pytest.mark.integration
async def test_known_employee() -> None:
    async with server_session() as session:
        result = await call(session, "E001")

    assert result["found"] is True
    assert result["employee_id"] == "E001"
    assert result["name"] == "Alice Morgan"
    assert result["role"] == "standard"
    assert result["policy_version"] == "2026.1"
    assert {rec["asset_tag"] for rec in result["equipment"]} == {"LT-0001", "KB-0001", "MS-0001"}


@pytest.mark.integration
async def test_unknown_employee() -> None:
    async with server_session() as session:
        result = await call(session, "E999")

    assert result["found"] is False
    assert result["error"] == "EMPLOYEE_NOT_FOUND"
    assert result["employee_id"] == "E999"


@pytest.mark.integration
async def test_lower_case_id_matches_through_server() -> None:
    async with server_session() as session:
        result = await call(session, "e001")

    assert result["found"] is True
    assert result["employee_id"] == "E001"
