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


async def call(session: ClientSession, tool: str, **arguments: str) -> dict[str, Any]:
    result = await session.call_tool(tool, arguments)
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


@pytest.mark.integration
async def test_lists_tools_with_schemas() -> None:
    async with server_session() as session:
        tools = await session.list_tools()

    by_name = {tool.name: tool for tool in tools.tools}
    assert set(by_name) == {"get_employee_info", "get_policy_limits"}
    for tool in by_name.values():
        assert tool.description

    employee = by_name["get_employee_info"].input_schema
    assert employee["required"] == ["employee_id"]
    assert employee["properties"]["employee_id"]["type"] == "string"

    policy = by_name["get_policy_limits"].input_schema
    assert policy["required"] == ["role"]
    assert policy["properties"]["role"]["type"] == "string"


@pytest.mark.integration
async def test_known_employee() -> None:
    async with server_session() as session:
        result = await call(session, "get_employee_info", employee_id="E001")

    assert result["found"] is True
    assert result["employee_id"] == "E001"
    assert result["name"] == "Alice Morgan"
    assert result["role"] == "standard"
    assert result["policy_version"] == "2026.1"
    assert {rec["asset_tag"] for rec in result["equipment"]} == {"LT-0001", "KB-0001", "MS-0001"}


@pytest.mark.integration
async def test_unknown_employee() -> None:
    async with server_session() as session:
        result = await call(session, "get_employee_info", employee_id="E999")

    assert result["found"] is False
    assert result["error"] == "EMPLOYEE_NOT_FOUND"
    assert result["employee_id"] == "E999"


@pytest.mark.integration
async def test_lower_case_id_matches_through_server() -> None:
    async with server_session() as session:
        result = await call(session, "get_employee_info", employee_id="e001")

    assert result["found"] is True
    assert result["employee_id"] == "E001"


@pytest.mark.integration
async def test_policy_limits_known_role() -> None:
    async with server_session() as session:
        result = await call(session, "get_policy_limits", role="Contractor")

    assert result["found"] is True
    assert result["role"] == "contractor"
    assert result["limits"]["monitor"] == {"max_quantity": 1, "refresh_months": 36}
    assert result["not_eligible"] == ["laptop", "docking_station"]
    assert result["policy_version"] == "2026.1"


@pytest.mark.integration
async def test_policy_limits_unknown_role() -> None:
    async with server_session() as session:
        result = await call(session, "get_policy_limits", role="intern")

    assert result["found"] is False
    assert result["error"] == "ROLE_NOT_FOUND"
    assert result["known_roles"] == ["contractor", "engineer", "manager", "standard"]
