"""Start the real MCP server over stdio and exercise it through an MCP client."""

import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

AS_OF = "2026-11-23"


@asynccontextmanager
async def server_session(audit_path: Path | None = None) -> AsyncIterator[ClientSession]:
    env = dict(os.environ)
    if audit_path is not None:
        env["EQUIPMENT_AUDIT_PATH"] = str(audit_path)
    params = StdioServerParameters(command=sys.executable, args=["-m", "equipment.server"], env=env)
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
    assert set(by_name) == {
        "get_employee_info",
        "get_policy_limits",
        "check_request_eligibility",
        "flag_for_human_review",
    }
    for tool in by_name.values():
        assert tool.description

    employee = by_name["get_employee_info"].input_schema
    assert employee["required"] == ["employee_id"]
    assert employee["properties"]["employee_id"]["type"] == "string"

    policy = by_name["get_policy_limits"].input_schema
    assert policy["required"] == ["role"]
    assert policy["properties"]["role"]["type"] == "string"

    eligibility = by_name["check_request_eligibility"].input_schema
    assert eligibility["required"] == ["employee_id", "item"]
    assert "as_of" in eligibility["properties"]

    flag = by_name["flag_for_human_review"].input_schema
    assert flag["required"] == ["employee_id", "request", "item", "reason_code", "reason"]
    assert "request_id" in flag["properties"]
    assert "audit_path" not in flag["properties"]


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


@pytest.mark.integration
async def test_eligibility_eligible_by_age() -> None:
    async with server_session() as session:
        result = await call(
            session, "check_request_eligibility", employee_id="E001", item="laptop", as_of=AS_OF
        )

    assert result["status"] == "ELIGIBLE"
    assert result["reason_code"] == "REFRESH_DUE"
    assert result["as_of"] == AS_OF
    assert result["next_eligible_date"] == "2026-03-14"


@pytest.mark.integration
async def test_eligibility_denied_with_next_date() -> None:
    async with server_session() as session:
        result = await call(
            session, "check_request_eligibility", employee_id="E002", item="monitor", as_of=AS_OF
        )

    assert result["status"] == "NOT_ELIGIBLE"
    assert result["reason_code"] == "WITHIN_REFRESH_INTERVAL"
    assert result["next_eligible_date"] == "2028-09-15"


@pytest.mark.integration
async def test_eligibility_unknown_employee_requires_review() -> None:
    async with server_session() as session:
        result = await call(
            session, "check_request_eligibility", employee_id="E999", item="headset", as_of=AS_OF
        )

    assert result["found"] is False
    assert result["status"] == "REVIEW_REQUIRED"
    assert result["reason_code"] == "EMPLOYEE_NOT_FOUND"


@pytest.mark.integration
async def test_eligibility_as_of_defaults_to_today() -> None:
    async with server_session() as session:
        result = await call(session, "check_request_eligibility", employee_id="E005", item="laptop")

    assert result["status"] == "NOT_ELIGIBLE"
    assert result["reason_code"] == "ROLE_NOT_ENTITLED"
    assert result["as_of"] == date.today().isoformat()


@pytest.mark.integration
async def test_eligibility_rejects_malformed_date() -> None:
    async with server_session() as session:
        result = await session.call_tool(
            "check_request_eligibility",
            {"employee_id": "E001", "item": "laptop", "as_of": "not-a-date"},
        )

    assert result.is_error


@pytest.mark.integration
async def test_flag_writes_audit_entry_in_server_process(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "escalations.jsonl"
    async with server_session(audit_path) as session:
        result = await call(
            session,
            "flag_for_human_review",
            employee_id="E003",
            request="My laptop screen is cracked.",
            item="laptop",
            reason_code="EXCEPTION_CLAIMED",
            reason="Cracked screen reported",
            request_id="req-int-1",
        )

    assert result["flagged"] is True
    assert result["ticket_id"].startswith("ESC-")

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["ticket_id"] == result["ticket_id"]
    assert entry["request_id"] == "req-int-1"
    assert entry["employee_id"] == "E003"
    assert entry["request"] == "My laptop screen is cracked."
    assert entry["reason_code"] == "EXCEPTION_CLAIMED"


@pytest.mark.integration
async def test_flag_rejects_empty_reason_without_writing(tmp_path: Path) -> None:
    audit_path = tmp_path / "escalations.jsonl"
    async with server_session(audit_path) as session:
        result = await call(
            session,
            "flag_for_human_review",
            employee_id="E003",
            request="My laptop screen is cracked.",
            item="laptop",
            reason_code="EXCEPTION_CLAIMED",
            reason="  ",
        )

    assert result["flagged"] is False
    assert result["error"] == "EMPTY_REASON"
    assert not audit_path.exists()
