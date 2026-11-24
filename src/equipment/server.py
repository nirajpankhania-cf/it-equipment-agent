"""MCP server exposing the core functions as tools. No business logic lives here.

Run over stdio:

    uv run python -m equipment.server

All logging goes to stderr. Nothing may print to stdout, because stdout carries the
protocol messages and a stray print corrupts the connection.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from equipment import core
from equipment.store import Store, load_store

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("equipment.server")

mcp = MCPServer("it-equipment")

# Fixed at startup; the tool takes no path argument (plan section 9).
AUDIT_PATH = Path(os.environ.get("EQUIPMENT_AUDIT_PATH", str(core.DEFAULT_AUDIT_PATH)))


@cache
def get_store() -> Store:
    """Load and validate the data once, on first use. Fails loudly if the data is bad."""
    store = load_store()
    log.info(
        "data loaded: %d employees, policy version %s, data version %s",
        len(store.employees_by_id),
        store.policy_version,
        store.data_version,
    )
    return store


@mcp.tool()
def get_employee_info(employee_id: str) -> dict[str, Any]:
    """Look up an employee by ID. Returns role, tenure in months, and equipment held with
    each item's age in months. Returns found=false with error EMPLOYEE_NOT_FOUND if the
    ID is not on file."""
    result = core.get_employee_info(employee_id, store=get_store())
    log.info("get_employee_info employee_id=%r found=%s", employee_id, result["found"])
    return result


@mcp.tool()
def get_policy_limits(role: str) -> dict[str, Any]:
    """Return the equipment entitlement for a role: for each item the maximum quantity
    and the refresh interval in months, plus the items the role is not eligible for.
    Returns found=false with error ROLE_NOT_FOUND if the role has no policy."""
    result = core.get_policy_limits(role, store=get_store())
    log.info("get_policy_limits role=%r found=%s", role, result["found"])
    return result


@mcp.tool()
def check_request_eligibility(
    employee_id: str, item: str, as_of: date | None = None
) -> dict[str, Any]:
    """Apply the equipment policy for one employee and one item. Returns status ELIGIBLE,
    NOT_ELIGIBLE, or REVIEW_REQUIRED with a reason_code and, where relevant, the next
    eligible date. REVIEW_REQUIRED must be escalated, never decided. Pass as_of as an
    ISO date (YYYY-MM-DD) to evaluate as of the request's submission date; defaults to
    today."""
    result = core.check_request_eligibility(employee_id, item, store=get_store(), as_of=as_of)
    log.info(
        "check_request_eligibility employee_id=%r item=%r as_of=%s status=%s reason=%s",
        employee_id,
        item,
        result["as_of"],
        result["status"],
        result["reason_code"],
    )
    return result


@mcp.tool()
def flag_for_human_review(
    employee_id: str, item: str, reason_code: str, reason: str, request_id: str | None = None
) -> dict[str, Any]:
    """Escalate a request to a human reviewer. Writes an audit entry and returns a ticket
    ID that must be quoted in the response. reason_code must be one of EMPLOYEE_NOT_FOUND,
    ITEM_NOT_IN_CATALOGUE, EXCEPTION_CLAIMED, BORDERLINE_TIMING, DATA_INCOMPLETE,
    REFLECTION_FAILED, GUARDRAIL_BLOCKED. reason is a short plain-language explanation and
    must not be empty. Pass the request_id so the audit entry can be matched to the trace."""
    result = core.flag_for_human_review(
        employee_id,
        item,
        reason_code,
        reason,
        store=get_store(),
        request_id=request_id,
        audit_path=AUDIT_PATH,
    )
    if result["flagged"]:
        log.warning(
            "flag_for_human_review ticket=%s employee_id=%r item=%r reason_code=%s request_id=%s",
            result["ticket_id"],
            employee_id,
            item,
            result["reason_code"],
            request_id,
        )
    else:
        log.warning("flag_for_human_review rejected: %s", result["error"])
    return result


def main() -> None:
    log.info("starting MCP server over stdio; audit file %s", AUDIT_PATH)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
