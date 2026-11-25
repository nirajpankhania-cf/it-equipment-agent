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
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from equipment import core
from equipment.store import Store, load_store

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("equipment.server")

mcp = MCPServer("it-equipment")

CATALOGUE_ITEMS = "laptop, monitor, keyboard, mouse, headset, docking_station"
ROLES = "standard, engineer, manager, contractor"

EmployeeId = Annotated[
    str,
    Field(
        description=(
            "The employee's ID as it appears in the request, for example E001. "
            "Matching ignores case and surrounding spaces."
        )
    ),
]

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
def get_employee_info(employee_id: EmployeeId) -> dict[str, Any]:
    """Look up an employee by ID. Call this first for every request. Returns role, tenure
    in months, and equipment held with each item's age in months. Returns found=false
    with error EMPLOYEE_NOT_FOUND if the ID is not on file."""
    result = core.get_employee_info(employee_id, store=get_store())
    log.info("get_employee_info employee_id=%r found=%s", employee_id, result["found"])
    return result


@mcp.tool()
def get_policy_limits(
    role: Annotated[
        str,
        Field(
            description=(
                f"One of the four roles: {ROLES}, for example engineer. Matching ignores case."
            )
        ),
    ],
) -> dict[str, Any]:
    """Return the equipment entitlement for a role: for each item the maximum quantity
    and the refresh interval in months, plus the items the role is not eligible for. Use
    it to see a role's full entitlement, for example to cite the rule in a denial.
    Returns found=false with error ROLE_NOT_FOUND if the role has no policy."""
    result = core.get_policy_limits(role, store=get_store())
    log.info("get_policy_limits role=%r found=%s", role, result["found"])
    return result


@mcp.tool()
def check_request_eligibility(
    employee_id: EmployeeId,
    item: Annotated[
        str,
        Field(
            description=(
                f"The equipment requested, one of the six catalogue items: {CATALOGUE_ITEMS}, "
                "for example laptop. Anything else returns ITEM_NOT_IN_CATALOGUE."
            )
        ),
    ],
    as_of: Annotated[
        date | None,
        Field(
            description=(
                "The date to evaluate as of, as an ISO date (YYYY-MM-DD), for example "
                "2026-11-24. Defaults to today."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """Apply the equipment policy for one employee and one item. Use it to decide one
    specific request; it does not see the request text, so it cannot judge claimed
    exceptions. Returns status ELIGIBLE, NOT_ELIGIBLE, or REVIEW_REQUIRED with a
    reason_code and, where relevant, the next eligible date. REVIEW_REQUIRED must be
    escalated, never decided. Pass as_of as an ISO date (YYYY-MM-DD) to evaluate as of
    the request's submission date; defaults to today."""
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


FLAG_DESCRIPTION = (
    "Escalate a request to a human reviewer. Call it once per request when the request "
    "must be escalated; it writes an audit entry and is not idempotent. Returns a ticket "
    "ID that must be quoted in the response. request is the full request as submitted "
    "(employee ID, item, reason, and submission date), so the reviewer sees what was "
    f"asked. reason_code must be one of {', '.join(core.MODEL_ESCALATION_REASON_CODES)}. "
    "reason is a short plain-language explanation of why a human is needed. Neither "
    "request nor reason may be empty. Pass the request_id so the audit entry can be "
    "matched to the trace."
)


@mcp.tool(description=FLAG_DESCRIPTION)
def flag_for_human_review(
    employee_id: EmployeeId,
    request: Annotated[
        str,
        Field(
            description=(
                "The full request as submitted: employee ID, item, reason, and submission "
                "date, for example 'Employee ID: E003 / Item requested: laptop / Reason "
                "given: The screen is cracked. / Submitted on: 2026-11-24'."
            )
        ),
    ],
    item: Annotated[
        str,
        Field(description="The item requested, as written in the request, for example laptop."),
    ],
    reason_code: Annotated[
        str,
        Field(
            description=(
                "Why a human is needed, one of the escalation reason codes, for example "
                "EXCEPTION_CLAIMED."
            )
        ),
    ],
    reason: Annotated[
        str,
        Field(
            description=(
                "A short plain-language explanation, for example 'Requester reports a cracked "
                "screen; damage cannot be verified from records'."
            )
        ),
    ],
    request_id: Annotated[
        str | None,
        Field(
            description=(
                "The agent's request ID, so the audit entry can be matched to the trace, "
                "for example req-20261124-151306-81ceff."
            )
        ),
    ] = None,
) -> dict[str, Any]:
    """See FLAG_DESCRIPTION, which is what the model reads. The reason-code list there is
    generated from core.MODEL_ESCALATION_REASON_CODES; the tool itself also accepts the
    internal codes in core.INTERNAL_ESCALATION_REASON_CODES."""
    result = core.flag_for_human_review(
        employee_id,
        request=request,
        item=item,
        reason_code=reason_code,
        reason=reason,
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
