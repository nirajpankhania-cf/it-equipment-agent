"""The tool functions as plain Python. server.py wraps these and adds nothing.

Errors are returned as structured results, not raised, so the agent can read the
outcome in an Observation and act on it.
"""

from __future__ import annotations

import calendar
import json
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from equipment.models import Employee
from equipment.store import Store, normalize_employee_id, normalize_name

# Escalation reason codes (docs/requirements.md section 5).
EMPLOYEE_NOT_FOUND = "EMPLOYEE_NOT_FOUND"
ITEM_NOT_IN_CATALOGUE = "ITEM_NOT_IN_CATALOGUE"
EXCEPTION_CLAIMED = "EXCEPTION_CLAIMED"
BORDERLINE_TIMING = "BORDERLINE_TIMING"
DATA_INCOMPLETE = "DATA_INCOMPLETE"
REFLECTION_FAILED = "REFLECTION_FAILED"
GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
ESCALATION_REASON_CODES = frozenset(
    {
        EMPLOYEE_NOT_FOUND,
        ITEM_NOT_IN_CATALOGUE,
        EXCEPTION_CLAIMED,
        BORDERLINE_TIMING,
        DATA_INCOMPLETE,
        REFLECTION_FAILED,
        GUARDRAIL_BLOCKED,
    }
)

# flag_for_human_review error codes.
EMPTY_REQUEST = "EMPTY_REQUEST"
EMPTY_REASON = "EMPTY_REASON"
INVALID_REASON_CODE = "INVALID_REASON_CODE"

# Other structured-result codes.
ROLE_NOT_FOUND = "ROLE_NOT_FOUND"
ROLE_NOT_ENTITLED = "ROLE_NOT_ENTITLED"
WITHIN_REFRESH_INTERVAL = "WITHIN_REFRESH_INTERVAL"
BELOW_MAX_QUANTITY = "BELOW_MAX_QUANTITY"
REFRESH_DUE = "REFRESH_DUE"

# Eligibility statuses. The agent may approve only on ELIGIBLE.
ELIGIBLE = "ELIGIBLE"
NOT_ELIGIBLE = "NOT_ELIGIBLE"
REVIEW_REQUIRED = "REVIEW_REQUIRED"

BORDERLINE_DAYS = 30

DEFAULT_AUDIT_PATH = Path("audit") / "escalations.jsonl"


def months_between(start: date, end: date) -> int:
    """Whole calendar months from ``start`` to ``end``. Negative if ``end`` is earlier."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months


def add_months(day: date, months: int) -> date:
    """Add whole months, clamping the day to the end of the target month."""
    month_index = day.month - 1 + months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(day.day, last_day))


def normalize_item(raw: str, catalogue: list[str]) -> str | None:
    """Map a requested item onto a catalogue name, or return None.

    Handles case, surrounding spaces, spaces or hyphens for underscores, and a plain
    plural. Anything looser is for the agent to resolve or escalate.
    """
    name = normalize_name(raw).replace("-", "_").replace(" ", "_")
    if name in catalogue:
        return name
    if name.endswith("s") and name[:-1] in catalogue:
        return name[:-1]
    return None


def get_employee_info(
    employee_id: str, *, store: Store, as_of: date | None = None
) -> dict[str, Any]:
    """Look up an employee by ID. Lookup is case-insensitive and ignores surrounding spaces.

    Tenure and equipment age are computed against ``as_of`` (default today) so the
    result cannot go stale.
    """
    as_of = as_of or date.today()
    normalized = normalize_employee_id(employee_id)
    employee = store.get_employee(normalized) if normalized else None
    if employee is None:
        return {
            "found": False,
            "error": EMPLOYEE_NOT_FOUND,
            "employee_id": normalized,
            "message": f"No employee with ID '{employee_id}' is on file",
            "policy_version": store.policy_version,
        }

    equipment = [
        {
            "item": rec.item,
            "asset_tag": rec.asset_tag,
            "issued_date": rec.issued_date.isoformat() if rec.issued_date else None,
            "age_months": months_between(rec.issued_date, as_of) if rec.issued_date else None,
            "data_quality": rec.data_quality,
        }
        for rec in employee.equipment
    ]
    return {
        "found": True,
        "employee_id": employee.employee_id,
        "name": employee.name,
        "role": employee.role,
        "start_date": employee.start_date.isoformat(),
        "tenure_months": months_between(employee.start_date, as_of),
        "equipment": equipment,
        "as_of": as_of.isoformat(),
        "policy_version": store.policy_version,
        "data_version": store.data_version,
    }


def get_policy_limits(role: str, *, store: Store) -> dict[str, Any]:
    """Return the entitlement for every catalogue item under ``role``.

    Role lookup is case-insensitive and ignores surrounding spaces. Items the role is
    not entitled to are listed under ``not_eligible`` so the agent can see a definite no.
    """
    normalized = normalize_name(role)
    limits = store.policies_file.policies.get(normalized) if normalized else None
    if limits is None:
        return {
            "found": False,
            "error": ROLE_NOT_FOUND,
            "role": normalized,
            "message": f"No policy for role '{role}'",
            "known_roles": sorted(store.policies_file.policies),
            "policy_version": store.policy_version,
        }

    entitlements = {
        item: {"max_quantity": limit.max_quantity, "refresh_months": limit.refresh_months}
        for item, limit in limits.items()
    }
    return {
        "found": True,
        "role": normalized,
        "limits": entitlements,
        "not_eligible": [item for item in store.catalogue if item not in entitlements],
        "policy_version": store.policy_version,
        "effective_date": store.policies_file.effective_date.isoformat(),
    }


def check_request_eligibility(
    employee_id: str, item: str, *, store: Store, as_of: date | None = None
) -> dict[str, Any]:
    """Apply the policy rules for one employee and one item as of a date.

    Returns ``status`` ELIGIBLE, NOT_ELIGIBLE, or REVIEW_REQUIRED with a ``reason_code``.
    REVIEW_REQUIRED carries one of the escalation codes and must never be approved
    automatically. The request text is not seen here, so EXCEPTION_CLAIMED is the
    agent's call.
    """
    as_of = as_of or date.today()
    base: dict[str, Any] = {
        "employee_id": normalize_employee_id(employee_id),
        "item": item,
        "as_of": as_of.isoformat(),
        "policy_version": store.policy_version,
    }

    employee = store.get_employee(employee_id) if employee_id.strip() else None
    if employee is None:
        return {
            **base,
            "found": False,
            "error": EMPLOYEE_NOT_FOUND,
            "status": REVIEW_REQUIRED,
            "reason_code": EMPLOYEE_NOT_FOUND,
            "message": f"No employee with ID '{employee_id}' is on file",
        }
    base["found"] = True
    base["employee_id"] = employee.employee_id
    base["role"] = employee.role

    catalogue_item = normalize_item(item, store.catalogue)
    if catalogue_item is None:
        return {
            **base,
            "error": ITEM_NOT_IN_CATALOGUE,
            "status": REVIEW_REQUIRED,
            "reason_code": ITEM_NOT_IN_CATALOGUE,
            "message": f"'{item}' is not in the equipment catalogue",
            "catalogue": list(store.catalogue),
        }
    base["item"] = catalogue_item

    limit = store.policies_file.policies[normalize_name(employee.role)].get(catalogue_item)
    if limit is None:
        return {
            **base,
            "status": NOT_ELIGIBLE,
            "reason_code": ROLE_NOT_ENTITLED,
            "message": f"Role '{employee.role}' is not entitled to a {catalogue_item}",
            "next_eligible_date": None,
        }
    base["max_quantity"] = limit.max_quantity
    base["refresh_months"] = limit.refresh_months

    held = _held_units(employee, catalogue_item)
    base["held_quantity"] = len(held)
    undated = [rec.asset_tag for rec in held if rec.issued_date is None]
    if undated:
        return {
            **base,
            "status": REVIEW_REQUIRED,
            "reason_code": DATA_INCOMPLETE,
            "message": (f"Issue date missing for {', '.join(undated)}; age cannot be determined"),
            "undated_asset_tags": undated,
        }

    if len(held) < limit.max_quantity:
        return {
            **base,
            "status": ELIGIBLE,
            "reason_code": BELOW_MAX_QUANTITY,
            "message": (f"Holds {len(held)} of {limit.max_quantity} permitted {catalogue_item}(s)"),
            "next_eligible_date": as_of.isoformat(),
        }

    oldest = min(rec.issued_date for rec in held if rec.issued_date is not None)
    next_eligible = add_months(oldest, limit.refresh_months)
    days_until = (next_eligible - as_of).days
    base["oldest_issued_date"] = oldest.isoformat()
    base["next_eligible_date"] = next_eligible.isoformat()
    base["days_until_eligible"] = max(days_until, 0)

    if days_until <= 0:
        return {
            **base,
            "status": ELIGIBLE,
            "reason_code": REFRESH_DUE,
            "message": (
                f"Oldest {catalogue_item} issued {oldest.isoformat()} is "
                f"{months_between(oldest, as_of)} months old; refresh interval is "
                f"{limit.refresh_months} months"
            ),
        }
    if days_until <= BORDERLINE_DAYS:
        return {
            **base,
            "status": REVIEW_REQUIRED,
            "reason_code": BORDERLINE_TIMING,
            "message": (
                f"Eligible for refresh on {next_eligible.isoformat()}, "
                f"{days_until} days from {as_of.isoformat()}"
            ),
        }
    return {
        **base,
        "status": NOT_ELIGIBLE,
        "reason_code": WITHIN_REFRESH_INTERVAL,
        "message": (
            f"At the maximum of {limit.max_quantity} and the oldest {catalogue_item} "
            f"(issued {oldest.isoformat()}) is not due for refresh until "
            f"{next_eligible.isoformat()}"
        ),
    }


def _held_units(employee: Employee, catalogue_item: str) -> list[Any]:
    return [rec for rec in employee.equipment if normalize_name(rec.item) == catalogue_item]


def flag_for_human_review(
    employee_id: str,
    request: str,
    item: str,
    reason_code: str,
    reason: str,
    *,
    store: Store,
    request_id: str | None = None,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Record an escalation as one JSON line in the audit file and return a ticket ID.

    ``request`` is the employee's original request text, stored so the reviewer sees what
    was asked. An unknown employee is still recorded, since identity problems are a reason
    to escalate. An empty request, an empty reason, or an unknown reason code is rejected
    without writing.
    """
    now = now or datetime.now(UTC)
    code = reason_code.strip().upper()
    if code not in ESCALATION_REASON_CODES:
        return {
            "flagged": False,
            "error": INVALID_REASON_CODE,
            "message": f"Unknown reason code '{reason_code}'",
            "valid_reason_codes": sorted(ESCALATION_REASON_CODES),
        }
    if not request.strip():
        return {
            "flagged": False,
            "error": EMPTY_REQUEST,
            "message": "The employee's original request text is required",
        }
    if not reason.strip():
        return {
            "flagged": False,
            "error": EMPTY_REASON,
            "message": "A reason for human review is required",
        }

    employee = store.get_employee(employee_id) if employee_id.strip() else None
    ticket_id = f"ESC-{now:%Y%m%d}-{uuid.uuid4().hex[:8].upper()}"
    entry = {
        "ticket_id": ticket_id,
        "created_at": now.isoformat(),
        "request_id": request_id,
        "employee_id": employee.employee_id if employee else normalize_employee_id(employee_id),
        "employee_found": employee is not None,
        "role": employee.role if employee else None,
        "request": request.strip(),
        "item": item,
        "reason_code": code,
        "reason": reason.strip(),
        "policy_version": store.policy_version,
        "data_version": store.data_version,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "flagged": True,
        "status": "ESCALATED",
        "ticket_id": ticket_id,
        "created_at": entry["created_at"],
        "request_id": request_id,
        "employee_id": entry["employee_id"],
        "employee_found": entry["employee_found"],
        "request": entry["request"],
        "item": item,
        "reason_code": code,
        "message": f"Ticket {ticket_id} created for human review",
        "policy_version": store.policy_version,
    }
