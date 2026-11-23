"""The tool functions as plain Python. server.py wraps these and adds nothing.

Errors are returned as structured results, not raised, so the agent can read the
outcome in an Observation and act on it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from equipment.store import Store, normalize_employee_id, normalize_name

EMPLOYEE_NOT_FOUND = "EMPLOYEE_NOT_FOUND"
ROLE_NOT_FOUND = "ROLE_NOT_FOUND"


def months_between(start: date, end: date) -> int:
    """Whole calendar months from ``start`` to ``end``. Negative if ``end`` is earlier."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months


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
