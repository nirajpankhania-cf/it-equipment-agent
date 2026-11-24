"""Shared structures for the agent and reflection: requests, observations, results,
and the grounding helpers both of them use to tie a decision back to tool results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, cast

from equipment import core
from equipment.store import normalize_employee_id

APPROVE = "APPROVE"
DENY = "DENY"
ESCALATE = "ESCALATE"
DECISIONS = (APPROVE, DENY, ESCALATE)

FLAG_TOOL = "flag_for_human_review"
ELIGIBILITY_TOOL = "check_request_eligibility"


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


@dataclass
class Observation:
    step: int
    tool: str
    arguments: dict[str, Any]
    result: Any
    is_error: bool


@dataclass
class Draft:
    """What the model submitted, or what the code produced in its place."""

    decision: str
    reason_code: str
    ticket_id: str | None
    draft_response: str


@dataclass
class AgentResult:
    decision: str
    reason_code: str
    ticket_id: str | None
    draft_response: str
    steps: int
    guardrail: str | None
    observations: list[Observation] = field(default_factory=list)
    reflection: dict[str, Any] | None = None

    @property
    def flag_calls(self) -> list[Observation]:
        return [o for o in self.observations if o.tool == FLAG_TOOL]

    def draft(self) -> Draft:
        return Draft(self.decision, self.reason_code, self.ticket_id, self.draft_response)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def eligibility_status(request: Request, observations: list[Observation]) -> str | None:
    """Status of the latest eligibility observation for this request's employee and item."""
    for obs in reversed(observations):
        if obs.tool != ELIGIBILITY_TOOL or not isinstance(obs.result, dict):
            continue
        if obs.result.get("employee_id") != normalize_employee_id(request.employee_id):
            continue
        item = obs.result.get("item")
        if isinstance(item, str) and core.normalize_item(request.item, [item]) == item:
            return cast(str | None, obs.result.get("status"))
    return None


def real_tickets(observations: list[Observation]) -> set[str]:
    return {
        obs.result["ticket_id"]
        for obs in observations
        if obs.tool == FLAG_TOOL
        and isinstance(obs.result, dict)
        and obs.result.get("flagged") is True
    }


def expected_status(decision: str) -> str | None:
    """The eligibility status a decision must rest on; None if any status is acceptable."""
    return {APPROVE: core.ELIGIBLE, DENY: core.NOT_ELIGIBLE}.get(decision)
