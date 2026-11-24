"""Reflection: check the agent's draft before it goes out.

Stage one is code checks that cannot be talked out of their answer. Stage two is one
model call that critiques the draft against the observations. A failure triggers one
revision and a full re-check. A second failure means the draft is not trustworthy, and
the caller escalates with ``REFLECTION_FAILED``.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

from anthropic.types import Message, MessageParam, ToolParam

from equipment.decisions import (
    ESCALATE,
    Draft,
    Observation,
    Request,
    eligibility_status,
    expected_status,
    real_tickets,
)
from equipment.llm import LLMClient
from equipment.observability import RequestLogger

REFLECTION_PROMPT_VERSION = "1.0"

CONFIRMED = "CONFIRMED"
REVISED = "REVISED"
FAILED = "FAILED"

SUBMIT_CRITIQUE = "submit_critique"
SUBMIT_REVISION = "submit_revision"

CRITIQUE_PROMPT = f"""You are reviewing a draft reply written by an IT equipment request agent \
(reflection prompt version {REFLECTION_PROMPT_VERSION}). You will be given the employee's \
request, the exact results the agent's tools returned, and the agent's draft decision and \
reply.

Answer two questions by calling submit_critique:
1. accurate: Does the draft accurately reflect what the tools returned? Every fact, date, \
and figure in the draft must appear in the tool results, and the decision must follow \
from them.
2. commits_beyond_data: Does the draft commit to anything the data did not confirm? \
Examples: a delivery or arrival date, a specific model or specification, a timeframe, or \
what a human reviewer will decide. Any such promise means true.

List each problem in one short sentence in issues. Be strict: a draft that promises \
anything beyond the tool results fails."""

REVISION_PROMPT = f"""You are correcting a draft reply written by an IT equipment request agent \
(reflection prompt version {REFLECTION_PROMPT_VERSION}). You will be given the request, the \
exact tool results, the draft, and the problems found with it.

Rewrite only the reply text and submit it with submit_revision. Keep the same decision. \
State only facts that appear in the tool results, quote dates exactly as the tools gave \
them, and remove every promise about delivery, timing, models, or what a reviewer will \
decide. If the decision is an escalation, keep the ticket ID in the reply and say a human \
will review the request. Keep it to a short, polite paragraph."""

CRITIQUE_TOOL: ToolParam = {
    "name": SUBMIT_CRITIQUE,
    "description": "Submit the critique of the draft. Call exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accurate": {"type": "boolean"},
            "commits_beyond_data": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["accurate", "commits_beyond_data", "issues"],
        "additionalProperties": False,
    },
}

REVISION_TOOL: ToolParam = {
    "name": SUBMIT_REVISION,
    "description": "Submit the corrected reply text. Call exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {"draft_response": {"type": "string"}},
        "required": ["draft_response"],
        "additionalProperties": False,
    },
}

MONTHS: list[str] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTH_PATTERN = "|".join(MONTHS)
ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
DAY_MONTH_YEAR = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})\s+(\d{{4}})\b", re.IGNORECASE
)
MONTH_DAY_YEAR = re.compile(
    rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE
)
WEEKDAY = re.compile(
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)
RELATIVE = re.compile(
    r"\b(tomorrow|next week|next month|this week|within \d+ (?:days|weeks)|"
    r"by the end of (?:the|this) (?:week|month)|in \d+ (?:days|weeks))\b",
    re.IGNORECASE,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Critique:
    accurate: bool
    commits_beyond_data: bool
    issues: list[str]

    @property
    def passed(self) -> bool:
        return self.accurate and not self.commits_beyond_data


@dataclass
class Pass:
    """One full check of one draft: the code checks, then the critique if they passed."""

    draft_response: str
    checks: list[Check]
    critique: Critique | None = None

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and (
            self.critique is None or self.critique.passed
        )

    def problems(self) -> list[str]:
        problems = [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]
        if self.critique:
            problems += self.critique.issues
        return problems


@dataclass
class Outcome:
    verdict: str
    passes: list[Pass] = field(default_factory=list)
    original_draft: str = ""
    final_draft: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dates_in(text: str) -> list[tuple[str, str | None]]:
    """Date mentions in the text as (mention, ISO form or None if not a calendar date)."""
    found: list[tuple[str, str | None]] = [(m, m) for m in ISO_DATE.findall(text)]
    for day, month, year in DAY_MONTH_YEAR.findall(text):
        found.append((f"{day} {month} {year}", _iso(year, month, day)))
    for month, day, year in MONTH_DAY_YEAR.findall(text):
        found.append((f"{month} {day} {year}", _iso(year, month, day)))
    found += [(m, None) for m in WEEKDAY.findall(text)]
    found += [(m, None) for m in RELATIVE.findall(text)]
    return found


def _iso(year: str, month: str, day: str) -> str | None:
    try:
        return date(int(year), MONTHS.index(month.lower()) + 1, int(day)).isoformat()
    except ValueError:
        return None


def run_code_checks(request: Request, draft: Draft, observations: list[Observation]) -> list[Check]:
    """Stage one. Three checks that depend only on the observations."""
    checks: list[Check] = []

    status = eligibility_status(request, observations)
    needed = expected_status(draft.decision)
    if needed is None:
        checks.append(
            Check("decision_matches_eligibility", True, f"{draft.decision} needs no status")
        )
    else:
        checks.append(
            Check(
                "decision_matches_eligibility",
                status == needed,
                f"{draft.decision} with eligibility status {status or 'missing'}",
            )
        )

    evidence = " ".join(json.dumps(o.result) for o in observations)
    ungrounded = [
        mention
        for mention, iso in dates_in(draft.draft_response)
        if iso is None or iso not in evidence
    ]
    checks.append(
        Check(
            "dates_grounded",
            not ungrounded,
            "every date appears in an observation"
            if not ungrounded
            else f"not in any observation: {', '.join(ungrounded)}",
        )
    )

    if draft.decision == ESCALATE:
        tickets = real_tickets(observations)
        cited = draft.ticket_id is not None and draft.ticket_id in draft.draft_response
        real = draft.ticket_id in tickets
        checks.append(
            Check(
                "escalation_ticket",
                real and cited,
                f"ticket {draft.ticket_id or 'missing'} "
                + (
                    "is real and cited"
                    if real and cited
                    else "is not real"
                    if not real
                    else "is not cited"
                ),
            )
        )
    else:
        checks.append(Check("escalation_ticket", draft.ticket_id is None, "not an escalation"))
    return checks


def parse_critique(response: Message) -> Critique | None:
    """Prefer the submit_critique tool call; fall back to a JSON object in the text."""
    for block in response.content:
        if block.type == "tool_use" and block.name == SUBMIT_CRITIQUE:
            return _critique_from(dict(block.input) if isinstance(block.input, dict) else {})
    text = " ".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return _critique_from(json.loads(match.group(0)))
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _critique_from(data: dict[str, Any]) -> Critique:
    issues = data.get("issues") or []
    return Critique(
        accurate=bool(data.get("accurate", False)),
        commits_beyond_data=bool(data.get("commits_beyond_data", True)),
        issues=[str(i) for i in issues],
    )


def parse_revision(response: Message) -> str | None:
    for block in response.content:
        if block.type == "tool_use" and block.name == SUBMIT_REVISION:
            value = block.input.get("draft_response") if isinstance(block.input, dict) else None
            return str(value) if value else None
    text = " ".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


class Reflection:
    def __init__(self, llm: LLMClient, logger: RequestLogger, *, out: Any = None) -> None:
        self.llm = llm
        self.logger = logger
        self.out = out or sys.stdout

    def print(self, text: str) -> None:
        print(text, file=self.out)

    async def run(self, request: Request, draft: Draft, observations: list[Observation]) -> Outcome:
        self.print("\n--- Reflection ---")
        outcome = Outcome(verdict=FAILED, original_draft=draft.draft_response)

        first = await self._check(request, draft, observations, label="Check")
        outcome.passes.append(first)
        if first.passed:
            outcome.verdict = CONFIRMED
            outcome.final_draft = draft.draft_response
            self.print("Verdict: CONFIRMED")
            self.logger.log("reflection", verdict=CONFIRMED, passes=1)
            return outcome

        self.print("Verdict: REVISE")
        revised_text = await self._revise(request, draft, observations, first.problems())
        if revised_text is None:
            self.print("Revision: model returned no usable text")
            self.print("Verdict: FAILED")
            self.logger.log("reflection", level="WARNING", verdict=FAILED, passes=1)
            return outcome

        self.print(f"Before: {draft.draft_response}")
        self.print(f"After: {revised_text}")
        revised = Draft(draft.decision, draft.reason_code, draft.ticket_id, revised_text)
        second = await self._check(request, revised, observations, label="Re-check")
        outcome.passes.append(second)
        if second.passed:
            outcome.verdict = REVISED
            outcome.final_draft = revised_text
            self.print("Verdict: REVISED")
            self.logger.log(
                "reflection",
                verdict=REVISED,
                passes=2,
                before=draft.draft_response,
                after=revised_text,
            )
        else:
            self.print("Verdict: FAILED")
            self.logger.log(
                "reflection", level="WARNING", verdict=FAILED, passes=2, problems=second.problems()
            )
        return outcome

    async def _check(
        self, request: Request, draft: Draft, observations: list[Observation], *, label: str
    ) -> Pass:
        checks = run_code_checks(request, draft, observations)
        for check in checks:
            self.print(
                f"{label} {check.name}: {'pass' if check.passed else 'FAIL'} ({check.detail})"
            )
            self.logger.log(
                "reflection_check", name=check.name, passed=check.passed, detail=check.detail
            )
        if not all(c.passed for c in checks):
            return Pass(draft.draft_response, checks)

        critique = await self._critique(request, draft, observations)
        self.print(
            f"{label} critique: accurate={'yes' if critique.accurate else 'NO'}, "
            f"commits_beyond_data={'YES' if critique.commits_beyond_data else 'no'}"
            + (f"; issues: {'; '.join(critique.issues)}" if critique.issues else "")
        )
        self.logger.log("reflection_critique", **asdict(critique))
        return Pass(draft.draft_response, checks, critique)

    async def _critique(
        self, request: Request, draft: Draft, observations: list[Observation]
    ) -> Critique:
        response = await self._call(
            CRITIQUE_PROMPT, _evidence(request, draft, observations), [CRITIQUE_TOOL], "critique"
        )
        critique = parse_critique(response)
        if critique is None:
            self.logger.log("reflection_critique_unparsed", level="WARNING")
            return Critique(
                accurate=False, commits_beyond_data=True, issues=["critique unreadable"]
            )
        return critique

    async def _revise(
        self, request: Request, draft: Draft, observations: list[Observation], problems: list[str]
    ) -> str | None:
        content = (
            _evidence(request, draft, observations)
            + "\n\nProblems found:\n"
            + "\n".join(f"- {p}" for p in problems)
        )
        response = await self._call(REVISION_PROMPT, content, [REVISION_TOOL], "revision")
        return parse_revision(response)

    async def _call(self, system: str, content: str, tools: list[ToolParam], stage: str) -> Message:
        messages: list[MessageParam] = [{"role": "user", "content": content}]
        with self.logger.span(
            "model_call",
            purpose="reflection",
            stage=stage,
            prompt_version=REFLECTION_PROMPT_VERSION,
        ):
            return await self.llm.complete(
                purpose="reflection", step=0, system=system, messages=messages, tools=tools
            )


def _evidence(request: Request, draft: Draft, observations: list[Observation]) -> str:
    tool_results = [
        {"tool": o.tool, "arguments": o.arguments, "result": o.result} for o in observations
    ]
    return (
        f"Request:\n{request.as_text()}\n\n"
        f"Tool results:\n{json.dumps(tool_results, indent=2)}\n\n"
        f"Draft decision: {draft.decision} ({draft.reason_code}), ticket: {draft.ticket_id}\n"
        f"Draft reply:\n{draft.draft_response}"
    )
