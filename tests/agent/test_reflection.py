"""Reflection: code checks, model critique, one revision, then REFLECTION_FAILED."""

import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from equipment.decisions import APPROVE, DENY, ESCALATE, Draft, Observation, Request
from equipment.llm import LLMClient, ModelPrice, Pricing
from equipment.observability import RequestLogger, RunLogger
from equipment.reflection import (
    CONFIRMED,
    FAILED,
    REVISED,
    Reflection,
    dates_in,
    parse_critique,
    parse_revision,
    run_code_checks,
)
from tests.agent.fake_model import (
    CLEAN_CRITIQUE,
    FakeModel,
    critique_message,
    revision_message,
    text_message,
)

PRICING = Pricing({"claude-fable-5-2": ModelPrice(8.0, 40.0)}, max_cost_per_run_usd=5.0)
REQUEST = Request("E001", "laptop", "It is slow.", date(2026, 11, 24))

EMPLOYEE = Observation(
    1,
    "get_employee_info",
    {"employee_id": "E001"},
    {
        "found": True,
        "employee_id": "E001",
        "role": "standard",
        "equipment": [{"item": "laptop", "issued_date": "2022-03-14", "age_months": 56}],
    },
    False,
)
ELIGIBLE = Observation(
    1,
    "check_request_eligibility",
    {"employee_id": "E001", "item": "laptop"},
    {
        "employee_id": "E001",
        "item": "laptop",
        "status": "ELIGIBLE",
        "reason_code": "REFRESH_DUE",
        "oldest_issued_date": "2022-03-14",
        "next_eligible_date": "2026-03-14",
    },
    False,
)
NOT_ELIGIBLE = Observation(
    1,
    "check_request_eligibility",
    {"employee_id": "E001", "item": "laptop"},
    {**ELIGIBLE.result, "status": "NOT_ELIGIBLE", "next_eligible_date": "2028-09-15"},
    False,
)
FLAG = Observation(
    2,
    "flag_for_human_review",
    {"employee_id": "E001", "item": "laptop", "reason_code": "EXCEPTION_CLAIMED", "reason": "x"},
    {"flagged": True, "ticket_id": "ESC-20261124-ABCD1234"},
    False,
)
OBSERVATIONS = [EMPLOYEE, ELIGIBLE]

PLANTED = Draft(
    APPROVE, "REFRESH_DUE", None, "Good news: approved. Your laptop will arrive Friday."
)
CLEAN = Draft(
    APPROVE,
    "REFRESH_DUE",
    None,
    "Your laptop issued on 2022-03-14 is past the 48-month refresh interval, so the request "
    "is approved.",
)


def names_failed(checks: list[Any]) -> list[str]:
    return [c.name for c in checks if not c.passed]


@pytest.fixture
def logger(tmp_path: Path) -> RequestLogger:
    return RunLogger(run_id="r1", logs_dir=tmp_path).for_request("req-1")


def make_reflection(
    logger: RequestLogger, responses: list[Any]
) -> tuple[Reflection, FakeModel, io.StringIO]:
    fake = FakeModel(responses)
    llm = LLMClient(logger, model="claude-fable-5-2", pricing=PRICING, create=fake.create)
    out = io.StringIO()
    return Reflection(llm, logger, out=out), fake, out


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestDatesIn:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("issued on 2022-03-14", [("2022-03-14", "2022-03-14")]),
            ("due on 15 September 2028", [("15 September 2028", "2028-09-15")]),
            ("due on 15th September 2028", [("15 September 2028", "2028-09-15")]),
            ("due on September 15, 2028", [("September 15 2028", "2028-09-15")]),
            ("arrives Friday", [("Friday", None)]),
            ("arrives next week", [("next week", None)]),
            ("within 3 days", [("within 3 days", None)]),
            ("no dates here", []),
            ("30 February 2028 is invalid", [("30 February 2028", None)]),
        ],
    )
    def test_extracts_mentions(self, text: str, expected: list[tuple[str, str | None]]) -> None:
        assert dates_in(text) == expected


class TestCodeChecks:
    def test_planted_draft_promising_friday_is_rejected(self) -> None:
        checks = run_code_checks(REQUEST, PLANTED, OBSERVATIONS)
        assert names_failed(checks) == ["dates_grounded"]
        failed = next(c for c in checks if c.name == "dates_grounded")
        assert "Friday" in failed.detail

    def test_clean_draft_passes_all_three(self) -> None:
        checks = run_code_checks(REQUEST, CLEAN, OBSERVATIONS)
        assert [c.name for c in checks] == [
            "decision_matches_eligibility",
            "dates_grounded",
            "escalation_ticket",
        ]
        assert names_failed(checks) == []

    def test_date_not_in_any_observation_fails(self) -> None:
        draft = Draft(
            APPROVE, "REFRESH_DUE", None, "Approved; your old laptop dates from 2019-01-01."
        )
        assert names_failed(run_code_checks(REQUEST, draft, OBSERVATIONS)) == ["dates_grounded"]

    def test_long_form_date_matching_an_observation_passes(self) -> None:
        draft = Draft(APPROVE, "REFRESH_DUE", None, "Your laptop was issued on 14 March 2022.")
        assert names_failed(run_code_checks(REQUEST, draft, OBSERVATIONS)) == []

    def test_approve_without_eligible_fails(self) -> None:
        checks = run_code_checks(REQUEST, CLEAN, [EMPLOYEE, NOT_ELIGIBLE])
        assert names_failed(checks) == ["decision_matches_eligibility"]

    def test_deny_needs_not_eligible(self) -> None:
        deny = Draft(DENY, "WITHIN_REFRESH_INTERVAL", None, "Not until 2028-09-15.")
        assert names_failed(run_code_checks(REQUEST, deny, [EMPLOYEE, NOT_ELIGIBLE])) == []
        # Against ELIGIBLE observations the decision fails, and so does the date, which
        # only appears in the NOT_ELIGIBLE result.
        assert names_failed(run_code_checks(REQUEST, deny, OBSERVATIONS)) == [
            "decision_matches_eligibility",
            "dates_grounded",
        ]

    def test_escalation_must_cite_real_ticket(self) -> None:
        good = Draft(
            ESCALATE,
            "EXCEPTION_CLAIMED",
            FLAG.result["ticket_id"],
            f"Ticket {FLAG.result['ticket_id']} raised.",
        )
        assert names_failed(run_code_checks(REQUEST, good, [*OBSERVATIONS, FLAG])) == []

        uncited = Draft(
            ESCALATE, "EXCEPTION_CLAIMED", FLAG.result["ticket_id"], "A human will review."
        )
        checks = run_code_checks(REQUEST, uncited, [*OBSERVATIONS, FLAG])
        assert names_failed(checks) == ["escalation_ticket"]
        assert "not cited" in next(c for c in checks if c.name == "escalation_ticket").detail

        invented = Draft(ESCALATE, "EXCEPTION_CLAIMED", "ESC-FAKE", "Ticket ESC-FAKE raised.")
        checks = run_code_checks(REQUEST, invented, [*OBSERVATIONS, FLAG])
        assert "not real" in next(c for c in checks if c.name == "escalation_ticket").detail

    def test_non_escalation_must_not_carry_a_ticket(self) -> None:
        draft = Draft(APPROVE, "REFRESH_DUE", "ESC-1", CLEAN.draft_response)
        assert names_failed(run_code_checks(REQUEST, draft, OBSERVATIONS)) == ["escalation_ticket"]


class TestParsing:
    def test_critique_from_tool_call(self) -> None:
        critique = parse_critique(
            critique_message(accurate=True, commits_beyond_data=True, issues=["promises Friday"])
        )
        assert critique is not None
        assert critique.accurate is True
        assert critique.commits_beyond_data is True
        assert critique.issues == ["promises Friday"]
        assert critique.passed is False

    def test_critique_from_json_text_fallback(self) -> None:
        critique = parse_critique(
            text_message(
                'Here it is:\n{"accurate": true, "commits_beyond_data": false, "issues": []}'
            )
        )
        assert critique is not None
        assert critique.passed is True

    def test_unparseable_critique_is_none(self) -> None:
        assert parse_critique(text_message("I am not sure.")) is None

    def test_revision_from_tool_call_or_text(self) -> None:
        assert parse_revision(revision_message("Better text.")) == "Better text."
        assert parse_revision(text_message("Plain better text.")) == "Plain better text."


class TestReflectionRun:
    async def test_clean_draft_is_confirmed_with_one_model_call(
        self, logger: RequestLogger
    ) -> None:
        reflection, fake, out = make_reflection(logger, [CLEAN_CRITIQUE])

        outcome = await reflection.run(REQUEST, CLEAN, OBSERVATIONS)

        assert outcome.verdict == CONFIRMED
        assert outcome.final_draft == CLEAN.draft_response
        assert len(fake.calls) == 1
        assert fake.calls[0]["tools"][0]["name"] == "submit_critique"
        trace = out.getvalue()
        assert "--- Reflection ---" in trace
        assert "Check decision_matches_eligibility: pass" in trace
        assert "Check dates_grounded: pass" in trace
        assert "Check escalation_ticket: pass" in trace
        assert "Check critique: accurate=yes, commits_beyond_data=no" in trace
        assert "Verdict: CONFIRMED" in trace

    async def test_planted_friday_draft_is_rejected_and_revised(
        self, logger: RequestLogger
    ) -> None:
        revised_text = "Your laptop issued on 2022-03-14 is due for refresh, so this is approved."
        reflection, fake, out = make_reflection(
            logger, [revision_message(revised_text), CLEAN_CRITIQUE]
        )

        outcome = await reflection.run(REQUEST, PLANTED, OBSERVATIONS)

        assert outcome.verdict == REVISED
        assert outcome.original_draft == PLANTED.draft_response
        assert outcome.final_draft == revised_text
        assert len(outcome.passes) == 2
        assert outcome.passes[0].passed is False
        assert outcome.passes[0].critique is None  # code checks failed, so no critique call
        assert outcome.passes[1].passed is True

        # Two model calls: the revision, then the re-check critique. No critique on the first pass.
        assert [c["tools"][0]["name"] for c in fake.calls] == ["submit_revision", "submit_critique"]
        assert "Friday" in fake.calls[0]["messages"][0]["content"]
        assert "dates_grounded" in fake.calls[0]["messages"][0]["content"]

        trace = out.getvalue()
        assert "Check dates_grounded: FAIL (not in any observation: Friday)" in trace
        assert "Verdict: REVISE" in trace
        assert f"Before: {PLANTED.draft_response}" in trace
        assert f"After: {revised_text}" in trace
        assert "Re-check dates_grounded: pass" in trace
        assert "Verdict: REVISED" in trace

        events = [line for line in read(logger.run.path) if line["event"] == "reflection"]
        assert events[-1]["verdict"] == REVISED
        assert events[-1]["before"] == PLANTED.draft_response
        costs = [line for line in read(logger.run.path) if line["event"] == "model_call"]
        assert all(line["purpose"] == "reflection" for line in costs)

    async def test_critique_catches_promise_the_code_checks_miss(
        self, logger: RequestLogger
    ) -> None:
        subtle = Draft(
            APPROVE, "REFRESH_DUE", None, "Approved. You will receive the latest Pro model."
        )
        reflection, fake, out = make_reflection(
            logger,
            [
                critique_message(accurate=True, commits_beyond_data=True, issues=["names a model"]),
                revision_message(CLEAN.draft_response),
                CLEAN_CRITIQUE,
            ],
        )

        outcome = await reflection.run(REQUEST, subtle, OBSERVATIONS)

        assert outcome.verdict == REVISED
        assert len(fake.calls) == 3
        assert "commits_beyond_data=YES; issues: names a model" in out.getvalue()

    async def test_second_failure_is_failed(self, logger: RequestLogger) -> None:
        reflection, fake, out = make_reflection(
            logger, [revision_message("Approved. It ships tomorrow.")]
        )

        outcome = await reflection.run(REQUEST, PLANTED, OBSERVATIONS)

        assert outcome.verdict == FAILED
        assert outcome.final_draft == ""
        assert len(outcome.passes) == 2
        assert outcome.passes[1].problems() == ["dates_grounded: not in any observation: tomorrow"]
        assert "Re-check dates_grounded: FAIL" in out.getvalue()
        assert "Verdict: FAILED" in out.getvalue()

    async def test_unreadable_critique_counts_as_failure(self, logger: RequestLogger) -> None:
        reflection, _, _ = make_reflection(
            logger,
            [text_message("no json"), revision_message(CLEAN.draft_response), CLEAN_CRITIQUE],
        )
        outcome = await reflection.run(REQUEST, CLEAN, OBSERVATIONS)
        assert outcome.verdict == REVISED
        assert outcome.passes[0].critique is not None
        assert outcome.passes[0].critique.issues == ["critique unreadable"]

    async def test_revision_without_text_is_failed(self, logger: RequestLogger) -> None:
        reflection, _, out = make_reflection(logger, [text_message("")])
        outcome = await reflection.run(REQUEST, PLANTED, OBSERVATIONS)
        assert outcome.verdict == FAILED
        assert "no usable text" in out.getvalue()
