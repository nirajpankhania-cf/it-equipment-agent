import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from equipment.core import (
    EMPTY_REASON,
    EMPTY_REQUEST,
    ESCALATION_REASON_CODES,
    EXCEPTION_CLAIMED,
    INVALID_REASON_CODE,
    flag_for_human_review,
)
from equipment.store import Store

NOW = datetime(2026, 11, 23, 10, 30, 0, tzinfo=UTC)
TICKET_PATTERN = re.compile(r"^ESC-20261123-[0-9A-F]{8}$")


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "escalations.jsonl"


def read_entries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def flag(store: Store, audit_path: Path, **overrides: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "employee_id": "E003",
        "request": "The screen on my laptop is cracked. I need a replacement.",
        "item": "laptop",
        "reason_code": EXCEPTION_CLAIMED,
        "reason": "Screen reported cracked; damage cannot be verified from records",
        "request_id": "req-001",
    }
    args.update(overrides)
    return flag_for_human_review(store=store, audit_path=audit_path, now=NOW, **args)


class TestNormal:
    def test_writes_one_entry_and_returns_ticket_id(self, store: Store, audit_path: Path) -> None:
        result = flag(store, audit_path)

        assert result["flagged"] is True
        assert result["status"] == "ESCALATED"
        assert TICKET_PATTERN.match(result["ticket_id"])
        assert result["created_at"] == "2026-11-23T10:30:00+00:00"
        assert result["request_id"] == "req-001"
        assert result["employee_id"] == "E003"
        assert result["employee_found"] is True
        assert result["request"] == "The screen on my laptop is cracked. I need a replacement."
        assert result["reason_code"] == EXCEPTION_CLAIMED
        assert result["ticket_id"] in result["message"]
        assert result["policy_version"] == "2026.1"

        entries = read_entries(audit_path)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["ticket_id"] == result["ticket_id"]
        assert entry["created_at"] == result["created_at"]
        assert entry["request_id"] == "req-001"
        assert entry["employee_id"] == "E003"
        assert entry["employee_found"] is True
        assert entry["role"] == "manager"
        assert entry["request"] == "The screen on my laptop is cracked. I need a replacement."
        assert entry["item"] == "laptop"
        assert entry["reason_code"] == EXCEPTION_CLAIMED
        assert entry["reason"].startswith("Screen reported cracked")
        assert entry["policy_version"] == "2026.1"
        assert entry["data_version"] == "2026.11"

    def test_creates_missing_audit_directory(self, store: Store, audit_path: Path) -> None:
        assert not audit_path.parent.exists()
        flag(store, audit_path)
        assert audit_path.exists()

    def test_request_id_is_optional(self, store: Store, audit_path: Path) -> None:
        result = flag(store, audit_path, request_id=None)
        assert result["flagged"] is True
        assert read_entries(audit_path)[0]["request_id"] is None

    def test_reason_code_is_normalized(self, store: Store, audit_path: Path) -> None:
        result = flag(store, audit_path, reason_code=" exception_claimed ")
        assert result["reason_code"] == EXCEPTION_CLAIMED

    def test_defaults_to_current_time(self, store: Store, audit_path: Path) -> None:
        result = flag_for_human_review(
            "E003",
            "Cracked screen",
            "laptop",
            EXCEPTION_CLAIMED,
            "damage claimed",
            store=store,
            audit_path=audit_path,
        )
        assert result["flagged"] is True
        assert datetime.fromisoformat(result["created_at"]).tzinfo is not None


class TestEdgeCases:
    @pytest.mark.parametrize("reason", ["", "   ", "\n"])
    def test_empty_reason_rejected_without_writing(
        self, store: Store, audit_path: Path, reason: str
    ) -> None:
        result = flag(store, audit_path, reason=reason)
        assert result == {
            "flagged": False,
            "error": EMPTY_REASON,
            "message": "A reason for human review is required",
        }
        assert not audit_path.exists()

    @pytest.mark.parametrize("request_text", ["", "   "])
    def test_empty_request_rejected_without_writing(
        self, store: Store, audit_path: Path, request_text: str
    ) -> None:
        result = flag(store, audit_path, request=request_text)
        assert result == {
            "flagged": False,
            "error": EMPTY_REQUEST,
            "message": "The employee's original request text is required",
        }
        assert not audit_path.exists()

    def test_request_text_is_stripped(self, store: Store, audit_path: Path) -> None:
        result = flag(store, audit_path, request="  Need a laptop.\n")
        assert result["request"] == "Need a laptop."
        assert read_entries(audit_path)[0]["request"] == "Need a laptop."

    def test_unknown_reason_code_rejected_without_writing(
        self, store: Store, audit_path: Path
    ) -> None:
        result = flag(store, audit_path, reason_code="BECAUSE")
        assert result["flagged"] is False
        assert result["error"] == INVALID_REASON_CODE
        assert result["valid_reason_codes"] == sorted(ESCALATION_REASON_CODES)
        assert not audit_path.exists()

    def test_unknown_employee_still_recorded(self, store: Store, audit_path: Path) -> None:
        result = flag(
            store,
            audit_path,
            employee_id="e999",
            item="headset",
            reason_code="EMPLOYEE_NOT_FOUND",
            reason="ID not on file",
        )
        assert result["flagged"] is True
        assert result["employee_id"] == "E999"
        assert result["employee_found"] is False

        entry = read_entries(audit_path)[0]
        assert entry["employee_id"] == "E999"
        assert entry["employee_found"] is False
        assert entry["role"] is None

    def test_two_calls_give_distinct_ids_and_two_lines(
        self, store: Store, audit_path: Path
    ) -> None:
        first = flag(store, audit_path)
        second = flag(store, audit_path)
        assert first["ticket_id"] != second["ticket_id"]

        entries = read_entries(audit_path)
        assert [e["ticket_id"] for e in entries] == [first["ticket_id"], second["ticket_id"]]

    def test_leaves_no_files_outside_injected_path(
        self, store: Store, audit_path: Path, tmp_path: Path
    ) -> None:
        flag(store, audit_path)
        assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == [
            Path("audit"),
            Path("audit/escalations.jsonl"),
        ]
