from datetime import date

import pytest

from equipment.core import EMPLOYEE_NOT_FOUND, get_employee_info, months_between
from equipment.store import Store
from tests.unit.conftest import AS_OF


class TestMonthsBetween:
    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (date(2026, 1, 15), date(2026, 1, 15), 0),
            (date(2026, 1, 15), date(2026, 2, 14), 0),
            (date(2026, 1, 15), date(2026, 2, 15), 1),
            (date(2022, 3, 14), date(2026, 11, 23), 56),
            (date(2023, 11, 23), date(2026, 11, 23), 36),
            (date(2026, 3, 1), date(2026, 1, 1), -2),
        ],
    )
    def test_whole_months(self, start: date, end: date, expected: int) -> None:
        assert months_between(start, end) == expected


class TestKnownEmployee:
    def test_returns_role_tenure_and_equipment(self, store: Store) -> None:
        result = get_employee_info("E001", store=store, as_of=AS_OF)

        assert result["found"] is True
        assert result["employee_id"] == "E001"
        assert result["name"] == "Alice Morgan"
        assert result["role"] == "standard"
        assert result["start_date"] == "2021-06-01"
        assert result["tenure_months"] == 65
        assert result["as_of"] == "2026-11-23"
        assert result["policy_version"] == store.policy_version
        assert result["data_version"] == store.data_version

        items = {rec["asset_tag"]: rec for rec in result["equipment"]}
        assert set(items) == {"LT-0001", "KB-0001", "MS-0001"}
        laptop = items["LT-0001"]
        assert laptop["item"] == "laptop"
        assert laptop["issued_date"] == "2022-03-14"
        assert laptop["age_months"] == 56
        assert laptop["data_quality"] is None

    def test_tenure_uses_as_of_not_today(self, store: Store) -> None:
        later = get_employee_info("E001", store=store, as_of=date(2027, 6, 1))
        assert later["tenure_months"] == 72

    def test_null_issued_date_is_passed_through_with_flag(self, store: Store) -> None:
        result = get_employee_info("E006", store=store, as_of=AS_OF)
        laptop = next(r for r in result["equipment"] if r["asset_tag"] == "LT-0006")
        assert laptop["issued_date"] is None
        assert laptop["age_months"] is None
        assert laptop["data_quality"] == "incomplete"


class TestEdgeCases:
    def test_unknown_id(self, store: Store) -> None:
        result = get_employee_info("E999", store=store, as_of=AS_OF)
        assert result == {
            "found": False,
            "error": EMPLOYEE_NOT_FOUND,
            "employee_id": "E999",
            "message": "No employee with ID 'E999' is on file",
            "policy_version": store.policy_version,
        }

    def test_empty_string(self, store: Store) -> None:
        result = get_employee_info("", store=store, as_of=AS_OF)
        assert result["found"] is False
        assert result["error"] == EMPLOYEE_NOT_FOUND

    def test_whitespace_only(self, store: Store) -> None:
        result = get_employee_info("   ", store=store, as_of=AS_OF)
        assert result["found"] is False
        assert result["error"] == EMPLOYEE_NOT_FOUND

    def test_lower_case_id_is_found(self, store: Store) -> None:
        result = get_employee_info("e003", store=store, as_of=AS_OF)
        assert result["found"] is True
        assert result["employee_id"] == "E003"
        assert result["role"] == "manager"

    def test_surrounding_whitespace_is_ignored(self, store: Store) -> None:
        result = get_employee_info("  E004 ", store=store, as_of=AS_OF)
        assert result["found"] is True
        assert result["employee_id"] == "E004"

    def test_employee_with_no_equipment(self, store: Store) -> None:
        result = get_employee_info("E007", store=store, as_of=AS_OF)
        assert result["found"] is True
        assert result["equipment"] == []
        assert result["tenure_months"] == 1
