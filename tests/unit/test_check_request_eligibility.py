from datetime import date, timedelta

import pytest

from equipment.core import (
    BELOW_MAX_QUANTITY,
    BORDERLINE_TIMING,
    DATA_INCOMPLETE,
    ELIGIBLE,
    EMPLOYEE_NOT_FOUND,
    ITEM_NOT_IN_CATALOGUE,
    NOT_ELIGIBLE,
    REFRESH_DUE,
    REVIEW_REQUIRED,
    ROLE_NOT_ENTITLED,
    WITHIN_REFRESH_INTERVAL,
    add_months,
    check_request_eligibility,
    normalize_item,
)
from equipment.store import Store
from tests.unit.conftest import AS_OF

CATALOGUE = ["laptop", "monitor", "keyboard", "mouse", "headset", "docking_station"]


class TestHelpers:
    @pytest.mark.parametrize(
        ("start", "months", "expected"),
        [
            (date(2024, 12, 20), 24, date(2026, 12, 20)),
            (date(2023, 11, 23), 36, date(2026, 11, 23)),
            (date(2024, 1, 31), 1, date(2024, 2, 29)),
            (date(2023, 1, 31), 1, date(2023, 2, 28)),
            (date(2025, 10, 1), 3, date(2026, 1, 1)),
        ],
    )
    def test_add_months(self, start: date, months: int, expected: date) -> None:
        assert add_months(start, months) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("laptop", "laptop"),
            ("Laptop", "laptop"),
            ("  monitor ", "monitor"),
            ("monitors", "monitor"),
            ("docking station", "docking_station"),
            ("Docking-Station", "docking_station"),
            ("standing desk", None),
            ("", None),
            ("s", None),
        ],
    )
    def test_normalize_item(self, raw: str, expected: str | None) -> None:
        assert normalize_item(raw, CATALOGUE) == expected


class TestNormalCases:
    def test_eligible_by_age(self, store: Store) -> None:
        result = check_request_eligibility("E001", "laptop", store=store, as_of=AS_OF)
        assert result["status"] == ELIGIBLE
        assert result["reason_code"] == REFRESH_DUE
        assert result["found"] is True
        assert result["role"] == "standard"
        assert result["held_quantity"] == 1
        assert result["max_quantity"] == 1
        assert result["refresh_months"] == 48
        assert result["oldest_issued_date"] == "2022-03-14"
        assert result["next_eligible_date"] == "2026-03-14"
        assert result["days_until_eligible"] == 0
        assert result["policy_version"] == "2026.1"
        assert result["as_of"] == "2026-11-23"

    def test_eligible_by_quantity(self, store: Store) -> None:
        result = check_request_eligibility("E003", "monitor", store=store, as_of=AS_OF)
        assert result["status"] == ELIGIBLE
        assert result["reason_code"] == BELOW_MAX_QUANTITY
        assert result["held_quantity"] == 1
        assert result["max_quantity"] == 2
        assert result["next_eligible_date"] == "2026-11-23"

    def test_employee_with_no_equipment_is_eligible_by_quantity(self, store: Store) -> None:
        result = check_request_eligibility("E007", "laptop", store=store, as_of=AS_OF)
        assert result["status"] == ELIGIBLE
        assert result["reason_code"] == BELOW_MAX_QUANTITY
        assert result["held_quantity"] == 0

    def test_denied_within_interval(self, store: Store) -> None:
        result = check_request_eligibility("E002", "monitor", store=store, as_of=AS_OF)
        assert result["status"] == NOT_ELIGIBLE
        assert result["reason_code"] == WITHIN_REFRESH_INTERVAL
        assert result["held_quantity"] == 1
        assert result["oldest_issued_date"] == "2025-09-15"
        assert result["next_eligible_date"] == "2028-09-15"
        assert result["days_until_eligible"] == (date(2028, 9, 15) - AS_OF).days
        assert "2028-09-15" in result["message"]

    def test_denied_by_role(self, store: Store) -> None:
        result = check_request_eligibility("E005", "laptop", store=store, as_of=AS_OF)
        assert result["status"] == NOT_ELIGIBLE
        assert result["reason_code"] == ROLE_NOT_ENTITLED
        assert result["role"] == "contractor"
        assert result["next_eligible_date"] is None
        assert "max_quantity" not in result

    def test_denied_by_role_docking_station(self, store: Store) -> None:
        result = check_request_eligibility("E010", "docking station", store=store, as_of=AS_OF)
        assert result["status"] == NOT_ELIGIBLE
        assert result["reason_code"] == ROLE_NOT_ENTITLED
        assert result["item"] == "docking_station"

    def test_at_max_quantity_of_two(self, store: Store) -> None:
        result = check_request_eligibility("E004", "monitor", store=store, as_of=AS_OF)
        assert result["status"] == NOT_ELIGIBLE
        assert result["reason_code"] == WITHIN_REFRESH_INTERVAL
        assert result["held_quantity"] == 2
        assert result["next_eligible_date"] == "2027-08-05"


class TestEdgeCases:
    def test_unknown_employee(self, store: Store) -> None:
        result = check_request_eligibility("E999", "laptop", store=store, as_of=AS_OF)
        assert result["found"] is False
        assert result["error"] == EMPLOYEE_NOT_FOUND
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == EMPLOYEE_NOT_FOUND
        assert result["employee_id"] == "E999"
        assert result["policy_version"] == "2026.1"

    def test_empty_employee_id(self, store: Store) -> None:
        result = check_request_eligibility("  ", "laptop", store=store, as_of=AS_OF)
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == EMPLOYEE_NOT_FOUND

    def test_item_not_in_catalogue(self, store: Store) -> None:
        result = check_request_eligibility("E004", "standing desk", store=store, as_of=AS_OF)
        assert result["found"] is True
        assert result["error"] == ITEM_NOT_IN_CATALOGUE
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == ITEM_NOT_IN_CATALOGUE
        assert result["item"] == "standing desk"
        assert result["catalogue"] == CATALOGUE

    def test_unknown_employee_wins_over_unknown_item(self, store: Store) -> None:
        result = check_request_eligibility("E999", "standing desk", store=store, as_of=AS_OF)
        assert result["reason_code"] == EMPLOYEE_NOT_FOUND

    def test_exactly_on_refresh_date(self, store: Store) -> None:
        result = check_request_eligibility("E009", "laptop", store=store, as_of=AS_OF)
        assert result["oldest_issued_date"] == "2023-11-23"
        assert result["next_eligible_date"] == "2026-11-23"
        assert result["status"] == ELIGIBLE
        assert result["reason_code"] == REFRESH_DUE
        assert result["days_until_eligible"] == 0

    def test_day_before_refresh_date_is_borderline(self, store: Store) -> None:
        result = check_request_eligibility(
            "E009", "laptop", store=store, as_of=AS_OF - timedelta(days=1)
        )
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == BORDERLINE_TIMING
        assert result["days_until_eligible"] == 1

    def test_29_days_before_is_borderline(self, store: Store) -> None:
        result = check_request_eligibility("E008", "laptop", store=store, as_of=date(2026, 11, 21))
        assert result["next_eligible_date"] == "2026-12-20"
        assert result["days_until_eligible"] == 29
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == BORDERLINE_TIMING

    def test_30_days_before_is_borderline(self, store: Store) -> None:
        result = check_request_eligibility("E008", "laptop", store=store, as_of=date(2026, 11, 20))
        assert result["days_until_eligible"] == 30
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == BORDERLINE_TIMING

    def test_31_days_before_is_denied(self, store: Store) -> None:
        result = check_request_eligibility("E008", "laptop", store=store, as_of=date(2026, 11, 19))
        assert result["days_until_eligible"] == 31
        assert result["status"] == NOT_ELIGIBLE
        assert result["reason_code"] == WITHIN_REFRESH_INTERVAL

    def test_null_issue_date(self, store: Store) -> None:
        result = check_request_eligibility("E006", "laptop", store=store, as_of=AS_OF)
        assert result["status"] == REVIEW_REQUIRED
        assert result["reason_code"] == DATA_INCOMPLETE
        assert result["undated_asset_tags"] == ["LT-0006"]
        assert result["held_quantity"] == 1
        assert "next_eligible_date" not in result

    def test_null_date_on_other_item_does_not_matter(self, store: Store) -> None:
        result = check_request_eligibility("E006", "monitor", store=store, as_of=AS_OF)
        assert result["status"] == ELIGIBLE
        assert result["reason_code"] == BELOW_MAX_QUANTITY

    def test_item_lookup_is_case_insensitive(self, store: Store) -> None:
        result = check_request_eligibility("e001", "LAPTOP", store=store, as_of=AS_OF)
        assert result["status"] == ELIGIBLE
        assert result["employee_id"] == "E001"
        assert result["item"] == "laptop"
