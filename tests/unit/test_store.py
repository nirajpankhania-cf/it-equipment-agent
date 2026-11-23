import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from equipment.store import DataValidationError, Store, load_store
from tests.unit.conftest import AS_OF


def minimal_policies() -> dict[str, Any]:
    return {
        "version": "test.1",
        "effective_date": "2026-01-01",
        "catalogue": ["laptop", "monitor"],
        "policies": {
            "standard": {
                "laptop": {"max_quantity": 1, "refresh_months": 48},
                "monitor": {"max_quantity": 1, "refresh_months": 36},
            }
        },
    }


def minimal_employees() -> dict[str, Any]:
    return {
        "version": "test.1",
        "effective_date": "2026-11-01",
        "employees": [
            {
                "employee_id": "T001",
                "name": "Test Person",
                "role": "standard",
                "start_date": "2020-01-01",
                "equipment": [
                    {"item": "laptop", "issued_date": "2022-01-01", "asset_tag": "LT-T001"}
                ],
            }
        ],
    }


def write_files(
    tmp_path: Path, employees: dict[str, Any], policies: dict[str, Any]
) -> tuple[Path, Path]:
    emp_path = tmp_path / "employees.json"
    pol_path = tmp_path / "policies.json"
    emp_path.write_text(json.dumps(employees))
    pol_path.write_text(json.dumps(policies))
    return emp_path, pol_path


def load(tmp_path: Path, employees: dict[str, Any], policies: dict[str, Any]) -> Store:
    emp_path, pol_path = write_files(tmp_path, employees, policies)
    return load_store(emp_path, pol_path, as_of=AS_OF)


class TestBundledData:
    def test_loads_and_exposes_versions(self, store: Store) -> None:
        assert store.policy_version == "2026.1"
        assert store.data_version == "2026.11"
        assert len(store.employees_by_id) == 10
        assert set(store.catalogue) == {
            "laptop",
            "monitor",
            "keyboard",
            "mouse",
            "headset",
            "docking_station",
        }

    def test_all_four_roles_present(self, store: Store) -> None:
        roles = {e.role for e in store.employees_by_id.values()}
        assert roles == {"standard", "engineer", "manager", "contractor"}
        assert set(store.policies_file.policies) == roles

    def test_exactly_one_incomplete_record(self, store: Store) -> None:
        flagged = [
            (e.employee_id, r.asset_tag)
            for e in store.employees_by_id.values()
            for r in e.equipment
            if r.issued_date is None
        ]
        assert flagged == [("E006", "LT-0006")]

    def test_contractor_has_no_laptop_or_dock_entitlement(self, store: Store) -> None:
        contractor = store.policies_file.policies["contractor"]
        assert "laptop" not in contractor
        assert "docking_station" not in contractor


class TestValidation:
    def test_minimal_dataset_is_valid(self, tmp_path: Path) -> None:
        s = load(tmp_path, minimal_employees(), minimal_policies())
        assert s.get_employee("T001") is not None

    def test_duplicate_employee_id(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        duplicate = dict(employees["employees"][0], equipment=[])
        employees["employees"].append(duplicate)
        with pytest.raises(DataValidationError, match="duplicate employee_id 'T001'"):
            load(tmp_path, employees, minimal_policies())

    def test_duplicate_id_differing_only_in_case(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"].append(
            dict(employees["employees"][0], employee_id="t001", equipment=[])
        )
        with pytest.raises(DataValidationError, match="duplicate employee_id"):
            load(tmp_path, employees, minimal_policies())

    def test_role_with_no_policy(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["role"] = "intern"
        with pytest.raises(DataValidationError, match="role 'intern' with no policy"):
            load(tmp_path, employees, minimal_policies())

    def test_equipment_item_not_in_catalogue(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["equipment"][0]["item"] = "standing_desk"
        with pytest.raises(DataValidationError, match="item 'standing_desk' not in catalogue"):
            load(tmp_path, employees, minimal_policies())

    def test_duplicate_catalogue_item(self, tmp_path: Path) -> None:
        policies = minimal_policies()
        policies["catalogue"].append("Laptop")
        with pytest.raises(DataValidationError, match="catalogue contains duplicate items"):
            load(tmp_path, minimal_employees(), policies)

    def test_policy_item_not_in_catalogue(self, tmp_path: Path) -> None:
        policies = minimal_policies()
        policies["policies"]["standard"]["headset"] = {"max_quantity": 1, "refresh_months": 24}
        with pytest.raises(DataValidationError, match="role 'standard' lists item 'headset'"):
            load(tmp_path, minimal_employees(), policies)

    def test_future_issued_date(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["equipment"][0]["issued_date"] = "2026-11-24"
        with pytest.raises(DataValidationError, match="is in the future"):
            load(tmp_path, employees, minimal_policies())

    def test_issued_date_on_as_of_is_allowed(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["equipment"][0]["issued_date"] = AS_OF.isoformat()
        load(tmp_path, employees, minimal_policies())

    def test_null_issued_date_without_flag(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["equipment"][0]["issued_date"] = None
        with pytest.raises(DataValidationError, match="data_quality is not 'incomplete'"):
            load(tmp_path, employees, minimal_policies())

    def test_null_issued_date_with_flag_is_allowed(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        record = employees["employees"][0]["equipment"][0]
        record["issued_date"] = None
        record["data_quality"] = "incomplete"
        s = load(tmp_path, employees, minimal_policies())
        employee = s.get_employee("T001")
        assert employee is not None
        assert employee.equipment[0].issued_date is None

    def test_missing_issued_date_key(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        del employees["employees"][0]["equipment"][0]["issued_date"]
        with pytest.raises(DataValidationError, match="issued_date"):
            load(tmp_path, employees, minimal_policies())

    def test_unknown_field_is_rejected(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["tenure_months"] = 12
        with pytest.raises(DataValidationError, match="tenure_months"):
            load(tmp_path, employees, minimal_policies())

    def test_all_problems_reported_together(self, tmp_path: Path) -> None:
        employees = minimal_employees()
        employees["employees"][0]["role"] = "intern"
        employees["employees"][0]["equipment"][0]["item"] = "tablet"
        with pytest.raises(DataValidationError) as exc_info:
            load(tmp_path, employees, minimal_policies())
        message = str(exc_info.value)
        assert "role 'intern'" in message
        assert "item 'tablet'" in message

    def test_missing_file(self, tmp_path: Path) -> None:
        _, pol_path = write_files(tmp_path, minimal_employees(), minimal_policies())
        with pytest.raises(DataValidationError, match="nowhere.json"):
            load_store(tmp_path / "nowhere.json", pol_path, as_of=AS_OF)

    def test_invalid_json(self, tmp_path: Path) -> None:
        emp_path, pol_path = write_files(tmp_path, minimal_employees(), minimal_policies())
        emp_path.write_text("{not json")
        with pytest.raises(DataValidationError, match="employees.json"):
            load_store(emp_path, pol_path, as_of=AS_OF)

    def test_as_of_defaults_to_today(self, tmp_path: Path) -> None:
        emp_path, pol_path = write_files(tmp_path, minimal_employees(), minimal_policies())
        s = load_store(emp_path, pol_path)
        assert s.get_employee("T001") is not None
        assert date.today() >= date(2026, 11, 23)
