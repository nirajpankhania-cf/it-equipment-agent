"""Load and validate the mock data files at startup.

Validation fails loudly on anything that would otherwise surface later as a wrong
decision: duplicate employee IDs, a role with no policy, an item outside the
catalogue, a future issued date, or a null issued date without the
``data_quality: incomplete`` flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ValidationError

from equipment.models import Employee, EmployeesFile, PoliciesFile

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_EMPLOYEES_PATH = DATA_DIR / "employees.json"
DEFAULT_POLICIES_PATH = DATA_DIR / "policies.json"


class DataValidationError(ValueError):
    """Raised when a data file fails validation. The message lists every problem found."""


def normalize_employee_id(raw: str) -> str:
    return raw.strip().upper()


def normalize_name(raw: str) -> str:
    """Normalize a role or item name for lookup."""
    return raw.strip().lower()


@dataclass(frozen=True)
class Store:
    employees_file: EmployeesFile
    policies_file: PoliciesFile
    employees_by_id: dict[str, Employee]

    @property
    def catalogue(self) -> list[str]:
        return self.policies_file.catalogue

    @property
    def policy_version(self) -> str:
        return self.policies_file.version

    @property
    def data_version(self) -> str:
        return self.employees_file.version

    def get_employee(self, employee_id: str) -> Employee | None:
        return self.employees_by_id.get(normalize_employee_id(employee_id))


def load_store(
    employees_path: Path = DEFAULT_EMPLOYEES_PATH,
    policies_path: Path = DEFAULT_POLICIES_PATH,
    *,
    as_of: date | None = None,
) -> Store:
    """Parse both files and cross-check them. Dates after ``as_of`` count as future."""
    as_of = as_of or date.today()
    employees_file = _parse(employees_path, EmployeesFile)
    policies_file = _parse(policies_path, PoliciesFile)

    problems = _check_policies(policies_file)
    problems += _check_employees(employees_file, policies_file, as_of)
    if problems:
        raise DataValidationError("\n".join(problems))

    employees_by_id = {normalize_employee_id(e.employee_id): e for e in employees_file.employees}
    return Store(employees_file, policies_file, employees_by_id)


def _parse[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return model.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DataValidationError(f"{path.name}: {exc}") from exc


def _check_policies(policies: PoliciesFile) -> list[str]:
    problems: list[str] = []
    catalogue = [normalize_name(item) for item in policies.catalogue]
    if len(set(catalogue)) != len(catalogue):
        problems.append("policies: catalogue contains duplicate items")
    for role, limits in policies.policies.items():
        for item in limits:
            if normalize_name(item) not in catalogue:
                problems.append(f"policies: role '{role}' lists item '{item}' not in catalogue")
    return problems


def _check_employees(employees: EmployeesFile, policies: PoliciesFile, as_of: date) -> list[str]:
    problems: list[str] = []
    catalogue = {normalize_name(item) for item in policies.catalogue}
    roles = {normalize_name(role) for role in policies.policies}
    seen: set[str] = set()

    for emp in employees.employees:
        eid = normalize_employee_id(emp.employee_id)
        if eid in seen:
            problems.append(f"employees: duplicate employee_id '{emp.employee_id}'")
        seen.add(eid)

        if normalize_name(emp.role) not in roles:
            problems.append(f"employees: {eid} has role '{emp.role}' with no policy")

        for rec in emp.equipment:
            if normalize_name(rec.item) not in catalogue:
                problems.append(
                    f"employees: {eid} asset {rec.asset_tag} item '{rec.item}' not in catalogue"
                )
            if rec.issued_date is not None and rec.issued_date > as_of:
                problems.append(
                    f"employees: {eid} asset {rec.asset_tag} issued_date {rec.issued_date} "
                    "is in the future"
                )
    return problems
