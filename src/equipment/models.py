"""Pydantic schemas for the mock employee and policy data files."""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EquipmentRecord(BaseModel):
    """One unit of equipment held by an employee."""

    model_config = ConfigDict(extra="forbid")

    item: str
    issued_date: date | None
    asset_tag: str = Field(min_length=1)
    data_quality: Literal["incomplete"] | None = None

    @model_validator(mode="after")
    def _null_date_requires_flag(self) -> Self:
        if self.issued_date is None and self.data_quality != "incomplete":
            raise ValueError(
                f"asset {self.asset_tag}: issued_date is null but data_quality is not 'incomplete'"
            )
        return self


class Employee(BaseModel):
    model_config = ConfigDict(extra="forbid")

    employee_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    start_date: date
    equipment: list[EquipmentRecord] = Field(default_factory=list)


class EmployeesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    effective_date: date
    employees: list[Employee]


class PolicyLimit(BaseModel):
    """Entitlement for one item under one role."""

    model_config = ConfigDict(extra="forbid")

    max_quantity: int = Field(ge=1)
    refresh_months: int = Field(ge=1)


class PoliciesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    effective_date: date
    catalogue: list[str] = Field(min_length=1)
    policies: dict[str, dict[str, PolicyLimit]]
