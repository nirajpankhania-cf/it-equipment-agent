"""Shared fixtures. Every date-dependent test uses AS_OF so results never drift."""

from datetime import date

import pytest

from equipment.store import Store, load_store

AS_OF = date(2026, 11, 23)


@pytest.fixture(scope="session")
def store() -> Store:
    return load_store(as_of=AS_OF)
