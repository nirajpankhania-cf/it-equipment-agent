import pytest

from equipment.core import ROLE_NOT_FOUND, get_policy_limits
from equipment.store import Store

FULL_CATALOGUE = {"laptop", "monitor", "keyboard", "mouse", "headset", "docking_station"}


class TestEachRole:
    def test_standard(self, store: Store) -> None:
        result = get_policy_limits("standard", store=store)
        assert result["found"] is True
        assert result["role"] == "standard"
        assert result["limits"] == {
            "laptop": {"max_quantity": 1, "refresh_months": 48},
            "monitor": {"max_quantity": 1, "refresh_months": 36},
            "keyboard": {"max_quantity": 1, "refresh_months": 24},
            "mouse": {"max_quantity": 1, "refresh_months": 24},
            "headset": {"max_quantity": 1, "refresh_months": 24},
            "docking_station": {"max_quantity": 1, "refresh_months": 36},
        }
        assert result["not_eligible"] == []
        assert result["policy_version"] == "2026.1"
        assert result["effective_date"] == "2026-01-01"

    def test_engineer(self, store: Store) -> None:
        result = get_policy_limits("engineer", store=store)
        assert result["found"] is True
        assert result["limits"]["laptop"] == {"max_quantity": 1, "refresh_months": 36}
        assert result["limits"]["monitor"] == {"max_quantity": 2, "refresh_months": 36}
        assert set(result["limits"]) == FULL_CATALOGUE
        assert result["not_eligible"] == []

    def test_manager(self, store: Store) -> None:
        result = get_policy_limits("manager", store=store)
        assert result["found"] is True
        assert result["limits"]["laptop"] == {"max_quantity": 1, "refresh_months": 24}
        assert result["limits"]["monitor"] == {"max_quantity": 2, "refresh_months": 36}
        assert set(result["limits"]) == FULL_CATALOGUE
        assert result["not_eligible"] == []

    def test_contractor(self, store: Store) -> None:
        result = get_policy_limits("contractor", store=store)
        assert result["found"] is True
        assert set(result["limits"]) == {"monitor", "keyboard", "mouse", "headset"}
        assert result["limits"]["monitor"] == {"max_quantity": 1, "refresh_months": 36}
        assert result["not_eligible"] == ["laptop", "docking_station"]

    def test_every_role_covers_the_whole_catalogue(self, store: Store) -> None:
        for role in ("standard", "engineer", "manager", "contractor"):
            result = get_policy_limits(role, store=store)
            assert set(result["limits"]) | set(result["not_eligible"]) == FULL_CATALOGUE


class TestEdgeCases:
    def test_unknown_role(self, store: Store) -> None:
        result = get_policy_limits("intern", store=store)
        assert result == {
            "found": False,
            "error": ROLE_NOT_FOUND,
            "role": "intern",
            "message": "No policy for role 'intern'",
            "known_roles": ["contractor", "engineer", "manager", "standard"],
            "policy_version": "2026.1",
        }

    @pytest.mark.parametrize("role", ["", "   "])
    def test_empty_role(self, store: Store, role: str) -> None:
        result = get_policy_limits(role, store=store)
        assert result["found"] is False
        assert result["error"] == ROLE_NOT_FOUND

    @pytest.mark.parametrize("role", ["Engineer", "ENGINEER", "  engineer "])
    def test_role_in_different_case_or_padded(self, store: Store, role: str) -> None:
        result = get_policy_limits(role, store=store)
        assert result["found"] is True
        assert result["role"] == "engineer"
        assert result["limits"]["monitor"]["max_quantity"] == 2
