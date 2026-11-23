"""Placeholder test proving the src layout and editable install work in CI."""

import equipment


def test_package_is_importable() -> None:
    assert equipment.__name__ == "equipment"
