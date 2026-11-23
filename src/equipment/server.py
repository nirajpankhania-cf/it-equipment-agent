"""MCP server exposing the core functions as tools. No business logic lives here.

Run over stdio:

    uv run python -m equipment.server

All logging goes to stderr. Nothing may print to stdout, because stdout carries the
protocol messages and a stray print corrupts the connection.
"""

from __future__ import annotations

import logging
import sys
from functools import cache
from typing import Any

from mcp.server.mcpserver import MCPServer

from equipment import core
from equipment.store import Store, load_store

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("equipment.server")

mcp = MCPServer("it-equipment")


@cache
def get_store() -> Store:
    """Load and validate the data once, on first use. Fails loudly if the data is bad."""
    store = load_store()
    log.info(
        "data loaded: %d employees, policy version %s, data version %s",
        len(store.employees_by_id),
        store.policy_version,
        store.data_version,
    )
    return store


@mcp.tool()
def get_employee_info(employee_id: str) -> dict[str, Any]:
    """Look up an employee by ID. Returns role, tenure in months, and equipment held with
    each item's age in months. Returns found=false with error EMPLOYEE_NOT_FOUND if the
    ID is not on file."""
    result = core.get_employee_info(employee_id, store=get_store())
    log.info("get_employee_info employee_id=%r found=%s", employee_id, result["found"])
    return result


def main() -> None:
    log.info("starting MCP server over stdio")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
