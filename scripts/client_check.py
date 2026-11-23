"""Milestone 2 check: call get_employee_info on the real server over stdio.

    uv run python scripts/client_check.py

Launches src/equipment/server.py as a subprocess, lists its tools, then looks up one
known and one unknown employee ID and prints both structured results.
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

KNOWN_ID = "E001"
UNKNOWN_ID = "E999"


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "equipment.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])

            for employee_id in (KNOWN_ID, UNKNOWN_ID):
                result = await session.call_tool("get_employee_info", {"employee_id": employee_id})
                print(f"\nget_employee_info({employee_id!r}) is_error={result.is_error}")
                print(json.dumps(result.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
