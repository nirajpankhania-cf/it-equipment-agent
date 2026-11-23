"""Milestone 2 and 3 check: call each tool on the real server over stdio.

    uv run python scripts/client_check.py

Launches src/equipment/server.py as a subprocess, lists its tools, then calls each
tool with one known and one unknown value and prints the structured results.
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CALLS = [
    ("get_employee_info", {"employee_id": "E001"}),
    ("get_employee_info", {"employee_id": "E999"}),
    ("get_policy_limits", {"role": "contractor"}),
    ("get_policy_limits", {"role": "intern"}),
]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "equipment.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])

            for tool_name, arguments in CALLS:
                result = await session.call_tool(tool_name, arguments)
                print(f"\n{tool_name}({arguments}) is_error={result.is_error}")
                print(json.dumps(result.structured_content, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
