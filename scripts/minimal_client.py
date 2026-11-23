"""Milestone 1: an MCP client that launches minimal_server.py over stdio and calls ping.

uv run python scripts/minimal_client.py
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).with_name("minimal_server.py")


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools:", [tool.name for tool in tools.tools])

            result = await session.call_tool("ping", {"message": "hello"})
            for block in result.content:
                if block.type == "text":
                    print("ping result:", block.text)


if __name__ == "__main__":
    asyncio.run(main())
