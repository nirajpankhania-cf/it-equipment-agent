"""Milestone 1: a minimal MCP server with one trivial tool, served over stdio.

Run directly, or let scripts/minimal_client.py launch it as a subprocess:

    uv run python scripts/minimal_server.py

The server must never print to stdout, because stdout carries the protocol messages.
MCPServer logs to stderr.
"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("minimal-server")


@mcp.tool()
def ping(message: str) -> str:
    """Echo the message back, prefixed with 'pong: '."""
    return f"pong: {message}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
