"""Plug-and-play MCP tool registry.

Reads mcp.json (same shape as Claude's .mcp.json), connects to every configured
MCP server, and returns their tools as LangChain tools the agent can bind. Add a
capability by adding a server entry to mcp.json — no code change.

Graceful: returns [] if there is no config or a server fails to load, so the
agent still runs without MCP.
"""

from __future__ import annotations

import json
import pathlib
import sys

_CONFIG = pathlib.Path("mcp.json")


async def load_mcp_tools() -> list:
    if not _CONFIG.exists():
        return []
    try:
        cfg = json.loads(_CONFIG.read_text())
    except Exception:
        return []

    # Local python stdio servers: run them with THIS interpreter (it has `mcp`).
    for srv in cfg.values():
        if isinstance(srv, dict) and srv.get("command") in ("python", "python3"):
            srv["command"] = sys.executable

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(cfg)
        return await client.get_tools()
    except Exception:
        return []
