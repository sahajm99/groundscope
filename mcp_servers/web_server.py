"""groundscope-web: web grounding (Tavily) as an MCP server (stdio). An open tool."""

from __future__ import annotations

import json

try:
    import _paths  # noqa: F401  (sys.path bootstrap when run as a script)
except ImportError:
    from mcp_servers import _paths  # noqa: F401

import anyio
from mcp.server.fastmcp import FastMCP

from app.config import settings

mcp = FastMCP("groundscope-web")
MAX_LIMIT = 8


def _client():
    from tavily import TavilyClient

    return TavilyClient(api_key=settings.tavily_api_key)


def _search(query: str, limit: int) -> dict:
    return _client().search(query=query, max_results=limit, search_depth="basic")


@mcp.tool()
async def web_search(query: str, limit: int = 4) -> str:
    """Search the public web (Tavily). Returns JSON
    {summary, configured, sources:[{kind:'web', label(title), detail(url), text(snippet)}]}."""
    if not settings.web_search_configured:
        return json.dumps({"summary": "Web search is not configured.", "configured": False, "sources": []})
    limit = max(1, min(int(limit), MAX_LIMIT))
    res = await anyio.to_thread.run_sync(_search, query, limit)
    results = res.get("results", [])
    sources = [
        {"kind": "web", "label": r.get("title", "web result"), "detail": r.get("url", ""), "text": r.get("content", "")}
        for r in results
    ]
    summary = f"{len(results)} web results; top: {results[0].get('title', '')}." if results else "No web results found."
    return json.dumps({"summary": summary, "configured": True, "sources": sources})


if __name__ == "__main__":
    mcp.run(transport="stdio")
