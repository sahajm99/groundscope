"""One tool bus over every MCP server in mcp.json.

Two tool classes (LLD section 1.3):
  context tools (hybrid_search, metadata_query): invoked by graph nodes with a
      server-injected session_id. For the ReAct worker they are exposed as
      session-scoped wrappers whose schema has NO session_id, so the model can ask to
      "search my documents" but can never name another tenant.
  open tools (everything else): the LLM may call them freely.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from app.agent.mcp_registry import ServerSet

log = logging.getLogger(__name__)

CONTEXT_TOOLS = frozenset({"hybrid_search", "metadata_query"})


class ToolUnavailable(RuntimeError):
    """No such tool is loaded (server missing or failed to start)."""


class ToolError(RuntimeError):
    """The tool ran and failed, or returned something unusable. The message is safe to
    log but NOT to show to visitors (it may contain DB hosts, keys, etc.)."""


class _SearchArgs(BaseModel):
    query: str = Field(description="What to look for in the user's documents.")


class _NoArgs(BaseModel):
    pass


class ToolBus:
    def __init__(self, tools_by_server: dict[str, list[BaseTool]]):
        self._by_server = {s: list(ts) for s, ts in tools_by_server.items()}
        self._tools: dict[str, BaseTool] = {t.name: t for ts in self._by_server.values() for t in ts}
        self.broken = False

    # -- introspection --------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self._tools)

    def servers(self) -> dict[str, list[str]]:
        return {s: [t.name for t in ts] for s, ts in self._by_server.items()}

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def open_tools(self) -> list[BaseTool]:
        return [t for n, t in self._tools.items() if n not in CONTEXT_TOOLS]

    # -- invocation -----------------------------------------------------
    async def call_raw(self, name: str, **args: Any) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolUnavailable(name)
        try:
            raw = await tool.ainvoke(args)
        except ToolException as e:
            raise ToolError(f"{name}: {e}") from e
        except Exception as e:  # transport-level: child died, pipe closed, timeout...
            self.broken = True
            log.warning("MCP transport failure on %s: %s: %s", name, type(e).__name__, e)
            raise ToolError(f"{name}: {type(e).__name__}: {e}") from e
        if isinstance(raw, list):
            raw = "".join(str(x) for x in raw)
        return str(raw)

    async def call(self, name: str, **args: Any) -> dict:
        raw = await self.call_raw(name, **args)
        try:
            out = json.loads(raw)
        except (TypeError, ValueError) as e:
            raise ToolError(f"{name}: non-JSON result: {raw[:120]}") from e
        if not isinstance(out, dict):
            raise ToolError(f"{name}: expected a JSON object, got {type(out).__name__}")
        return out

    # -- session-scoped wrappers for the LLM ----------------------------
    def scoped_tools(self, session_id: str, embed: Callable[[str], list[float]]) -> list[BaseTool]:
        out: list[BaseTool] = []
        if "hybrid_search" in self._tools:

            async def search_documents(query: str) -> str:
                emb = await asyncio.to_thread(embed, query)  # embedding is CPU work; keep the loop free
                res = await self.call("hybrid_search", session_id=session_id, query=query, query_embedding=emb)
                return json.dumps(res)

            out.append(
                StructuredTool.from_function(
                    coroutine=search_documents,
                    name="search_documents",
                    args_schema=_SearchArgs,
                    description="Search the user's uploaded documents and the sample corpus (hybrid vector + keyword).",
                )
            )
        if "metadata_query" in self._tools:

            async def list_documents() -> str:
                return json.dumps(await self.call("metadata_query", session_id=session_id))

            out.append(
                StructuredTool.from_function(
                    coroutine=list_documents,
                    name="list_documents",
                    args_schema=_NoArgs,
                    description="List the documents available to the user (names, pages, chunk counts).",
                )
            )
        return out


_bus: ToolBus | None = None
_servers: ServerSet | None = None


async def get_bus() -> ToolBus:
    """Process-wide bus, built on first use; rebuilt once if a keepalive transport broke."""
    global _bus, _servers
    if _bus is None or _bus.broken:
        from app.agent.mcp_registry import load_servers

        if _servers is not None:
            await _servers.close()
        _servers = await load_servers()
        _bus = ToolBus(_servers.tools)
    return _bus


async def close_bus() -> None:
    global _bus, _servers
    if _servers is not None:
        await _servers.close()
    _bus, _servers = None, None


def reset_bus() -> None:
    """Test seam: forget the cached bus without closing (use close_bus for real sessions)."""
    global _bus, _servers
    _bus, _servers = None, None
