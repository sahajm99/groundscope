"""Plug-and-play MCP tool registry.

Reads mcp.json (same shape as Claude's .mcp.json), connects to every configured MCP
server, and returns their tools as LangChain tools the agent can bind. Add a capability
by adding a server entry to mcp.json; no code change.

Two session modes per server:
  - default: a fresh stdio session per tool call (the adapter's behaviour). Fine for
    rare, light tools (utils, web).
  - "keepalive": true -- one long-lived child process for the lifetime of the app, held
    open by a host task (anyio requires the session context to be entered and exited in
    the same task). Used for retrieval, which is on the hot path and imports numpy +
    psycopg, so spawning it per call (three at once under fan-out) is too slow and too
    heavy for a 512 MB host.

Local python servers are launched with THIS interpreter, the repo root as cwd, and the
full parent environment: mcp's stdio client otherwise forwards only PATH/HOME-style
variables, so the children would never see DATABASE_URL or TAVILY_API_KEY in Docker.

Graceful per server: a server that fails to start is logged and omitted; the others
still load.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool

log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _ROOT / "mcp.json"
_START_TIMEOUT_S = 60


def read_config(path: pathlib.Path = _CONFIG) -> tuple[dict, set[str]]:
    """Return (connections for the adapter, names of keepalive servers)."""
    if not path.exists():
        return {}, set()
    try:
        cfg = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        log.warning("mcp.json unreadable: %s", e)
        return {}, set()
    keepalive: set[str] = set()
    conns: dict = {}
    for name, srv in cfg.items():
        if not isinstance(srv, dict):
            continue
        srv = dict(srv)
        if srv.pop("keepalive", False):
            keepalive.add(name)
        if srv.get("command") in ("python", "python3"):
            srv["command"] = sys.executable
            srv["env"] = {**os.environ, **srv.get("env", {})}
            srv["cwd"] = str(_ROOT)
        conns[name] = srv
    return conns, keepalive


class _Host:
    """Owns the long-lived sessions. Enter and exit happen in one task (_run)."""

    def __init__(self, client, names: list[str]):
        self._client = client
        self._names = names
        self.tools: dict[str, list[BaseTool]] = {}
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="mcp-keepalive-host")
        await self._ready.wait()

    async def _run(self) -> None:
        from langchain_mcp_adapters.tools import load_mcp_tools

        try:
            async with AsyncExitStack() as stack:
                for name in self._names:
                    try:
                        session = await asyncio.wait_for(
                            stack.enter_async_context(self._client.session(name)), _START_TIMEOUT_S
                        )
                        self.tools[name] = await load_mcp_tools(session)
                    except Exception as e:  # noqa: BLE001
                        log.warning("MCP server %r (keepalive) failed to start: %s: %s", name, type(e).__name__, e)
                self._ready.set()
                await self._stop.wait()
        finally:
            self._ready.set()

    async def close(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 15)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()


@dataclass
class ServerSet:
    """Tools per server plus the lifecycle handle for keepalive sessions."""

    tools: dict[str, list[BaseTool]] = field(default_factory=dict)
    keepalive: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)
    _host: _Host | None = None

    async def close(self) -> None:
        if self._host is not None:
            await self._host.close()
            self._host = None


async def load_servers(config_path: pathlib.Path = _CONFIG) -> ServerSet:
    conns, keepalive = read_config(config_path)
    out = ServerSet(keepalive=keepalive)
    if not conns:
        return out
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except Exception as e:  # noqa: BLE001
        log.warning("MCP adapters unavailable: %s", e)
        return out
    client = MultiServerMCPClient(conns)

    for name in conns:
        if name in keepalive:
            continue
        try:
            out.tools[name] = await asyncio.wait_for(client.get_tools(server_name=name), _START_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001
            out.errors[name] = type(e).__name__
            log.warning("MCP server %r failed to load: %s: %s", name, type(e).__name__, e)

    ka = [n for n in conns if n in keepalive]
    if ka:
        host = _Host(client, ka)
        await host.start()
        out.tools.update(host.tools)
        for n in ka:
            if n not in host.tools:
                out.errors[n] = "start_failed"
        out._host = host
    return out


async def load_mcp_tools() -> list[BaseTool]:
    """Back-compat flat list of every loaded tool."""
    s = await load_servers()
    return [t for tools in s.tools.values() for t in tools]
