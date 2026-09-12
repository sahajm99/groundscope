"""ToolBus: one registry over every MCP server; open vs context tools; scoped wrappers."""

from __future__ import annotations

import json

import pytest
from langchain_core.tools import StructuredTool, ToolException

from app.agent.toolbus import CONTEXT_TOOLS, ToolBus, ToolError, ToolUnavailable


def _fake(name, fn):
    async def coro(**kw):
        return fn(**kw)

    return StructuredTool.from_function(coroutine=coro, name=name, description=name, infer_schema=False)


def _bus():
    return ToolBus(
        {
            "groundscope-retrieval": [
                _fake("hybrid_search", lambda **kw: json.dumps({"score": 0.1, "sources": [], "summary": "ok", "seen": kw})),
                _fake("metadata_query", lambda **kw: json.dumps({"documents": [], "summary": "none", "seen": kw})),
            ],
            "groundscope-utils": [_fake("calculator", lambda **kw: "4")],
        }
    )


async def test_call_parses_json_and_passes_args():
    out = await _bus().call("hybrid_search", session_id="s1", query="q", query_embedding=[0.1])
    assert out["seen"] == {"session_id": "s1", "query": "q", "query_embedding": [0.1]}


async def test_call_unknown_tool_raises():
    with pytest.raises(ToolUnavailable):
        await _bus().call("nope")


async def test_call_non_json_raises_toolerror():
    with pytest.raises(ToolError):
        await _bus().call("calculator", expression="2+2")


async def test_tool_exception_becomes_toolerror():
    def boom(**kw):
        raise ToolException("connection to server at 10.0.0.1 failed")

    bus = ToolBus({"s": [_fake("hybrid_search", boom)]})
    with pytest.raises(ToolError):
        await bus.call("hybrid_search", session_id="s", query="q")


def test_open_tools_exclude_context_tools():
    names = {t.name for t in _bus().open_tools()}
    assert names == {"calculator"}
    assert not (names & CONTEXT_TOOLS)


def test_servers_lists_tool_names_per_server():
    assert _bus().servers() == {
        "groundscope-retrieval": ["hybrid_search", "metadata_query"],
        "groundscope-utils": ["calculator"],
    }


async def test_scoped_tools_hide_session_id_and_inject_it():
    tools = {t.name: t for t in _bus().scoped_tools("tenant-A", embed=lambda q: [0.5])}
    assert set(tools) == {"search_documents", "list_documents"}
    for t in tools.values():
        assert "session_id" not in t.args
    out = json.loads(await tools["search_documents"].ainvoke({"query": "hello"}))
    assert out["seen"]["session_id"] == "tenant-A"
    assert out["seen"]["query_embedding"] == [0.5]
    out = json.loads(await tools["list_documents"].ainvoke({}))
    assert out["seen"]["session_id"] == "tenant-A"


@pytest.mark.integration
async def test_registry_loads_all_three_servers_with_env_inheritance(db):
    """Real subprocesses via mcp.json. The retrieval child must see DATABASE_URL (mcp's stdio
    client strips the environment by default) and must be able to import the app package."""
    from app.agent.mcp_registry import load_servers

    servers = await load_servers()
    try:
        names = {t.name for tools in servers.tools.values() for t in tools}
        assert {"hybrid_search", "metadata_query", "web_search", "calculator", "current_datetime"} <= names
        bus = ToolBus(servers.tools)
        out = await bus.call("hybrid_search", session_id="nosuch", query="Zephyr routing engine")
        assert out["sources"], "child process reached the test DB"
        assert (await bus.call("metadata_query", session_id="nosuch"))["documents"]
        out = json.loads(await bus.call_raw("calculator", expression="6*7"))
        assert out == 42
    finally:
        await servers.close()


@pytest.mark.integration
async def test_keepalive_retrieval_session_is_reused(db, monkeypatch):
    """groundscope-retrieval is marked keepalive in mcp.json: one session, many calls;
    a per-call server (utils) opens a session per call."""
    import langchain_mcp_adapters.client as client_mod
    import langchain_mcp_adapters.tools as tools_mod

    from app.agent.mcp_registry import load_servers

    opened: list[str] = []
    real = tools_mod.create_session

    def counting(connection):
        opened.append(connection["args"][0])
        return real(connection)

    monkeypatch.setattr(tools_mod, "create_session", counting)
    monkeypatch.setattr(client_mod, "create_session", counting)

    servers = await load_servers()
    try:
        assert "groundscope-retrieval" in servers.keepalive
        bus = ToolBus(servers.tools)
        await bus.call("metadata_query", session_id="a")
        await bus.call("metadata_query", session_id="b")
        await bus.call_raw("calculator", expression="1+1")
        await bus.call_raw("calculator", expression="2+2")
    finally:
        await servers.close()
    assert opened.count("mcp_servers/retrieval_server.py") == 1  # one long-lived session
    assert opened.count("mcp_servers/util_server.py") == 3  # list + 2 calls
