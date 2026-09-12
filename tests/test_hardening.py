"""Production hardening for a 512 MB single-worker host (from the adversarial review)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from langchain_core.tools import StructuredTool

from app.agent import toolbus
from app.agent.toolbus import ToolBus


def _tool(name, fn):
    async def coro(**kw):
        return fn(**kw)

    return StructuredTool.from_function(coroutine=coro, name=name, description=name, infer_schema=False)


# -- 2.1 bus rebuild --------------------------------------------------------------------------
async def test_concurrent_rebuilds_load_the_servers_once(monkeypatch):
    toolbus.reset_bus()
    loads = {"n": 0}

    class FakeServers:
        tools = {"s": [_tool("hybrid_search", lambda **kw: "{}")]}
        errors: dict = {}
        loaded_at = 0.0

        async def close(self):
            pass

    async def fake_load_servers():
        loads["n"] += 1
        await asyncio.sleep(0.05)
        return FakeServers()

    import app.agent.mcp_registry as reg

    monkeypatch.setattr(reg, "load_servers", fake_load_servers)
    buses = await asyncio.gather(*(toolbus.get_bus() for _ in range(5)))
    assert loads["n"] == 1
    assert len({id(b) for b in buses}) == 1
    toolbus.reset_bus()


async def test_timeout_on_a_keepalive_tool_marks_the_bus_broken(monkeypatch):
    from app.agent import graph

    async def hang(**kw):
        await asyncio.sleep(5)
        return "{}"

    bus = ToolBus({"groundscope-retrieval": [_tool("hybrid_search", lambda **kw: "{}")]})
    bus._tools["hybrid_search"].coroutine = hang  # type: ignore[attr-defined]
    monkeypatch.setattr(graph.settings, "tool_timeout_s", 0.05)

    async def fake_get_bus():
        return bus

    monkeypatch.setattr(graph, "get_bus", fake_get_bus)
    with pytest.raises(TimeoutError):
        await graph._call("hybrid_search", session_id="s", query="q")
    assert bus.broken is True


# -- 2.3 concurrency cap on /ask ---------------------------------------------------------------
def test_ask_returns_503_when_the_concurrency_cap_is_full(monkeypatch):
    from app.api import ask as ask_api
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "k")
    monkeypatch.setattr(ask_api, "_inflight", asyncio.Semaphore(0))  # already full
    from app.main import app

    with TestClient(app) as client:
        r = client.post("/ask", json={"question": "hello"})
    assert r.status_code == 503


# -- 2.5 calculator caps + tool calls per round --------------------------------------------------
@pytest.mark.parametrize("expr", ["'a' * 10**9", "9**10**8", "2**2**30", "1" + "+1" * 300])
def test_calculator_rejects_resource_bombs(expr):
    from mcp_servers.util_server import calculator

    out = calculator(expr)
    assert out.startswith("error:"), out


def test_calculator_still_computes():
    from mcp_servers.util_server import calculator

    assert calculator("47 * 89") == "4183"
    assert calculator("(2+3)**4") == "625"


async def test_tool_calls_per_round_are_capped(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.agent import graph
    from tests.test_graph import FakeBus, run
    from tests.test_graph import _tool as graph_tool

    monkeypatch.setattr(graph, "_embed", lambda text: [0.0])
    monkeypatch.setattr(graph, "complete_json", lambda s, u, **kw: {"route": "tools", "subqueries": []})

    class Chat:
        def __init__(self, tools, model):
            self.tools = tools

        async def ainvoke(self, msgs):
            if self.tools:
                calls = [{"name": "calculator", "args": {"expression": "1+1"}, "id": f"c{i}"} for i in range(50)]
                return AIMessage(content="", tool_calls=calls)
            return AIMessage(content="done")

    monkeypatch.setattr(graph, "_chat_model", lambda tools, model: Chat(tools, model))
    monkeypatch.setattr(graph.settings, "max_tool_rounds", 0)
    bus = FakeBus({})
    bus.open_tools = lambda: [graph_tool("calculator")]
    events, answer = await run("compute", bus, monkeypatch)
    assert sum(1 for e in events if e["type"] == "tool_call") == graph.MAX_TOOL_CALLS_PER_ROUND


# -- 2.6 trace privacy + cached health ---------------------------------------------------------
async def test_metadata_trace_does_not_leak_the_session_cookie(monkeypatch):
    from app.agent import graph
    from tests.test_graph import FakeBus, run

    monkeypatch.setattr(graph, "_embed", lambda text: [0.0])
    bus = FakeBus({"metadata_query": lambda **kw: {"summary": "Documents available", "documents": []}})
    events, _ = await run("what documents do I have?", bus, monkeypatch)
    assert all(e.get("input") != "sess" for e in events)  # "sess" is the test session id


def test_health_reads_the_cached_bus_without_rebuilding(monkeypatch):
    toolbus.reset_bus()
    import app.agent.mcp_registry as reg

    async def boom():
        raise AssertionError("health must not load servers")

    monkeypatch.setattr(reg, "load_servers", boom)
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["mcp"] == {}
    toolbus.reset_bus()


# -- 2.7 eval hard checks ----------------------------------------------------------------------
def test_must_contain_uses_word_boundaries():
    from evals import judge as J

    case = {"kind": "grounded", "expect_grounded": True, "must_contain": ["11"]}
    assert J.deterministic_checks(case, "See page 2011 for details.", [{"kind": "doc"}]) == ["must_contain:11"]
    assert J.deterministic_checks(case, "It fell to 11% in 2025.", [{"kind": "doc"}]) == []


def test_citation_kind_failures_are_hard_failures():
    from evals import run_evals as R
    from tests.test_evals import CASES, TH, judge_ok

    def web_instead(case):
        return [], {"answer": "Tailwind.", "citations": [{"kind": "web"}]}, {"collected": []}

    rep = R.evaluate(CASES, web_instead, judge_ok, TH, pace=0)
    assert not rep.passed
    assert "hard_checks" in rep.summary["failed_thresholds"]
