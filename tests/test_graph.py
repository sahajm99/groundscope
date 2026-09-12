"""The LangGraph supervisor on the tool bus, with the LLM and the bus faked.
No DB, no LLM, no subprocesses: these run everywhere."""

from __future__ import annotations

import pytest

from app.agent import graph
from app.agent.toolbus import ToolBus, ToolError

DOC = {"kind": "doc", "label": "sample.txt p.1", "detail": "p.1", "text": "Tailwind is the routing engine."}
WEB = {"kind": "web", "label": "T", "detail": "https://u", "text": "web text"}


class FakeBus(ToolBus):
    def __init__(self, handlers):
        super().__init__({})
        self.handlers = handlers
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name, **args):
        self.calls.append((name, args))
        h = self.handlers.get(name)
        if h is None:
            raise ToolError(f"{name}: connection to server at 10.0.0.1 failed: FATAL secret-host")
        return h(**args)

    def open_tools(self):
        return []


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(graph, "_embed", lambda text: [0.0])
    monkeypatch.setattr(graph.settings, "tavily_api_key", "x")
    monkeypatch.setattr(graph, "complete", lambda system, user, **kw: "ANSWER [sample.txt p.1]")
    monkeypatch.setattr(graph, "complete_json", lambda system, user: {"route": "knowledge", "subqueries": []})
    graph._graph = None


async def run(question, bus, monkeypatch):
    async def fake_get_bus():
        return bus

    monkeypatch.setattr(graph, "get_bus", fake_get_bus)
    events, answer = [], None
    async for chunk in graph.run_agent_graph("sess", question):
        if chunk["kind"] == "trace":
            events.append(chunk["payload"])
        else:
            answer = chunk["payload"]
    assert answer is not None, "graph must always emit an answer"
    return events, answer


async def test_grounded_path_uses_mcp_hybrid_search(monkeypatch):
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "1 chunk", "score": 0.2, "sources": [DOC]}})
    events, answer = await run("What is Tailwind?", bus, monkeypatch)
    assert [c[0] for c in bus.calls] == ["hybrid_search"]
    assert bus.calls[0][1]["session_id"] == "sess"
    assert bus.calls[0][1]["query_embedding"] == [0.0]
    assert answer["citations"] == [{"label": "sample.txt p.1", "kind": "doc", "detail": "p.1"}]
    assert any(e["type"] == "tool_result" and e["tool"] == "hybrid_search" and e["score"] == 0.2 for e in events)


async def test_weak_docs_fall_back_to_web_via_mcp(monkeypatch):
    bus = FakeBus(
        {
            "hybrid_search": lambda **kw: {"summary": "1 chunk", "score": 0.9, "sources": [DOC]},
            "web_search": lambda **kw: {"summary": "1 web results", "configured": True, "sources": [WEB]},
        }
    )
    events, answer = await run("Who is the CEO of Microsoft?", bus, monkeypatch)
    assert [c[0] for c in bus.calls] == ["hybrid_search", "web_search"]
    assert answer["citations"] == [{"label": "T", "kind": "web", "detail": "https://u"}]
    assert any(e["type"] == "tool_result" and e["tool"] == "web_search" and e["links"] == [{"title": "T", "url": "https://u"}] for e in events)


async def test_retrieval_error_degrades_to_web_without_leaking_the_error(monkeypatch):
    bus = FakeBus({"web_search": lambda **kw: {"summary": "1 web results", "configured": True, "sources": [WEB]}})
    events, answer = await run("anything", bus, monkeypatch)
    assert any(e["type"] == "decision" and "Retrieval unavailable" in e["summary"] for e in events)
    assert not any("secret-host" in (e["summary"] or "") for e in events), "raw tool errors must not reach the panel"
    assert answer["citations"][0]["kind"] == "web"


async def test_nothing_groundable_refuses(monkeypatch):
    bus = FakeBus(
        {
            "hybrid_search": lambda **kw: {"summary": "none", "score": None, "sources": []},
            "web_search": lambda **kw: {"summary": "No web results found.", "configured": True, "sources": []},
        }
    )
    events, answer = await run("zzz", bus, monkeypatch)
    assert any(e["type"] == "refusal" for e in events)
    assert answer["answer"].startswith("I can't ground an answer")
    assert answer["citations"] == []


async def test_weak_docs_without_web_key_still_answer_from_docs(monkeypatch):
    monkeypatch.setattr(graph.settings, "tavily_api_key", "")
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "1 chunk", "score": 0.9, "sources": [DOC]}})
    events, answer = await run("q", bus, monkeypatch)
    assert [c[0] for c in bus.calls] == ["hybrid_search"]
    assert answer["citations"][0]["kind"] == "doc"


async def test_metadata_route_uses_mcp(monkeypatch):
    bus = FakeBus({"metadata_query": lambda **kw: {"summary": "Documents available - sample.txt: 1 pages, 1 chunks", "documents": []}})
    events, answer = await run("what documents do I have?", bus, monkeypatch)
    assert bus.calls == [("metadata_query", {"session_id": "sess"})]
    assert answer["answer"].startswith("Documents available")


async def test_steps_are_renumbered_monotonically(monkeypatch):
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "1 chunk", "score": 0.2, "sources": [DOC]}})
    events, _ = await run("q", bus, monkeypatch)
    assert [e["step"] for e in events] == list(range(1, len(events) + 1))
