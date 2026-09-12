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
    assert answer["citations"] == [{"label": "sample.txt p.1", "kind": "doc", "detail": "p.1", "snippet": DOC["text"]}]
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
    assert answer["citations"] == [{"label": "T", "kind": "web", "detail": "https://u", "snippet": WEB["text"]}]
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


async def test_refusal_answer_carries_no_citations(monkeypatch):
    """If the model answers with the refusal sentence, the sources it declined to use must
    not be shown as citations (found by /qa: a refusal listed junk web links as sources)."""
    monkeypatch.setattr(graph, "complete", lambda system, user, **kw: "I can't ground an answer to that in your documents or the web.")
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "1 chunk", "score": 0.9, "sources": [DOC]},
                   "web_search": lambda **kw: {"summary": "1 web results", "configured": True, "sources": [WEB]}})
    events, answer = await run("xxxxxxxx", bus, monkeypatch)
    assert answer["answer"].startswith("I can't ground an answer")
    assert answer["citations"] == []
    assert any(e["type"] == "refusal" for e in events)


def _tool(name):
    from langchain_core.tools import StructuredTool

    async def coro(**kw):
        return "4"

    return StructuredTool.from_function(coroutine=coro, name=name, description=f"{name} desc", infer_schema=False)


async def test_planner_lists_only_action_tools_not_web_search(monkeypatch):
    """Found by the first real eval run: with web_search advertised as an action tool the planner
    sent plain factual questions to the ReAct worker instead of the grounded knowledge path."""
    seen = {}

    def fake_json(system, user):
        seen["system"] = system
        return {"route": "knowledge", "subqueries": []}

    monkeypatch.setattr(graph, "complete_json", fake_json)
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC]}})
    bus.open_tools = lambda: [_tool("web_search"), _tool("calculator")]
    await run("q", bus, monkeypatch)
    assert "calculator" in seen["system"]
    assert "web_search" not in seen["system"]


class FakeChat:
    """Scripted chat model: with tools bound it keeps asking for the calculator; without
    tools it answers. Raises for a given model name to exercise failover."""

    def __init__(self, tools, model, log, fail_model=None):
        self.tools, self.model, self.log, self.fail_model = tools, model, log, fail_model

    async def ainvoke(self, msgs):
        from langchain_core.messages import AIMessage

        self.log.append((self.model, bool(self.tools)))
        if self.model == self.fail_model:
            raise RuntimeError("429 rate limit")
        if self.tools:
            return AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "c1"}])
        return AIMessage(content="FINAL 4")


async def test_tool_worker_synthesizes_a_final_answer_when_rounds_run_out(monkeypatch):
    log = []
    monkeypatch.setattr(graph, "complete_json", lambda s, u: {"route": "tools", "subqueries": []})
    monkeypatch.setattr(graph, "_chat_model", lambda tools, model: FakeChat(tools, model, log))
    bus = FakeBus({})
    bus.open_tools = lambda: [_tool("calculator")]
    events, answer = await run("2+2?", bus, monkeypatch)
    assert answer["answer"] == "FINAL 4"
    assert sum(1 for e in events if e["type"] == "tool_call") == graph.settings.max_tool_rounds + 1
    assert log[-1][1] is False  # the final call had no tools bound


async def test_tool_worker_fails_over_to_the_fallback_model(monkeypatch):
    log = []
    monkeypatch.setattr(graph, "complete_json", lambda s, u: {"route": "tools", "subqueries": []})
    monkeypatch.setattr(graph, "_chat_model",
                        lambda tools, model: FakeChat(tools, model, log, fail_model=graph.settings.llm_model))
    bus = FakeBus({})
    bus.open_tools = lambda: [_tool("calculator")]
    _, answer = await run("2+2?", bus, monkeypatch)
    assert answer["answer"] == "FINAL 4"
    assert any(m == graph.settings.llm_fallback_model for m, _ in log)


async def test_synthesis_sees_the_whole_chunk(monkeypatch):
    """Found by the evals: sources were cut to 1,200 chars but a 400-word chunk is ~2,500,
    so facts in the second half of a chunk were never shown to the model (it refused)."""
    captured = {}

    def fake_complete(system, user, **kw):
        captured["user"] = user
        return "ANSWER [sample.txt p.1]"

    monkeypatch.setattr(graph, "complete", fake_complete)
    long_chunk = ("filler sentence about logistics. " * 70) + "The readmission rate fell from 17% to 11%."
    assert len(long_chunk) > 2000
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [dict(DOC, text=long_chunk)]}})
    await run("readmission?", bus, monkeypatch)
    assert "fell from 17% to 11%" in captured["user"]


async def test_citations_carry_a_snippet_and_a_stable_shape(monkeypatch):
    """AI-Mode style sources: each citation shows what was actually used, not just a label."""
    web = dict(WEB, text="Satya Nadella has been CEO of Microsoft since 2014. " * 5)
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.9, "sources": [DOC]},
                   "web_search": lambda **kw: {"summary": "", "configured": True, "sources": [web]}})
    _, answer = await run("who is the CEO?", bus, monkeypatch)
    c = answer["citations"][0]
    assert c["kind"] == "web" and c["label"] == "T" and c["detail"] == "https://u"
    assert c["snippet"].startswith("Satya Nadella has been CEO") and len(c["snippet"]) <= 200


def test_explicit_multi_question_input_is_split_without_the_llm():
    """Funnel principle: two explicit questions never depend on the planner model splitting them."""
    q = "What is Zephyr Logistics' routing engine called? Who is the current CEO of Microsoft?"
    assert graph.split_questions(q) == [
        "What is Zephyr Logistics' routing engine called?",
        "Who is the current CEO of Microsoft?",
    ]
    assert graph.split_questions("What is Tailwind?") == []
    assert graph.split_questions("Why? What is Tailwind?") == []  # fragments do not count
    assert len(graph.split_questions("A b c d? E f g h? I j k l? M n o p?")) == 3  # capped


async def test_planner_uses_the_deterministic_split(monkeypatch):
    calls = {"llm": 0}

    def fake_json(system, user):
        calls["llm"] += 1
        return {"route": "knowledge", "subqueries": []}

    monkeypatch.setattr(graph, "complete_json", fake_json)
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC]}})
    events, _ = await run("What is Zephyr's routing engine called? Who is the CEO of Microsoft?", bus, monkeypatch)
    assert calls["llm"] == 0
    assert {e["branch"] for e in events if e["type"] == "tool_call"} == {0, 1}
