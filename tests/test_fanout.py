"""Query fan-out: planner decomposition -> Send workers (parallel, per-branch gate) -> aggregate."""

from __future__ import annotations

import asyncio

from app.agent import graph
from app.agent.toolbus import ToolError
from tests.test_graph import DOC, WEB, FakeBus, _patch, run  # noqa: F401  (autouse fixture)


def _plan(subqueries):
    return lambda s, u: {"route": "knowledge", "subqueries": subqueries}


async def test_multi_part_question_fans_out_with_branch_ids(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["A", "B"]))
    started = []

    def hs(**kw):
        started.append(kw["query"])
        return {"summary": "1 chunk", "score": 0.2, "sources": [dict(DOC, label=f"sample.txt p.{kw['query']}")]}

    bus = FakeBus({"hybrid_search": hs})
    events, answer = await run("A and B?", bus, monkeypatch)
    assert sorted(started) == ["A", "B"]
    assert {e["branch"] for e in events if e["type"] == "tool_call"} == {0, 1}
    assert {c["label"] for c in answer["citations"]} == {"sample.txt p.A", "sample.txt p.B"}
    assert any(e["type"] == "decision" and "2 sub-queries" in e["summary"] for e in events)
    assert any(e["type"] == "decision" and e["summary"].startswith("Merged 2 branches") for e in events)


async def test_workers_run_concurrently(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["A", "B", "C"]))
    gate = asyncio.Event()
    active = {"n": 0, "max": 0}

    bus = FakeBus({})

    async def slow_call(name, **kw):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        if active["n"] == 3:
            gate.set()
        await asyncio.wait_for(gate.wait(), 5)  # only releases once all three are in flight
        active["n"] -= 1
        return {"summary": "1 chunk", "score": 0.2, "sources": [DOC]}

    bus.call = slow_call
    await run("A, B and C?", bus, monkeypatch)
    assert active["max"] == 3


async def test_cap_at_three_subqueries(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["1", "2", "3", "4"]))
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC]}})
    events, _ = await run("q", bus, monkeypatch)
    assert {e["branch"] for e in events if e["type"] == "tool_call"} == {0, 1, 2}


async def test_bad_decomposition_falls_back_to_the_question(monkeypatch):
    def boom(s, u):
        raise ValueError("bad json")

    monkeypatch.setattr(graph, "complete_json", boom)
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC]}})
    events, _ = await run("q", bus, monkeypatch)
    assert [c[1]["query"] for c in bus.calls] == ["q"]

    monkeypatch.setattr(graph, "complete_json", lambda s, u: {"route": "knowledge", "subqueries": ["", 7, None]})
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC]}})
    events, _ = await run("q2", bus, monkeypatch)
    assert [c[1]["query"] for c in bus.calls] == ["q2"]


async def test_mixed_branches_doc_and_web(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["docq", "webq"]))

    def hs(**kw):
        return {"summary": "", "score": 0.2 if kw["query"] == "docq" else 0.9, "sources": [DOC]}

    bus = FakeBus({"hybrid_search": hs, "web_search": lambda **kw: {"summary": "", "configured": True, "sources": [WEB]}})
    events, answer = await run("docq and webq", bus, monkeypatch)
    kinds = [c["kind"] for c in answer["citations"]]
    assert "doc" in kinds and "web" in kinds
    web_calls = [c for c in bus.calls if c[0] == "web_search"]
    assert len(web_calls) == 1 and web_calls[0][1]["query"] == "webq"
    web_events = [e for e in events if e["tool"] == "web_search"]
    assert web_events and all(e["branch"] == 1 for e in web_events)


async def test_one_branch_failing_does_not_kill_the_answer(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["ok", "bad"]))

    def hs(**kw):
        if kw["query"] == "bad":
            raise ToolError("hybrid_search: boom")
        return {"summary": "", "score": 0.2, "sources": [DOC]}

    bus = FakeBus({"hybrid_search": hs, "web_search": lambda **kw: {"summary": "", "configured": True, "sources": [WEB]}})
    events, answer = await run("ok and bad", bus, monkeypatch)
    assert answer["citations"]
    assert any("Retrieval unavailable" in e["summary"] and e["branch"] == 1 for e in events)


async def test_duplicate_sources_across_branches_are_deduped(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["A", "B"]))
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": [DOC, DOC]}})
    _, answer = await run("A and B", bus, monkeypatch)
    assert answer["citations"] == [{"label": "sample.txt p.1", "kind": "doc", "detail": "p.1"}]


async def test_branch_timeout_degrades(monkeypatch):
    monkeypatch.setattr(graph, "complete_json", _plan(["slow"]))
    monkeypatch.setattr(graph.settings, "tool_timeout_s", 0.05)
    bus = FakeBus({"web_search": lambda **kw: {"summary": "", "configured": True, "sources": [WEB]}})

    async def hang(name, **kw):
        if name == "hybrid_search":
            await asyncio.sleep(5)
        return {"summary": "", "configured": True, "sources": [WEB]}

    bus.call = hang
    events, answer = await run("slow", bus, monkeypatch)
    assert any("Retrieval unavailable (TimeoutError" in e["summary"] for e in events)
    assert answer["citations"][0]["kind"] == "web"


async def test_different_chunks_with_the_same_page_label_all_reach_synthesis(monkeypatch):
    """Found by the first real eval run: dedupe by (kind, label, detail) collapsed every chunk
    of a one-page document into one, so the chunk holding the answer was dropped."""
    monkeypatch.setattr(graph, "complete_json", _plan(["A"]))
    docs = [dict(DOC, text="chunk one about Slipstream"), dict(DOC, text="chunk two about Northstar")]
    bus = FakeBus({"hybrid_search": lambda **kw: {"summary": "", "score": 0.2, "sources": docs}})
    captured = {}

    def fake_complete(system, user, **kw):
        captured["user"] = user
        return "ANSWER [sample.txt p.1]"

    monkeypatch.setattr(graph, "complete", fake_complete)
    _, answer = await run("A", bus, monkeypatch)
    assert "chunk one about Slipstream" in captured["user"]
    assert "chunk two about Northstar" in captured["user"]
    assert answer["citations"] == [{"label": "sample.txt p.1", "kind": "doc", "detail": "p.1"}]  # one citation per page
