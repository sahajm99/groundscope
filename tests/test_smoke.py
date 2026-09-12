"""End-to-end: real graph, real MCP subprocesses, real DB, real LLM (+ web). The 2.0a/2.0b DoD."""

from __future__ import annotations

import pytest

from app.agent import graph
from app.agent.toolbus import close_bus

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@pytest.fixture
async def real_bus(db, llm_keys):
    from app.agent.toolbus import get_bus

    bus = await get_bus()
    assert {"hybrid_search", "web_search"} <= set(bus.names()), bus.servers()
    yield bus
    await close_bus()


async def _ask(q):
    graph._graph = None
    events, answer = [], None
    async for chunk in graph.run_agent_graph("smokesession", q):
        if chunk["kind"] == "trace":
            events.append(chunk["payload"])
        else:
            answer = chunk["payload"]
    assert answer is not None
    return events, answer


async def test_grounds_in_docs_through_mcp(real_bus):
    events, answer = await _ask("What is the name of Zephyr Logistics' routing engine?")
    assert "tailwind" in answer["answer"].lower(), answer
    assert any(c["kind"] == "doc" and c["label"].startswith("sample.txt") for c in answer["citations"]), answer
    assert any(e["type"] == "tool_result" and e["tool"] == "hybrid_search" and e["score"] is not None for e in events)


async def test_falls_back_to_web_through_mcp(real_bus):
    from app.config import settings

    if not settings.web_search_configured:
        pytest.fail("web search key required for the smoke test")
    events, answer = await _ask("Who is the current CEO of Microsoft?")
    assert any(c["kind"] == "web" for c in answer["citations"]), answer
    assert any(e["type"] == "tool_result" and e["tool"] == "web_search" for e in events)


async def test_multi_part_question_shows_parallel_branches(real_bus):
    events, answer = await _ask(
        "What is Zephyr Logistics' routing engine called? Who is the current CEO of Microsoft?"
    )
    branches = {e.get("branch") for e in events if e["type"] == "tool_call"}
    assert len(branches) >= 2, f"expected fan-out, got branches={branches}; events={[e['summary'] for e in events]}"
    assert "tailwind" in answer["answer"].lower(), answer
