"""The Groundscope-owned MCP servers, called in-process (the registry/subprocess path is
covered in test_toolbus.py)."""

from __future__ import annotations

import json

import pytest


@pytest.mark.integration
async def test_hybrid_search_returns_structured_json(db):
    from mcp_servers.retrieval_server import hybrid_search

    out = json.loads(await hybrid_search(session_id="nosuch", query="What is Zephyr's routing engine called?"))
    assert set(out) >= {"summary", "score", "sources"}
    assert out["sources"], "seeded sample.txt should match"
    assert out["sources"][0]["kind"] == "doc"
    assert out["sources"][0]["label"].startswith("sample.txt p.")
    assert isinstance(out["score"], float)


@pytest.mark.integration
async def test_hybrid_search_accepts_precomputed_embedding(db):
    from app.ingestion.embedder import get_embedder
    from mcp_servers.retrieval_server import hybrid_search

    emb = get_embedder().embed(["Tailwind routing engine"])[0]
    out = json.loads(await hybrid_search(session_id="nosuch", query="Tailwind routing engine", query_embedding=emb))
    assert out["sources"] and "Tailwind" in out["sources"][0]["text"]


@pytest.mark.integration
async def test_hybrid_search_clamps_limit_and_validates_session(db):
    from mcp_servers.retrieval_server import hybrid_search

    out = json.loads(await hybrid_search(session_id="nosuch", query="Zephyr", limit=10_000))
    assert len(out["sources"]) <= 10
    with pytest.raises(ValueError):
        await hybrid_search(session_id="bad session;drop", query="x")


@pytest.mark.integration
async def test_metadata_query_lists_global_corpus(db):
    from mcp_servers.retrieval_server import metadata_query

    out = json.loads(await metadata_query(session_id="nosuch"))
    assert any(d["file_name"] == "sample.txt" for d in out["documents"])
    assert "sample.txt" in out["summary"]


async def test_web_search_unconfigured_is_structured(monkeypatch):
    from app.config import settings
    from mcp_servers.web_server import web_search

    monkeypatch.setattr(settings, "tavily_api_key", "")
    out = json.loads(await web_search(query="anything"))
    assert out == {"summary": "Web search is not configured.", "configured": False, "sources": []}


async def test_web_search_maps_tavily_results(monkeypatch):
    from app.config import settings
    from mcp_servers import web_server

    monkeypatch.setattr(settings, "tavily_api_key", "x")

    class FakeClient:
        def search(self, query, max_results, search_depth):
            return {"results": [{"title": "T", "url": "https://u", "content": "C"}]}

    monkeypatch.setattr(web_server, "_client", lambda: FakeClient())
    out = json.loads(await web_server.web_search(query="q"))
    assert out["configured"] is True
    assert out["sources"] == [{"kind": "web", "label": "T", "detail": "https://u", "text": "C"}]
    assert out["summary"].startswith("1 web results")
