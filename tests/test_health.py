"""GET /health reports which MCP servers loaded (so a prod deploy shows the tool bus state)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent import toolbus
from app.agent.toolbus import ToolBus


def test_health_lists_mcp_servers_and_tools(monkeypatch):
    from langchain_core.tools import StructuredTool

    async def noop(**kw):
        return "{}"

    fake = ToolBus({"groundscope-retrieval": [StructuredTool.from_function(coroutine=noop, name="hybrid_search", description="x", infer_schema=False)]})

    async def fake_get_bus():
        return fake

    monkeypatch.setattr(toolbus, "get_bus", fake_get_bus)
    monkeypatch.setattr("app.main.get_bus", fake_get_bus, raising=False)
    from app.main import app

    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["mcp"] == {"groundscope-retrieval": ["hybrid_search"]}
    assert body["status"] == "ok"
