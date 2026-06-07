"""LangGraph StateGraph agent (v2.0): planner-supervisor with a knowledge path
(grounded RAG: vector -> gate -> web -> synth) and a tool-worker path (ReAct loop
over plug-and-play MCP tools). Same TraceEvent schema, streamed via the injected
StreamWriter. LangSmith auto-instruments the graph into one nested trace.

Selected with AGENT_ENGINE=langgraph (default).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import StreamWriter

from app.agent import tools
from app.agent.llm import complete, complete_json
from app.agent.loop import _SYNTH_SYS, is_metadata
from app.agent.trace import TraceEvent
from app.config import settings


class S(TypedDict, total=False):
    session_id: str
    question: str
    step: int
    route: str
    collected: list
    best: Optional[float]
    web_ran: bool
    answer: str
    citations: list


def _emit(writer: StreamWriter, state: S, **kw) -> None:
    state["step"] = state.get("step", 0) + 1
    writer({"kind": "trace", "payload": TraceEvent(step=state["step"], **kw).to_dict()})


# ── MCP tools (plug-and-play, loaded once) ───────────────────────────
_mcp_tools = None


async def _get_mcp_tools() -> list:
    global _mcp_tools
    if _mcp_tools is None:
        from app.agent.mcp_registry import load_mcp_tools

        _mcp_tools = await load_mcp_tools()
    return _mcp_tools


# ── Node: planner / supervisor ───────────────────────────────────────
async def planner_node(state: S, writer: StreamWriter) -> S:
    q = state["question"]
    if is_metadata(q):
        route = "metadata"
    else:
        mcp = await _get_mcp_tools()
        if mcp:
            desc = "; ".join(f"{t.name}: {(t.description or '')[:80]}" for t in mcp)
            try:
                d = await asyncio.to_thread(
                    complete_json,
                    'Return JSON {"route":"tools"|"knowledge"}. Use "tools" ONLY if answering '
                    "requires one of these action tools: " + desc + '. Use "knowledge" for '
                    "questions about the user's uploaded documents or general facts.",
                    q,
                )
                route = d.get("route", "knowledge")
                route = route if route in ("tools", "knowledge") else "knowledge"
            except Exception:  # noqa: BLE001
                route = "knowledge"
        else:
            route = "knowledge"
    _emit(writer, state, type="decision", input=q[:200], summary=f"Planner routed to '{route}'.")
    return {"route": route, "collected": [], "web_ran": False}


# ── Node: tool-worker (ReAct over MCP tools) ─────────────────────────
async def tool_worker_node(state: S, writer: StreamWriter) -> S:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    mcp = await _get_mcp_tools()
    if not mcp:
        _emit(writer, state, type="refusal", summary="No action tools are configured.")
        ans = "I don't have a tool available to do that right now."
        writer({"kind": "answer", "payload": {"answer": ans, "citations": []}})
        return {"answer": ans}

    llm = ChatOpenAI(
        model=settings.llm_model, base_url=settings.llm_base_url,
        api_key=settings.llm_api_key, temperature=0,
    ).bind_tools(mcp)
    tool_map = {t.name: t for t in mcp}
    msgs = [
        SystemMessage("You are an agent that answers using the provided tools. "
                      "Call tools as needed, then give a concise final answer."),
        HumanMessage(state["question"]),
    ]
    citations: list = []
    ai = None
    for _ in range(settings.max_tool_rounds + 1):
        ai = await llm.ainvoke(msgs)
        msgs.append(ai)
        if not ai.tool_calls:
            break
        for tc in ai.tool_calls:
            _emit(writer, state, type="tool_call", tool=tc["name"],
                  input=str(tc["args"])[:200], summary=f"Calling {tc['name']}.")
            tool = tool_map.get(tc["name"])
            try:
                result = await tool.ainvoke(tc["args"]) if tool else f"unknown tool {tc['name']}"
            except Exception as e:  # noqa: BLE001
                result = f"error: {e}"
            _emit(writer, state, type="tool_result", tool=tc["name"], summary=str(result)[:200])
            citations.append({"label": tc["name"], "kind": "tool", "detail": str(tc["args"])[:120]})
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    ans = ai.content if (ai and isinstance(ai.content, str)) else str(ai.content if ai else "")
    _emit(writer, state, type="synthesis", summary="Answered using tools.")
    writer({"kind": "answer", "payload": {"answer": ans, "citations": citations}})
    return {"answer": ans}


# ── Knowledge path: metadata / vector / gate / web / synth ───────────
def metadata_node(state: S, writer: StreamWriter) -> S:
    _emit(writer, state, type="tool_call", tool="metadata_query", input=state["session_id"], summary="Listing documents.")
    summary, _ = tools.metadata_query(state["session_id"])
    _emit(writer, state, type="tool_result", tool="metadata_query", summary=summary)
    writer({"kind": "answer", "payload": {"answer": summary, "citations": []}})
    return {"answer": summary}


def vector_node(state: S, writer: StreamWriter) -> S:
    _emit(writer, state, type="tool_call", tool="hybrid_search", input=state["question"][:200], summary="Searching your documents (vector + BM25).")
    summary, sources, best = tools.hybrid_search(state["session_id"], state["question"])
    _emit(writer, state, type="tool_result", tool="hybrid_search", summary=summary, score=best,
          preview=(sources[0].text[:220] + "…") if sources else None)
    return {"collected": list(sources), "best": best}


def web_node(state: S, writer: StreamWriter) -> S:
    best = state.get("best")
    reason = (f"No documents (distance {best:.3f} > {settings.relevance_distance_threshold})."
              if best is not None else "No matching document chunks.")
    _emit(writer, state, type="decision", summary=f"{reason} Falling back to the web.")
    _emit(writer, state, type="tool_call", tool="web_search", input=state["question"][:200], summary="Searching the web.")
    wsummary, wsources = tools.web_search(state["question"])
    _emit(writer, state, type="tool_result", tool="web_search", summary=wsummary,
          links=[{"title": s.label, "url": s.detail} for s in wsources])
    return {"collected": list(wsources), "web_ran": True}


def synth_node(state: S, writer: StreamWriter) -> S:
    collected = state.get("collected", [])
    best = state.get("best")
    if not state.get("web_ran") and best is not None and collected:
        _emit(writer, state, type="decision", summary=f"Documents are relevant (distance {best:.3f} ≤ {settings.relevance_distance_threshold}).")
    if not collected:
        _emit(writer, state, type="refusal", summary="No groundable sources found.")
        ans = ("I can't ground an answer to that in your documents or the web. "
               "Try uploading a relevant document, or email sahajm99@gmail.com.")
        writer({"kind": "answer", "payload": {"answer": ans, "citations": []}})
        return {"answer": ans}

    block = "\n\n".join(
        f"[{s.label}{(' — ' + s.detail) if s.kind == 'web' else ''}]\n{s.text[:1200]}" for s in collected
    )
    ans = complete(_SYNTH_SYS, f"QUESTION:\n{state['question']}\n\nSOURCES:\n{block}")
    _emit(writer, state, type="synthesis", summary="Synthesized a grounded answer.")
    citations = [{"label": s.label, "kind": s.kind, "detail": s.detail} for s in collected]
    writer({"kind": "answer", "payload": {"answer": ans, "citations": citations}})
    return {"answer": ans, "citations": citations}


def _gate(state: S) -> str:
    grounded = bool(state.get("collected")) and state.get("best") is not None \
        and state["best"] <= settings.relevance_distance_threshold
    return "synth" if (grounded or not settings.web_search_configured) else "web"


def _plan_branch(state: S) -> str:
    r = state.get("route")
    return r if r in ("metadata", "tools") else "vector"


def _build():
    g = StateGraph(S)
    g.add_node("planner", planner_node)
    g.add_node("tools", tool_worker_node)
    g.add_node("metadata", metadata_node)
    g.add_node("vector", vector_node)
    g.add_node("web", web_node)
    g.add_node("synth", synth_node)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _plan_branch,
                            {"metadata": "metadata", "tools": "tools", "vector": "vector"})
    g.add_conditional_edges("vector", _gate, {"synth": "synth", "web": "web"})
    g.add_edge("web", "synth")
    g.add_edge("tools", END)
    g.add_edge("metadata", END)
    g.add_edge("synth", END)
    return g.compile()


_graph = None


async def run_agent_graph(session_id: str, question: str) -> AsyncIterator[dict]:
    global _graph
    if _graph is None:
        _graph = _build()
    step = 0
    async for chunk in _graph.astream(
        {"session_id": session_id, "question": question, "step": 0}, stream_mode="custom"
    ):
        if chunk.get("kind") == "trace":
            step += 1
            chunk["payload"]["step"] = step
        yield chunk
