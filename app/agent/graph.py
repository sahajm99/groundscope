"""LangGraph StateGraph implementation of the same agent (premise #4: mirror the
Etech7 stack). Same nodes as loop.py, same TraceEvent schema, streamed live via
LangGraph's custom stream writer.

NOTE: the explicit loop in loop.py is the default engine (proven). This graph is
selected with AGENT_ENGINE=langgraph and should be smoke-tested at runtime before
becoming the default — LangGraph custom streaming is unverified here.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

from app.agent import tools
from app.agent.llm import complete
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


def _emit(state: S, **kw) -> None:
    state["step"] = state.get("step", 0) + 1
    get_stream_writer()({"kind": "trace", "payload": TraceEvent(step=state["step"], **kw).to_dict()})


def route_node(state: S) -> S:
    route = "metadata" if is_metadata(state["question"]) else "documents"
    _emit(state, type="decision", input=state["question"][:200], summary=f"Routed to '{route}'.")
    return {"route": route, "collected": [], "web_ran": False}


def metadata_node(state: S) -> S:
    _emit(state, type="tool_call", tool="metadata_query", input=state["session_id"], summary="Listing documents.")
    summary, _ = tools.metadata_query(state["session_id"])
    _emit(state, type="tool_result", tool="metadata_query", summary=summary)
    get_stream_writer()({"kind": "answer", "payload": {"answer": summary, "citations": []}})
    return {"answer": summary}


def vector_node(state: S) -> S:
    _emit(state, type="tool_call", tool="vector_search", input=state["question"][:200], summary="Searching your documents.")
    summary, sources, best = tools.vector_search(state["session_id"], state["question"])
    _emit(state, type="tool_result", tool="vector_search", summary=summary, score=best)
    return {"collected": list(sources), "best": best}


def web_node(state: S) -> S:
    best = state.get("best")
    reason = (f"No documents (distance {best:.3f} > {settings.relevance_distance_threshold})."
              if best is not None else "No matching document chunks.")
    _emit(state, type="decision", summary=f"{reason} Falling back to the web.")
    _emit(state, type="tool_call", tool="web_search", input=state["question"][:200], summary="Searching the web.")
    wsummary, wsources = tools.web_search(state["question"])
    _emit(state, type="tool_result", tool="web_search", summary=wsummary)
    return {"collected": list(wsources), "web_ran": True}  # docs were weak — web only


def synth_node(state: S) -> S:
    collected = state.get("collected", [])
    best = state.get("best")
    if not state.get("web_ran") and best is not None and collected:
        _emit(state, type="decision", summary=f"Documents are relevant (distance {best:.3f} ≤ {settings.relevance_distance_threshold}).")
    if not collected:
        _emit(state, type="refusal", summary="No groundable sources found.")
        ans = ("I can't ground an answer to that in your documents or the web. "
               "Try uploading a relevant document, or email sahajm99@gmail.com.")
        get_stream_writer()({"kind": "answer", "payload": {"answer": ans, "citations": []}})
        return {"answer": ans}

    block = "\n\n".join(
        f"[{s.label}{(' — ' + s.detail) if s.kind == 'web' else ''}]\n{s.text[:1200]}" for s in collected
    )
    ans = complete(_SYNTH_SYS, f"QUESTION:\n{state['question']}\n\nSOURCES:\n{block}")
    _emit(state, type="synthesis", summary="Synthesized a grounded answer.")
    citations = [{"label": s.label, "kind": s.kind, "detail": s.detail} for s in collected]
    get_stream_writer()({"kind": "answer", "payload": {"answer": ans, "citations": citations}})
    return {"answer": ans, "citations": citations}


def _gate(state: S) -> str:
    grounded = bool(state.get("collected")) and state.get("best") is not None \
        and state["best"] <= settings.relevance_distance_threshold
    return "synth" if (grounded or not settings.web_search_configured) else "web"


def _build():
    g = StateGraph(S)
    g.add_node("route", route_node)
    g.add_node("metadata", metadata_node)
    g.add_node("vector", vector_node)
    g.add_node("web", web_node)
    g.add_node("synth", synth_node)
    g.add_conditional_edges("route", lambda s: "metadata" if s.get("route") == "metadata" else "vector",
                            {"metadata": "metadata", "vector": "vector"})
    g.add_conditional_edges("vector", _gate, {"synth": "synth", "web": "web"})
    g.add_edge("web", "synth")
    g.add_edge(START, "route")
    g.add_edge("metadata", END)
    g.add_edge("synth", END)
    return g.compile()


_graph = None


async def run_agent_graph(session_id: str, question: str) -> AsyncIterator[dict]:
    global _graph
    if _graph is None:
        _graph = _build()
    async for chunk in _graph.astream(
        {"session_id": session_id, "question": question, "step": 0}, stream_mode="custom"
    ):
        yield chunk
