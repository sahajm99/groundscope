"""Langfuse observability. No-op when keys are absent, so it never blocks the demo.

One Langfuse trace per question; each agent step becomes a span sharing the
exact TraceEvent schema the SSE panel uses, so the live panel and the dashboard
show the same thing.
"""

from __future__ import annotations

from app.config import settings
from app.agent.trace import TraceEvent


class _NullTrace:
    def event(self, _ev: TraceEvent) -> None: ...
    def finish(self, _answer: str) -> None: ...


class _LangfuseTrace:
    def __init__(self, client, question: str):
        self._t = client.trace(name="groundscope.ask", input={"question": question})

    def event(self, ev: TraceEvent) -> None:
        self._t.span(
            name=f"{ev.type}:{ev.tool or 'agent'}",
            input={"input": ev.input},
            output={"summary": ev.summary, "score": ev.score},
            metadata={"step": ev.step, "ms": ev.ms},
        )

    def finish(self, answer: str) -> None:
        self._t.update(output={"answer": answer})


_client = None


def _get_client():
    global _client
    if _client is None:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _client


def start_trace(question: str):
    """Return a trace handle (.event / .finish). Null object if not configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return _NullTrace()
    try:
        return _LangfuseTrace(_get_client(), question)
    except Exception:  # noqa: BLE001
        return _NullTrace()
