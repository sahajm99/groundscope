"""POST /ask — streams the agent's trace events live, then the grounded answer.

Same-origin SSE (text/event-stream): each agent step is a `trace` event; the
final message is an `answer` event. The frontend renders trace events into the
live panel as they arrive — the "watch it think" moment.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request, Response
from sse_starlette.sse import EventSourceResponse

from app import sessions
from app.agent.loop import run_agent
from app.config import settings


def _engine():
    if settings.agent_engine == "langgraph":
        from app.agent.graph import run_agent_graph

        return run_agent_graph
    return run_agent

router = APIRouter()

# Global in-flight cap: each question can fan out into 3 branches (DB, LLM, web, child
# processes); an unbounded number of them on a 512 MB host takes everyone down.
_inflight = asyncio.Semaphore(settings.max_concurrent_questions)


@router.post("/ask")
async def ask(request: Request, response: Response):
    if not settings.llm_configured:
        return Response("The agent isn't configured yet (no LLM key).", status_code=503)
    if _inflight.locked():
        return Response("The agent is busy with other questions right now. Try again in a moment.",
                        status_code=503, headers={"Retry-After": "10"})

    sid = sessions.get_or_create_session(request, response)
    ip = sessions.client_ip(request)
    if sessions.rate_limited(ip):
        return Response("Too many questions — slow down a moment.", status_code=429)
    if sessions.daily_cap_reached():
        return Response("Demo quota reached for today, try again tomorrow.", status_code=429)

    body = await request.json()
    question = (body or {}).get("question", "")
    if not isinstance(question, str) or not question.strip() or len(question) > 500:
        return Response("Ask a question (under 500 characters).", status_code=400)

    agent = _engine()
    await _inflight.acquire()

    async def event_stream():
        try:
            async for msg in agent(sid, question.strip()):
                yield {"event": msg["kind"], "data": json.dumps(msg["payload"])}
        except Exception as e:  # noqa: BLE001
            yield {"event": "answer", "data": json.dumps(
                {"answer": f"Something went wrong generating the answer ({type(e).__name__}).", "citations": []}
            )}
        finally:
            _inflight.release()

    # set-cookie from get_or_create_session must ride on the streaming response
    return EventSourceResponse(event_stream(), headers=dict(response.headers))
