"""groundscope-retrieval: Groundscope's retrieval as an MCP server (stdio).

Context tools: the orchestrator injects session_id; the LLM never chooses a tenant.
Structured JSON returns so the deterministic corrective-RAG gate can read `score`.
The orchestrator passes a precomputed `query_embedding` so this subprocess never loads
a second copy of the embedding model on a 512 MB host; external callers may omit it and
the server embeds lazily.
"""

from __future__ import annotations

import json
import re

try:
    import _paths  # noqa: F401  (sys.path bootstrap when run as a script)
except ImportError:
    from mcp_servers import _paths  # noqa: F401

import anyio
from mcp.server.fastmcp import FastMCP

from app import storage

mcp = FastMCP("groundscope-retrieval")

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_LIMIT = 10


def _check_session(session_id: str) -> str:
    if not _SESSION_RE.match(session_id or ""):
        raise ValueError("invalid session_id")
    return session_id


def _embed(query: str) -> list[float]:
    from app.ingestion.embedder import get_embedder

    return get_embedder().embed([query])[0]


@mcp.tool()
async def hybrid_search(
    session_id: str, query: str, limit: int = 6, query_embedding: list[float] | None = None
) -> str:
    """Hybrid (pgvector + BM25, RRF-fused) search over the session's documents plus the
    global corpus. Returns JSON {summary, score, sources:[{kind,label,detail,text}]};
    score is the best cosine distance (lower is closer)."""
    sid = _check_session(session_id)
    limit = max(1, min(int(limit), MAX_LIMIT))
    emb = query_embedding if query_embedding else await anyio.to_thread.run_sync(_embed, query)
    hits, best = await anyio.to_thread.run_sync(storage.hybrid_search, sid, emb, query, limit)
    sources = [
        {"kind": "doc", "label": f"{h.file_name} p.{h.page_number}", "detail": f"p.{h.page_number}", "text": h.text}
        for h in hits
    ]
    if not hits:
        summary = "No matching chunks in the uploaded documents."
    else:
        dist = f"{best:.3f}" if best is not None else "n/a"
        summary = (f"{len(hits)} chunks (vector+BM25, RRF-fused); best vector distance "
                   f"{dist} from {hits[0].file_name} p.{hits[0].page_number}.")
    return json.dumps({"summary": summary, "score": best, "sources": sources})


@mcp.tool()
async def metadata_query(session_id: str) -> str:
    """List the documents available to a session (uploads + global corpus).
    Returns JSON {summary, documents:[{file_name,pages,chunk_count,uploaded_at}]}."""
    sid = _check_session(session_id)
    docs = await anyio.to_thread.run_sync(storage.list_documents, sid)
    if not docs:
        summary = "No documents available in this session."
    else:
        summary = "Documents available — " + "; ".join(
            f"{d['file_name']}: {d['pages']} pages, {d['chunk_count']} chunks" for d in docs
        )
    return json.dumps({"summary": summary, "documents": docs})


if __name__ == "__main__":
    mcp.run(transport="stdio")
