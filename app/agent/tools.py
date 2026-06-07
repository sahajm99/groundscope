"""The three agent tools. All retrieval tools are SESSION-SCOPED.

Each returns (summary, sources) where summary is a short human/LLM-readable
string and sources is a list of grounding references for the final citation.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.ingestion.embedder import get_embedder
from app import storage

try:
    from langsmith import traceable
except ImportError:  # langsmith optional
    def traceable(**_kwargs):
        def deco(fn):
            return fn
        return deco


@dataclass
class Source:
    kind: str          # "doc" | "web"
    label: str         # "<file> p.N" or page/article title
    detail: str        # url for web, page for doc
    text: str          # the snippet used for grounding


@traceable(run_type="retriever", name="hybrid_search")
def hybrid_search(session_id: str, query: str, limit: int = 6) -> tuple[str, list[Source], float | None]:
    """Hybrid retrieval (dense pgvector + sparse BM25, RRF-fused) over this
    session's docs + the global seeded corpus."""
    emb = get_embedder().embed([query])[0]
    hits, best = storage.hybrid_search(session_id, emb, query, limit=limit)
    if not hits:
        return ("No matching chunks in the uploaded documents.", [], None)
    sources = [
        Source(kind="doc", label=f"{h.file_name} p.{h.page_number}", detail=f"p.{h.page_number}", text=h.text)
        for h in hits
    ]
    dist = f"{best:.3f}" if best is not None else "n/a"
    summary = f"{len(hits)} chunks (vector+BM25, RRF-fused); best vector distance {dist} from {hits[0].file_name} p.{hits[0].page_number}."
    return (summary, sources, best)


@traceable(run_type="tool", name="metadata_query")
def metadata_query(session_id: str) -> tuple[str, list[Source]]:
    """Structured: which documents exist, page/chunk counts."""
    docs = storage.list_documents(session_id)
    if not docs:
        return ("No documents available in this session.", [])
    lines = [f"{d['file_name']}: {d['pages']} pages, {d['chunk_count']} chunks" for d in docs]
    return ("Documents available — " + "; ".join(lines), [])


@traceable(run_type="tool", name="web_search")
def web_search(query: str, limit: int = 4) -> tuple[str, list[Source]]:
    """Internet grounding via Tavily. Returns snippets + URLs."""
    if not settings.web_search_configured:
        return ("Web search is not configured.", [])
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    res = client.search(query=query, max_results=limit, search_depth="basic")
    results = res.get("results", [])
    if not results:
        return ("No web results found.", [])
    sources = [
        Source(kind="web", label=r.get("title", "web result"), detail=r.get("url", ""), text=r.get("content", ""))
        for r in results
    ]
    summary = f"{len(results)} web results; top: {results[0].get('title', '')}."
    return (summary, sources)
