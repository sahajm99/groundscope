# High-Level Design — Groundscope

**Version:** v1 (as built & verified, 2026-06-07)
**Repo:** `groundscope/` (single FastAPI service)
**Tagline:** observable agentic RAG — watch the agent decide between your documents and the web, grounded or it refuses.

---

## 1. Purpose & context

A demo that proves agentic-AI engineering: a tool-using agent whose reasoning is
*visible* in real time, every answer grounded in a cited source (document page or
web URL) or explicitly refused. Runs entirely on free tiers. Evolution of the
`talk-to-your-data` classic-RAG project into an agent + observability.

### System context

```
            ┌──────────────────────────────┐
  Visitor ─▶│ Browser (same-origin static UI)│
            │  chat  +  LIVE TRACE panel     │
            └───────┬───────────────┬────────┘
        POST /ask (SSE)        POST /ingest, GET /documents
                    │               │
            ┌───────▼───────────────▼────────┐
            │ FastAPI + agent (single service)│
            │  tools: vector / metadata / web │
            └───┬─────────┬─────────┬─────────┘
                │         │         │
        ┌───────▼──┐ ┌────▼─────┐ ┌─▼────────┐ ┌──────────┐
        │ Groq LLM │ │ Supabase │ │ Tavily   │ │ Langfuse │
        │ (Llama)  │ │ pgvector │ │ web      │ │ traces   │
        └──────────┘ │ +metadata│ └──────────┘ └──────────┘
                     └──────────┘
        embeddings: local fastembed (in-process, no network)
```

No separate frontend deploy, no second backend. SSE stays same-origin.

---

## 2. Components

| Component | Responsibility |
|---|---|
| **Static UI** (`static/index.html`) | Chat, documents bar, and the live trace panel (SSE consumer) |
| **/ask** (`api/ask.py`) | Stream agent trace events + final grounded answer (SSE) |
| **/ingest, /documents** (`api/ingest.py`) | Upload → extract→chunk→embed→store; list session docs |
| **Agent** (`agent/loop.py`, `agent/graph.py`) | route → vector_search → relevance gate → web fallback → synthesize/refuse |
| **Tools** (`agent/tools.py`) | `vector_search`, `metadata_query` (session-scoped), `web_search` |
| **Ingestion** (`ingestion/*`) | PDF/DOCX text extraction, word-window chunking, embedding |
| **Storage** (`storage.py`) | Supabase pgvector chunks + `documents` metadata; session-scoped retrieval |
| **Observability** (`observability.py`) | Per-question Langfuse trace; each step a span (no-op without keys) |
| **Sessions** (`sessions.py`) | UUID-cookie identity, rate limit, daily cap, TTL purge |

---

## 3. Data flow

### Ingest (`POST /ingest`)
file → `extract_pages` (reject no-text) → `chunk_pages` (400-word windows w/ overlap,
page-tagged) → `embedder.embed` (local fastembed, 384-dim) → `storage.add_document`
(insert chunks as `vector(384)` + a `documents` row), all under the session UUID.

### Ask (`POST /ask`, SSE)
1. Guards: LLM-key (503), rate limit + daily cap (429), input validation (400).
2. **route** — deterministic metadata-intent detector → `documents` or `metadata`.
3. **vector_search** — embed query → pgvector cosine search scoped to (session, GLOBAL) → top-6.
4. **relevance gate** — best cosine distance ≤ 0.5 → docs are grounding; else → web.
5. **web_search** (fallback) — Tavily → snippets + URLs (replaces weak doc context).
6. **synthesize | refuse** — LLM answers strictly from sources, citing `[file p.N]` or
   `[Web: title — url]`; refuses if nothing groundable.
Every node emits one `TraceEvent` → (a) SSE to the panel, (b) Langfuse span.

---

## 4. Key decisions (as built)

| Decision | Choice | Why |
|---|---|---|
| Agent brain | **Groq Llama 3.3 70B** (OpenAI-compatible, free) | free, fast, swappable via env |
| Embeddings | **local fastembed `bge-small`, 384-dim** | no key, no ingest rate-limit, open-source; dim asserted at startup |
| Vector + metadata DB | **Supabase Postgres + pgvector** | one DB for vectors and metadata, free tier |
| Routing | **deterministic detector**, not an LLM call | small LLMs misroute; filter-before-LLM funnel |
| Relevance gate | **cosine distance ≤ 0.5** | calibrated to bge-small (relevant ~0.44, irrelevant ~0.61) |
| Agent engine | **explicit loop** default; **LangGraph** selectable | loop is proven; LangGraph mirrors the Etech7 stack, promote after smoke test |
| Web search | **Tavily** | built for LLM grounding, free 1k/mo |
| Frontend | **same-origin static from FastAPI** | avoids cross-origin SSE buffering/CORS |

---

## 5. Deployment topology

- Single Docker service (FastAPI + static UI) on Render/Fly free tier.
- Env: `LLM_API_KEY`, `DATABASE_URL` (Supabase **Session pooler**, IPv4), `TAVILY_API_KEY`,
  optional `LANGFUSE_*`. Embeddings need no key.
- **Keep-warm cron** pings `/health` (which also queries the DB) to fight host cold-start
  and the Supabase 7-day idle pause.
- Linked from the portfolio via the `liveUrl` field (renders a "Live Demo" button).

---

## 6. Non-functional

- **Cost:** $0 — free hosted LLM/search, local embeddings, free DB tier.
- **Grounding guarantee:** answers cite a source or refuse; retrieved text is treated as
  untrusted (prompt-injection isolation).
- **Reliability:** missing key / tool errors degrade gracefully (503 / inline message), never a blank crash.
- **Bounds:** `max_tool_rounds=2`, `max_tokens` capped, per-IP + global daily caps.

---

## 7. Known limitations / future

- **In-memory sessions** (lost on restart; uploads orphaned in DB). v2: persist sessions.
- **Demo caps**: 10 MB / 120 pages — a full book is rejected until caps are raised.
- **No OCR**: scanned/image PDFs are rejected (no text layer).
- **Relevance gate** is a single global threshold — per-doc/relative gating is a v2 refinement.
- **LangGraph engine** built but not yet runtime-verified.
- **v1.1**: visual grounding (PDF bounding-box highlighting), ported from talk-to-your-data.
