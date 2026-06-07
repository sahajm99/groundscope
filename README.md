# Groundscope

An **observable agentic RAG** demo. Upload a document (or use the seeded book), ask a
question, and **watch the agent decide** in real time: search your docs, query metadata,
or go to the web, then answer with a citation, or refuse if it can't ground.

Built to demonstrate production agentic-AI patterns: a LangGraph agent with tools, every
step traced (live in the UI + in Langfuse), running entirely on free tiers.

> Evolution of [talk-to-your-data](../talk-to-your-data) (classic RAG) into an
> agent-with-tools + observability. Design doc: `talk-to-your-data/docs/2026-06-06-agentic-observable-demo-design.md`.

## How it works

```
Browser (same-origin static UI)
  ├─ chat
  └─ LIVE TRACE PANEL ◀── SSE: every agent step
        │ POST /ask (stream)
        ▼
FastAPI + LangGraph agent
  tools (all session-scoped):
    • vector_search   — your uploads + seeded book (pgvector, cosine)
    • metadata_query  — "what/how many docs" (Postgres)
    • web_search      — Tavily, returns URLs
  relevance gate (distance > 0.35 → web) · max 2 tool rounds · grounded-or-refuse
        │
        ▼
Supabase (pgvector + metadata) · Langfuse (traces)
```

## Stack (all free tier)

| Layer | Default | Swap |
|---|---|---|
| Agent brain | Groq Llama 3.3 70B (`LLM_*`) | any OpenAI-compatible |
| Embeddings | **local fastembed `bge-small`, 384-dim** | Gemini/OpenAI (`EMBED_PROVIDER=openai`) |
| Vector + metadata DB | Supabase Postgres + pgvector | — |
| Web search | Tavily | — |
| Observability | Langfuse Cloud | LangSmith |
| Host | Render / Fly (single service) | — |

**Embedding note:** v1 defaults to a **local** open-source embedding model (no API key,
no ingest rate-limit). The design doc pinned Gemini-768; local `bge-small` (384-dim) is a
zero-signup refinement. It's a one-env-var swap (`EMBED_PROVIDER=openai`, set
`VECTOR_DIM=768`). The dimension is asserted at startup against the pgvector column.

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements.txt
cp .env.example .env        # fill LLM_API_KEY (Groq), DATABASE_URL (Supabase), TAVILY_API_KEY
uvicorn app.main:app --reload
# open http://localhost:8000
```

## Keys you'll need (all free, no card)
- **Groq** — console.groq.com/keys (agent brain)
- **Supabase** — a free project → connection string into `DATABASE_URL`
- **Tavily** — app.tavily.com (1k searches/mo)
- **Langfuse** — cloud.langfuse.com (traces) — optional to boot

## Status
v1 in progress. Built: config, ingestion (extract/chunk/embed), pgvector + metadata storage,
FastAPI shell. Next: LangGraph agent + tools, SSE trace stream + panel, hardening, deploy.
Visual grounding (PDF box highlighting) is v1.1.
