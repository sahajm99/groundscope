[![ci](https://github.com/sahajm99/groundscope/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sahajm99/groundscope/actions/workflows/ci.yml)

# Groundscope

Groundscope is a $0, open agent harness that answers questions with citations to real
sources, your documents, connected data, and the web, or refuses. The planner splits a
question into parallel branches, every branch grounds through one pluggable MCP tool bus
with tenant scoping the model cannot touch, and a golden-set eval gate in CI turns a change
red when answers stop being grounded. It is the public, free-tier twin of the Jarvis agent
architecture. The grounded, cited answer is the product; the harness is what makes it
trustworthy.

Live: https://groundscope.onrender.com · Design: `docs/v2-agentic-design.md` · Roadmap: `docs/v2-roadmap.md`

## What v2.0 proves

| Layer | What ships | Provable by |
|---|---|---|
| **Tool bus** | Three Groundscope MCP servers (`retrieval`, `web`, `utils`) plus anything in `mcp.json`, bound through one registry into one `ToolBus`. Retrieval tools take their tenant id from the orchestrator, never from the model. | `GET /health` lists the servers; `tests/test_smoke.py` grounds and web-falls-back through MCP subprocesses |
| **Orchestration** | LangGraph supervisor: planner decomposes (max 3 sub-queries), `Send` fans out to parallel async workers, each with its own corrective gate (docs, then web), `aggregate` merges. | Branch lanes in the live trace (`docs/screenshots/fanout-lanes.png`); concurrent `retrieval_worker` spans in one LangSmith run |
| **Quality gate** | 18-case golden set run through the real agent, scored by an LLM judge (faithfulness, answer relevance, context precision) plus deterministic citation checks. | The CI badge above; a regressed grounding fails the `evals` step; `eval-report.json` is attached to every run |
| **Ops** | GitHub Actions: ruff, pyright, pytest (unit, integration against a pgvector service, real-stack smoke), evals, docker build. Render deploys `main`. | The Actions tab |

## How it works

```
Browser (same-origin static UI)
  chat  ·  LIVE TRACE (branch lanes)  ◀── SSE: every agent step, tagged with its branch
        │ POST /ask
        ▼
FastAPI + LangGraph supervisor
  planner ─▶ {route, subqueries}            one LLM call; metadata questions are routed deterministically
  fanout  ══Send × N (≤3)══▶ retrieval_worker   async, in parallel
      each: embed → MCP hybrid_search(session injected) → gate (distance ≤ threshold?)
            grounded → keep docs   |   weak → MCP web_search (Tavily)
  aggregate ─▶ dedupe sources, best distance ─▶ synth (cite [file p.N] / [Web: title — url]) or refuse
  tools path: ReAct over open MCP tools + session-scoped search_documents / list_documents
        │
MCP tool bus (mcp.json → app/agent/mcp_registry.py → app/agent/toolbus.py)
  groundscope-retrieval  hybrid_search, metadata_query   (keep-alive child; JSON returns incl. score)
  groundscope-web        web_search                      (per-call child)
  groundscope-utils      calculator, current_datetime    (per-call child)
        │
Supabase Postgres + pgvector (+ BM25 tsvector, RRF fusion) · Groq (tiered router + circuit breaker) · LangSmith
```

Two rules the design keeps deterministic on purpose:
- **Structured tool returns.** `hybrid_search` returns JSON with the best cosine distance so the
  relevance gate is code, not a prompt.
- **Server-injected tenant scoping.** `session_id` is filled by the graph node. The tools the
  model can call directly (`search_documents`, `list_documents`) have no tenant field in their
  schema.

## Stack (all free tier)

| Layer | Default |
|---|---|
| Agent brain | Groq `openai/gpt-oss-120b`, fallback `openai/gpt-oss-20b` (any OpenAI-compatible endpoint via `LLM_*`) |
| Eval judge | Groq `openai/gpt-oss-20b` (`EVAL_JUDGE_MODEL`), a different model than the agent |
| Embeddings | local fastembed `BAAI/bge-small-en-v1.5`, 384-dim (no key) |
| Vector + metadata DB | Supabase Postgres + pgvector (any Postgres with pgvector works) |
| Web search | Tavily |
| Observability | LangSmith (`LANGSMITH_TRACING=true`) |
| Tool protocol | MCP (FastMCP servers, langchain-mcp-adapters client) |
| CI / hosting | GitHub Actions / Render (Docker) |

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows; use bin/activate on *nix
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env    # LLM_API_KEY (Groq), DATABASE_URL, TAVILY_API_KEY, optional LANGSMITH_*

# a local pgvector for development and tests
docker run -d --name gs-pg -p 5433:5432 -e POSTGRES_USER=gs -e POSTGRES_PASSWORD=gs \
  -e POSTGRES_DB=groundscope pgvector/pgvector:pg16
export DATABASE_URL=postgresql://gs:gs@localhost:5433/groundscope
python -m scripts.seed data/sample.txt data/corpus/*.txt   # idempotent

uvicorn app.main:app --reload            # http://localhost:8000
```

## Tests and evals

```bash
ruff check . && pyright                  # lint + typecheck
pytest -q                                # unit + integration (needs the DB) + smoke (needs LLM/Tavily keys)
pytest -q -m "not llm"                   # everything that runs without an LLM key
python -m evals.run_evals                # golden set through the real agent; writes eval-report.json
python -m evals.run_evals --only g01,g16 # a subset while iterating
```

The eval gate fails (exit 1) when any mean drops below its threshold (faithfulness 0.85,
answer relevance 0.70, context precision 0.60, deterministic checks 0.85). Never lower a
threshold to make CI pass; fix the agent or a wrong golden expectation. Golden-set format:
`evals/README.md`.

CI needs two repository secrets for the eval and smoke steps: `LLM_API_KEY` and
`TAVILY_API_KEY`. Without them the eval step exits 2 with an explicit message.

## Adding a tool

Add a server entry to `mcp.json`. Local Python servers run with the app's interpreter, the
repo root as working directory, and the full environment. Mark a hot-path server
`"keepalive": true` to keep one child process alive for the life of the app. Tool names in
`app/agent/toolbus.py:CONTEXT_TOOLS` are treated as tenant-scoped context tools; everything
else is an open tool the ReAct worker may call.

## Layout

```
app/agent/graph.py        LangGraph supervisor (planner, fan-out workers, aggregate, synth, ReAct)
app/agent/toolbus.py      ToolBus: open vs context tools, session-scoped wrappers
app/agent/mcp_registry.py loads mcp.json (env inheritance, keep-alive host)
app/agent/llm.py          tiered model router + circuit breaker (+ model override for the judge)
mcp_servers/              retrieval_server.py, web_server.py, util_server.py (FastMCP, stdio)
evals/                    golden.jsonl, judge.py, run_evals.py
tests/                    unit, integration (pgvector), smoke (real stack)
static/index.html         chat + live trace with branch lanes
docs/                     design, HLD/LLD, roadmap, decisions, progress, blocked items
```

## Status

v2.0 shipped (tool bus, fan-out, evals-in-CI). Next per the roadmap: v2.1 durable
checkpointer + HITL interrupt + guardrails, v2.2 cost accounting + Connected Tools panel,
v2.3 Postgres RLS tenancy, v2.4 Groundscope as a public MCP server. Known blockers:
`docs/BLOCKED.md`.
