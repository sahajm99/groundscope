# High-Level Design — Groundscope v2 (Agent Harness)

**Version:** v2 (design) · **Date:** 2026-06-08
**Repo:** `groundscope/` (single FastAPI service + static UI)
**Derives from:** `docs/v2-agentic-design.md` (master design)
**Tagline:** a $0, runnable **agent harness** — orchestration, a uniform MCP tool bus,
control, observability, evals, and ops — demonstrated through grounded Q&A.

---

## 1. Purpose & context

v2 reframes Groundscope from "observable agentic RAG" to an **agent harness**. The
grounded-Q&A task is the demonstration workload; the engineering on display is the
six-layer harness (orchestration, tools, control/safety, observability, quality gate,
operations). Everything runs on free tiers.

### System context (v2)

```
            ┌─────────────────────────────────────────────┐
  Visitor ─▶│ Browser (same-origin static UI)             │
            │  chat  ·  LIVE TRACE (fan-out branches)      │
            │  CONNECTED TOOLS panel  ·  cost line         │
            └───────┬───────────────┬─────────────┬────────┘
        POST /ask (SSE)      /ingest /documents   GET /tools
                    │               │             │
            ┌───────▼───────────────▼─────────────▼────────┐
            │ FastAPI  +  LangGraph supervisor (harness)    │
            │  planner → fan-out workers → gate → synth      │
            │  HITL interrupt · guardrails · budgets         │
            └──┬──────────────┬───────────────┬─────────────┘
               │ binds ONE    │               │ checkpoint/resume
        ┌──────▼──────────┐   │        ┌───────▼─────────────┐
        │ MCP TOOL BUS    │   │        │ Supabase Postgres    │
        │ groundscope-    │   │        │  pgvector + metadata │
        │  retrieval ◀────┼───┘        │  + LangGraph         │
        │  web · utils    │            │    checkpointer (RLS)│
        │ (+ external MCP)│            └──────────────────────┘
        └──┬───────┬──────┘
           │       │
     ┌─────▼──┐ ┌──▼──────┐ ┌──────────┐ ┌──────────┐
     │ Groq   │ │ Tavily/ │ │ LangSmith│ │ GitHub   │
     │ LLM    │ │ Gemini  │ │ traces + │ │ Actions  │
     │ router │ │ web     │ │ cost     │ │ CI: lint/│
     └────────┘ └─────────┘ └──────────┘ │ type/test│
   embeddings: local fastembed (in-proc)  │ /eval/build
                                          └──────────┘
```

One service, one DB, same-origin SSE. The harness is the FastAPI + LangGraph core; the
tool bus is MCP; CI/CD is GitHub Actions; deploy is Render.

---

## 2. Components (v2)

| Component | Responsibility | v1→v2 |
|---|---|---|
| **Static UI** (`static/index.html`) | chat, live trace (fan-out branches), **Connected Tools panel**, cost line | reskinned + tools panel |
| **/ask** (`api/ask.py`) | stream trace events + grounded answer (SSE) | + fan-out + cost events |
| **/tools** (`api/tools.py`) | list connected MCP servers + their tools (for the UI panel) | **new** |
| **/ingest, /documents** | upload→extract→chunk→embed→store; list docs | unchanged |
| **Supervisor graph** (`agent/graph.py`) | planner → fan-out workers → aggregator → gate → synth/refuse; HITL interrupt | expanded |
| **MCP tool bus** (`agent/mcp_registry.py` + `mcp_servers/*`) | one registry binding `groundscope-retrieval`, `-web`, `-utils`, + external | unified |
| **MCP servers** (`mcp_servers/retrieval_server.py`, `web_server.py`, `util_server.py`) | hybrid_search/metadata (structured JSON), web/gemini, calc/datetime/fetch | retrieval+web **new** |
| **LLM router** (`agent/llm.py`) | tiered model failover + circuit breaker, cost/token accounting | + cost accounting |
| **Checkpointer** | LangGraph Postgres checkpointer (durable, HITL resume) | **new** |
| **Guardrails** (`agent/guards.py`) | input (injection/PII) + output checks; budgets | **new** |
| **Eval harness** (`evals/`) | golden set + Ragas/DeepEval metrics, CI-run | **new** |
| **Observability** (`observability.py`) | LangSmith trace per question; cost/latency per node | + cost |
| **Sessions/Tenancy** (`sessions.py`) | cookie identity, rate/daily caps; **Postgres RLS** per-tenant | + RLS |
| **CI/CD** (`.github/workflows/`) | lint → typecheck → test → eval → docker build; preview deploy | **new** |

---

## 3. Data flow

### Ingest (`POST /ingest`) — unchanged from v1
file → `extract_pages` → `chunk_pages` (400-word windows, page-tagged) →
`embedder.embed` (local fastembed, 384-dim) → `storage.add_document` (chunks as
`vector(384)` + a `documents` row), under the session/tenant id.

### Ask (`POST /ask`, SSE) — v2 with fan-out
1. **Guards** — LLM-key (503), rate/daily cap (429), input validation (400), input
   guardrail (prompt-injection/PII screen).
2. **Planner** — classify intent; for complex questions, **decompose into sub-queries**.
3. **Fan-out** — run sub-queries as **parallel workers** (`Send` API). Each worker
   invokes the `groundscope-retrieval` MCP tool (session injected) and/or web tools.
   "Open" utility tools (calc/datetime/fetch) are available to the ReAct worker.
4. **Aggregate** — merge worker results; dedupe sources.
5. **Gate (corrective RAG)** — best vector distance ≤ threshold → ground in docs; else
   → web fallback (Tavily/Gemini). Deterministic, on top of the tool bus.
6. **Synthesize | refuse** — LLM answers strictly from sources, citing `[file p.N]` or
   `[Web: title — url]`; refuses if nothing groundable. Output guardrail screens the
   answer.
7. **(Optional) HITL** — before a sensitive tool action, `interrupt()`; the run pauses
   to the checkpointer and resumes on approve/edit/reject.

Every node emits a `TraceEvent` → (a) SSE panel, (b) LangSmith span. v2 events add a
`branch` id (which fan-out worker) and `cost`/`tokens`/`ms`.

---

## 4. Key decisions (v2)

| Decision | Choice | Why |
|---|---|---|
| Product framing | **agent harness; RAG = demo workload** | answer quality bounded on purpose; harness is the artifact |
| Tool architecture | **unify on MCP** (one bus, 3 owned servers + external) | composable, plug-and-play, UI-exposable, "Groundscope as MCP server" falls out |
| Retrieval tool I/O | **structured JSON** (`score` + `sources`) | the deterministic gate needs the distance score |
| Tenant scoping | **server-injected `session_id`**, not LLM-chosen | isolation must be deterministic, not model-controlled |
| Grounded-RAG gate | **deterministic orchestration on top of the bus** | corrective RAG is a feature, not a free ReAct side-effect |
| Orchestration | **LangGraph supervisor + `Send` fan-out** | parallel sub-queries; visible branches; mirrors production patterns |
| Durability | **LangGraph Postgres checkpointer** | crash/pause/HITL resume; free on Supabase |
| Engine | **`AGENT_ENGINE=langgraph` (default)** | nested LangSmith traces; loop kept as fallback |
| Quality | **evals-in-CI** (golden set, fail on regression) | a quality contract, the DevOps proof |
| Ops | **GitHub Actions** lint→type→test→eval→build + preview deploys | the CI/CD flex |
| Hosting | **Render** (Docker free) | current live deploy, no card |

---

## 5. Deployment topology

- Single Docker service (FastAPI + static UI + bundled MCP servers) on **Render** free.
- **GitHub Actions** on every PR: `ruff` lint → `mypy`/`pyright` typecheck → `pytest`
  → **eval** (golden set, threshold gate) → `docker build`. Merge to `main` triggers
  the Render deploy; PRs get a preview check.
- Env: `LLM_API_KEY`, `DATABASE_URL` (Supabase session pooler, IPv4), `TAVILY_API_KEY`,
  optional `GEMINI_API_KEY`, `LANGSMITH_*`. Embeddings need no key.
- Dockerfile copies `mcp_servers/` + `mcp.json` and prefetches the embed model.
- Linked from the portfolio via the project `liveUrl` ("Live Demo" button).

---

## 6. Non-functional

- **Cost:** $0 — free hosted LLM/search, local embeddings, free DB, free CI.
- **Grounding guarantee:** cite a source or refuse; retrieved text is untrusted.
- **Reliability:** missing key / tool error degrades gracefully (503 / inline); circuit
  breaker prevents retry storms; checkpointer survives restarts.
- **Isolation:** per-tenant RLS + server-injected session scoping; LLM never selects
  the tenant.
- **Bounds:** tool-round, token, and per-request budgets; per-IP + global daily caps.
- **Quality:** CI eval gate blocks merges that regress faithfulness/relevance/precision.

---

## 7. Migration from v1 (what changes, what stays)

**Stays:** ingestion pipeline, pgvector + BM25 hybrid storage, the corrective-RAG gate,
the tiered LLM router + circuit breaker, the SSE trace contract, Render deploy.

**Changes:** native tools → MCP servers (structured returns, injected session);
single-path planner → fan-out supervisor; in-memory sessions → durable checkpointer +
RLS tenants; manual calibration → eval-in-CI; basic UI → reskin + Connected Tools panel.

**Removed framing:** "observable agentic RAG" as the headline; Langfuse references
(observability is LangSmith).

---

## 8. Known limitations / future

- Demo caps (10 MB / 120 pages); no OCR (no text-layer PDFs rejected).
- Fan-out increases token spend per complex query — bounded by the per-request budget.
- Local MCP-over-stdio adds a small per-call hop (negligible at demo scale; documented
  trade for a uniform bus).
- Cross-encoder rerank, agent memory (Mem0/LangGraph store), and visual PDF grounding
  remain post-v2 refinements.
