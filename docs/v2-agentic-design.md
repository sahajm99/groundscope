# Groundscope v2 — Agent Harness Design

**Date:** 2026-06-08
**Status:** master design doc (the spec the HLD/LLD/roadmap derive from)
**Supersedes:** the v1 "observable agentic RAG" framing.

---

## 1. Thesis — what Groundscope actually is

Groundscope v2 is an **agent harness**: the machinery that wraps a raw LLM and turns
it into a reliable, observable, operable agent — orchestration, a uniform tool layer,
control & safety, observability, quality gates, and operations.

**The grounded, cited answer is the product; the harness is what makes it trustworthy.**
(Revised 2026-09-12: an earlier draft called answer quality "bounded on purpose"; that is
withdrawn. Groundscope should answer like a search engine's AI mode, with reference links,
and refuse when it cannot ground.)

Positioning (public, task-first):

> *Groundscope is a $0, open agent harness that answers questions with citations to real
> sources, your documents, connected data, and the web, or refuses. The planner splits a
> question into parallel branches, every branch grounds through one pluggable MCP tool bus
> with tenant scoping the model cannot touch, and a golden-set eval gate in CI turns a change
> red when answers stop being grounded. It is the public, free-tier twin of the Jarvis agent
> architecture. The grounded, cited answer is the product; the harness is what makes it
> trustworthy.*

---

## 2. The brain and the hands (as built, grounded in code)

### 🧠 Brain — two models, two jobs
| Role | Model | Source |
|---|---|---|
| Reasoning | Groq `llama-3.3-70b-versatile` → `llama-3.1-8b-instant` → optional 3rd provider | `app/agent/llm.py` |
| Retrieval (embeddings) | local `BAAI/bge-small-en-v1.5`, 384-dim (fastembed) | `app/config.py` |

The reasoning brain is already wrapped in a **tiered router + circuit breaker**
(5 consecutive primary fails → 60 s cooldown → skip to fallback). OpenAI-compatible
client, so a provider swap is an env change. This is a real harness capability that
exists in v1.

### ✋ Hands — today they are split (the core problem v2 fixes)
v1 has **two tool systems that don't compose**:

- **Native in-process tools** (hardcoded into the graph): `hybrid_search`,
  `metadata_query`, `web_search`.
- **MCP tools** (plug-and-play via `mcp.json` → `mcp_servers/util_server.py`):
  `calculator`, `current_datetime`.

The planner picks **one path per query** (`graph.py` `_plan_branch`): knowledge (RAG)
**or** tools (MCP ReAct loop). It cannot search a document *and* compute in the same
answer. That split is the single biggest weakness the harness framing exposes.

### Decision (locked) — unify on MCP
**MCP becomes the single tool bus.** Native retrieval/web/metadata tools are migrated
into Groundscope's own MCP servers, so the agent sees one tool registry, the planner
can compose across all tools, and the UI can show one live "Connected Tools" panel.

```
agent ──binds──► ONE tool registry (MCP protocol)
   ├─ groundscope-retrieval  → hybrid_search, metadata_query   ← also our PUBLIC MCP server
   ├─ groundscope-web        → web_search (Tavily) [+ gemini_grounding]
   └─ groundscope-utils      → calculator, datetime [+ fetch_url]
```

Two things this design must respect (carried into the LLD):
1. **Structured returns, not strings.** The corrective-RAG gate needs the vector
   *distance score* back to decide doc-vs-web fallback, so `groundscope-retrieval`
   returns structured JSON (`score` + `sources`), not a plain string.
2. **Tenant scoping is never LLM-controlled.** `session_id` is injected by the
   orchestration layer when it invokes retrieval/metadata tools — the LLM never
   chooses which tenant's data to read. Utility tools (calc, datetime, fetch) are
   "open" tools the LLM may call freely; retrieval/metadata are "context" tools the
   graph invokes with a server-injected session token. This keeps isolation
   deterministic *and* keeps a uniform bus.

The double win: `groundscope-retrieval` is **both** the agent's internal retrieval
tool **and** the public MCP server other agents can call. One server, two roles.

---

## 3. The harness spine — six layers

The whole system reorganizes around six layers. Every layer answers the same four
questions: *what it is · what v1 has · what v2 adds · how it's visibly provable.*

| Layer | v1 (shipped) | v2 adds | Provable by |
|---|---|---|---|
| **1 Orchestration** | planner → knowledge/tool path → synth (LangGraph) | **query fan-out** (decompose → parallel sub-queries → merge); plan-and-execute | trace panel showing parallel branches |
| **2 Tool layer** | split native + MCP | **unify on MCP** bus; expose Groundscope *as* an MCP server; `fetch_url`, `gemini_grounding` | "Connected Tools" UI panel + tool-call spans |
| **3 Control / safety** | circuit breaker, grounded-or-refuse gate | durable Postgres checkpointer + **HITL interrupt/resume**; input/output guardrails; token & loop budgets | a paused run you resume; a blocked unsafe query |
| **4 Observability** | LangSmith nested traces, SSE step stream | **cost & token per query**; latency per node | live "watch it think" panel + cost line |
| **5 Quality gate** | manual relevance calibration | **evals-in-CI** (golden set; faithfulness / answer-relevance / context-precision) | green CI badge + score trend |
| **6 Operations** | Docker → Render, Vercel-linked portfolio | **GitHub Actions** (lint → type → test → eval → build); preview deploys; multi-tenancy (Postgres RLS) | the Actions tab; a PR preview URL |

---

## 4. Layer detail

### Layer 1 — Orchestration
A LangGraph `StateGraph` supervisor. v1 already routes planner → (knowledge | tool)
→ synth. v2 adds **query fan-out**: the planner decomposes a complex question into
independent sub-queries, runs them as **parallel workers** (LangGraph `Send` API),
and an aggregator merges results before the gate. The corrective-RAG gate
(doc → web → ground-or-refuse) stays as deterministic orchestration *on top of* the
tool bus — it is a feature, not something to dissolve into a free ReAct loop. Fan-out
is the headline orchestration flex and is visible as parallel branches in the trace.

### Layer 2 — Tool layer
Unify on MCP (§2). All capabilities become MCP tools across three Groundscope-owned
servers plus any external MCP server dropped into `mcp.json`. The agent binds one
registry; the deterministic gate and the ReAct path both draw from it. New tools:
`fetch_url` (utility), `gemini_grounding` (Gemini "Grounding with Google Search" as a
web tool). Groundscope's own retrieval server is published so other agents can call it.

### Layer 3 — Control / safety
- **Durable checkpointer** — LangGraph Postgres checkpointer (Supabase), `thread_id =
  session/tenant`. Any pause/crash/HITL resumes exactly where it left off.
- **HITL gate** — a LangGraph `interrupt()` before a sensitive tool action
  (approve / edit / reject), resumed via the checkpointer.
- **Guardrails** — input (prompt-injection / PII) and output checks; retrieved text is
  treated as untrusted (already the v1 stance), now formalized.
- **Budgets** — token and tool-round caps (v1 has `max_tool_rounds`), extended to a
  per-request token budget surfaced in the trace.

### Layer 4 — Observability
LangSmith already gives one nested trace per question (auto via LangGraph) and the SSE
panel streams every step. v2 adds **cost & token accounting per query** (estimated from
model + token counts) and **per-node latency**, both surfaced in the trace and a small
cost line in the UI.

### Layer 5 — Quality gate
A **golden set** of (question → expected-grounding) cases. Metrics: faithfulness,
answer-relevance, context-precision (Ragas/DeepEval), judged by a free LLM. Runs in CI
on every PR; a regression below threshold fails the build. This is the DevOps proof
that the harness has a quality contract, not vibes.

### Layer 6 — Operations
A **GitHub Actions** pipeline: `lint → typecheck → test → eval → docker build`, with
preview deploys per PR. Sessions formalize into tenants with **Postgres RLS** and
per-tenant scoping/caps. Deploy target stays Render (Docker, free tier).

---

## 5. Non-goals (read this before judging the design)

- **Not reimplementing an agent framework.** Groundscope is built *on* LangGraph and
  MCP. The contribution is the operational layer around the agent — fan-out
  orchestration, evals-in-CI, observability, cost control, multi-tenancy — **not** a
  from-scratch graph engine.
- **Not a better answer engine.** Answer quality is intentionally bounded; the demo
  workload exists to exercise the harness.
- **Not a Google-AI-Mode clone.** Cloning a consumer search product is the wrong axis;
  the differentiator is production system rigor.
- **No swarm / peer-debate agents.** Supervisor is easier to trace and audit; swarm
  only wins on latency, which is not a goal here.
- **No paid tiers.** Every layer has a $0 path or it doesn't ship.

---

## 6. Free-tier stack (verified, current)

| Concern | Choice | Note |
|---|---|---|
| Orchestration | **LangGraph** (MIT) | OSS core free; Platform cloud is the paid skip |
| Reasoning LLM | **Groq** (Llama 3.3 70B / 3.1 8B) + optional **Gemini** failover | Groq free tier, no card |
| Embeddings | **local fastembed `bge-small`** (384-dim) | free forever, no rate limit |
| Vector + relational + checkpointer | **Supabase Postgres + pgvector** | one DB; session pooler (IPv4) |
| Web search | **Tavily** (~1K/mo) + **Gemini grounding** tool | both free tier |
| Observability | **LangSmith** (free 5K traces/mo) | nested graph traces auto via LangGraph |
| Eval | **Ragas / DeepEval** | libs free; judge on Groq/Gemini |
| Guardrails | **LLM Guard** (self-host) | no per-call cost |
| Multi-tenancy | **Postgres RLS** + per-tenant scoping | architectural, $0 |
| Hosting | **Render** (Docker free) + **Vercel** (portfolio) | current live deploy |
| CI/CD | **GitHub Actions** | public repo free |

Everything stays $0.

---

## 7. Framework choice — stay on LangGraph (condensed)

LangGraph uniquely gives the production triad for free in its OSS core: **durable
checkpointing + native HITL interrupts + LangSmith tracing**. A working LangGraph
engine already exists in the repo (`agent/graph.py`); v2 expands it rather than
rebuilding. Alternatives considered and why they lose here: CrewAI (rigid for dynamic
flow, weaker audit), OpenAI Agents SDK (provider lock-in, conflicts with free-Groq),
LlamaIndex Workflows (great RAG, weaker complex orchestration), AutoGen/AG2
(non-deterministic, hard to trace), Smolagents (build-everything-yourself + sandbox
risk). Temporal is a durable-execution backbone, not an agent framework — LangGraph
checkpointing is enough for v2.

---

## 8. Open decisions
1. Build the full spine, or ship **Layer 1 (fan-out) + Layer 5 (evals-in-CI)** first as
   the v2.0 increment? (Recommend the latter — orchestration flex + DevOps flex in one.)
2. Keep Supabase as the single DB (vector + metadata + checkpointer) vs. split — keep
   Supabase for simplicity.
3. UI reskin (warm-dark + gold, "Connected Tools" panel) — designed separately.
