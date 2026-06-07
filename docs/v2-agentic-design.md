# Groundscope v2 — Production-Grade Agentic System (Free Tier)

**Date:** 2026-06-07
**Goal:** evolve v1 (a single-agent router loop) into an **Etech7-grade multi-agent system** — supervisor/planner-workers, evaluator, human-in-the-loop, durable checkpointing, model routing, memory, guardrails, eval, multi-tenancy — running on **$0** (free tiers + OSS).

Informed by web-current research (2026) on agentic patterns, framework pro/cons, and free-tier services.

---

## 1. Where v1 sits vs. the Etech7 platform

v1 is a **single-agent, single-loop** agentic-RAG with a corrective doc→web gate (already a real pattern: **CRAG-lite + RAG-as-tool**). The Etech7 platform is a **supervisor multi-agent StateGraph** with the full production hardening stack. v2 closes that gap — every Etech7 pattern has a free equivalent.

| Etech7 pattern | v1 today | v2 (free) |
|---|---|---|
| Supervisor / planner-worker | single loop | **LangGraph supervisor graph** (planner → workers → synthesizer) |
| Parallel fan-out (Send API) | sequential | LangGraph **Send API** parallel retrieval/web workers |
| Evaluator / critic | none | **CRAG grader** + **answer critic** (Self-RAG verify) node |
| Human-in-the-loop gate | none | LangGraph `interrupt()` gate before sensitive actions |
| Durable checkpointing | in-memory (lost on restart) | **LangGraph Postgres checkpointer** (Neon/Supabase) |
| Model routing / failover | single model | Groq → Gemini **fallback router** (LiteLLM or try/except) |
| Tool-use via MCP | direct fns | expose tools as an **MCP server** (FastMCP) + consume MCP |
| Hybrid RAG (BM25+vector) | vector only | **pgvector + Postgres full-text (tsvector) BM25** fusion |
| Reranking | none | **BGE-reranker-v2-m3** (local, free) |
| Circuit breaker | rate limit only | LLM **circuit breaker** (trip at ~5 fails / 60s cooldown) |
| Backpressure / queue | none | **Upstash QStash** or self-host Redis for ingest queue |
| Per-tenant concurrency caps | global only | per-tenant caps + **Postgres RLS** multi-tenancy |
| Structured-output tool agent | freeform synth | schema-constrained JSON outputs (Pydantic) |
| Observability / tracing | LangSmith (flat) | LangSmith **nested graph traces** (auto via LangGraph) |
| Eval | none | **Ragas/DeepEval** golden-set CI, judged by free LLM |
| Memory | none | **Mem0 / LangGraph store** (episodic + semantic) |

---

## 2. Trending agentic patterns (2026) and which v2 adopts

From the research, the patterns currently used in production agentic systems — v2 should demonstrably include the starred ones:

- **ReAct** (reason+act loop) — the substrate ✦
- **Supervisor multi-agent (orchestrator-workers)** — the 2026 production default ✦
- **Plan-and-execute** — planner emits a sub-task plan ✦
- **Parallel fan-out / map-reduce** ✦
- **Evaluator-optimizer / reflection** — separate critic loop ✦
- **Agentic RAG + Corrective/Self-RAG** — grade retrieval, verify claims ✦ (v1 already has the seed)
- **Router / handoff** — route by intent/difficulty ✦ (v1 has deterministic route)
- **Tool use / function calling** ✦ + **MCP tool servers** ✦
- **Agent memory** (short/long, episodic/semantic) ✦
- **Human-in-the-loop gates** ✦
- **Durable execution / checkpointing** ✦
- **Guardrails / circuit-breakers** ✦
- **Model routing / fallback** ✦
- Swarm / peer agents — *deliberately skipped* (supervisor is easier to trace/audit; swarm only wins on latency)

---

## 3. Target v2 architecture

```
                          ┌──────────── Supervisor (LangGraph StateGraph) ────────────┐
  question ─▶ guardrails ─▶│  planner ─▶ fan-out (Send API) ─┬─ doc_retrieval_worker   │
             (input)       │                                 ├─ web_search_worker      │
                           │                                 └─ metadata_worker        │
                           │             aggregator ◀────────────────────────────────  │
                           │   reranker ─▶ evaluator (CRAG grade + claim-verify)        │
                           │       │  not-grounded → re-plan / web                      │
                           │   synthesizer ─▶ HITL gate (if sensitive) ─▶ answer        │
                           └─────────────────────────┬─────────────────────────────────┘
       model router (Groq→Gemini failover) ·  Postgres checkpointer (durable) ·
       memory (Mem0) ·  LangSmith nested traces ·  per-tenant RLS + concurrency caps
```

Every node persists state to the Postgres checkpointer (thread_id = session/tenant), so any pause/crash/HITL resumes exactly where it left off — exactly the Etech7 MongoDB-checkpointer pattern, on free Postgres.

---

## 4. Framework choice — stay on LangGraph

| Framework | OSS/free | Best for | Verdict for Groundscope v2 |
|---|---|---|---|
| **LangGraph** | ✅ MIT (LangSmith/Platform paid) | stateful, durable, auditable production agents w/ HITL | **✅ CHOSEN** — mirrors Etech7, durable checkpointing + HITL native, LangSmith already wired |
| CrewAI | ✅ OSS core | fast role-based prototypes | fastest ramp, but rigid for dynamic flow; weaker determinism/audit |
| AutoGen / AG2 / MAF | ✅ OSS | emergent multi-agent debate, code-exec | confusing 3-way split (2026); non-deterministic; best in Azure |
| OpenAI Agents SDK | ✅ MIT (OpenAI platform paid) | fast handoffs on OpenAI models | OpenAI lock-in; no long-term memory OOTB; conflicts with free-Groq goal |
| LlamaIndex Workflows | ✅ MIT (LlamaCloud paid) | retrieval-first agents, best parsing | great RAG/parsing, weaker complex orchestration |
| Pydantic AI | ✅ MIT | typed single-agent Python | type-safe + lovely DX; multi-agent needs manual wiring |
| Google ADK | ✅ Apache-2.0 | multimodal, GCP ecosystem | strong eval/debugger, but GCP lock-in + 1-tool-per-agent limits |
| Smolagents | ✅ Apache-2.0 | minimal code-agents, research | tiny/transparent, but build everything else yourself + sandbox risk |
| Temporal (durable exec) | ⚠️ OSS self-host / Cloud paid | crash-proof orchestration backbone | not an agent framework; LangGraph checkpointing is enough for v2 |

**Why LangGraph wins here:** the goal is "build an agent like Etech7," and Etech7 *is* LangGraph. It uniquely gives durable checkpointing + native HITL interrupts + LangSmith tracing — the exact production triad — for free in the OSS core. We already have a working LangGraph engine in the repo (`agent/graph.py`); v2 expands it.

---

## 5. The free-tier stack (verified 2026)

| Component | Free option | Limit note |
|---|---|---|
| Orchestration | **LangGraph** (MIT) | free OSS; Platform cloud is the paid skip |
| LLM | **Groq** free (Llama/Qwen) + **Gemini** free failover | Groq ~30 RPM/1K RPD; Gemini as 2nd tier |
| Embeddings | **local fastembed / BGE-M3** | free forever, no rate limit (current v1) |
| Vector DB | **Supabase pgvector** (current) or Qdrant 1GB free | keep Supabase (does vector + metadata) |
| Relational/checkpointer | **Supabase/Neon Postgres** | LangGraph Postgres checkpointer lives here |
| Web search | **Tavily** ~1K/mo (current) | Brave free tier ended Feb 2026 |
| Reranker | **BGE-reranker-v2-m3** (local) | ~350ms CPU, free |
| Memory | **Mem0** or LangGraph store | self-host = no limits |
| Observability | **LangSmith** free 5K traces/mo (current) | or self-host Langfuse |
| Eval | **Ragas / DeepEval** | libs free; judge on Groq/Gemini |
| Guardrails | **LLM Guard** (PII/prompt-injection) | self-host, no per-call cost |
| Queue/backpressure | **Upstash QStash** 1K/day | or self-host Redis |
| Multi-tenancy | **Postgres RLS** + per-tenant vector namespace | architectural, $0 |
| Hosting | **Koyeb** free (no sleep) or Render | Koyeb stays warm; Fly no longer free |
| CI/CD | **GitHub Actions** free | public repos free |

Everything stays $0.

---

## 6. Phased roadmap (each phase = a resume-grade capability)

- **v2.0 — Multi-agent core.** Expand `graph.py` into a supervisor StateGraph: planner → parallel workers (doc/web/metadata via Send API) → aggregator → **evaluator (CRAG grader)** → synthesizer. Add the **Postgres checkpointer** (durable). LangSmith now shows one nested trace per question. *→ "supervisor multi-agent + durable checkpointing + evaluator."*
- **v2.1 — Retrieval quality.** Hybrid retrieval (**pgvector + Postgres tsvector BM25** fusion) + **BGE reranker**. *→ "hybrid RAG + reranking," matches the Etech7 BM25+vector bullet.*
- **v2.2 — Resilience + guardrails.** **Model router** (Groq→Gemini failover), **LLM circuit breaker**, **LLM Guard** (PII/prompt-injection), token/loop budget caps. *→ "circuit breaker, model routing/failover, guardrails."*
- **v2.3 — Memory.** **Mem0 / LangGraph store** for cross-session episodic + semantic memory (remember a user's docs and past questions). *→ "agent memory."*
- **v2.4 — Eval harness.** **Ragas/DeepEval** golden-set in CI (GitHub Actions), judged by a free LLM — faithfulness, answer-relevance, context-precision. *→ "evaluation of agent behavior," directly from the resume summary.*
- **v2.5 — Multi-tenancy.** Formalize sessions as tenants: **Postgres RLS** + per-tenant vector scoping + **per-tenant concurrency caps**. *→ "multi-tenant … per-tenant concurrency caps."*
- **v2.6 — Human-in-the-loop.** A LangGraph `interrupt()` gate before a "sensitive action" (e.g., writing a note / sending an email tool) with approve/edit/reject, resumed via the durable checkpointer. *→ "human-in-the-loop gate + interrupt/resume."*
- **v2.7 — MCP.** Expose Groundscope's tools as a **FastMCP server** (mirrors the Etech7 33-endpoint server) and/or consume external MCP tools. *→ "MCP tool servers."*

Do them in order; **v2.0 + v2.1 + v2.4** alone already read as a production agentic-RAG system.

---

## 7. What v2 proves (resume mapping)

After v2, Groundscope demonstrably shows, as a clickable artifact, the exact phrases on the Etech7 resume: *production multi-agent **LangGraph** orchestration, supervisor/planner-workers with parallel fan-out, an **evaluator** node, **human-in-the-loop** gate, durable **checkpointing** with interrupt/resume, **model routing/failover**, **MCP** tool-use, **hybrid RAG** + reranking, **circuit breaker** + per-tenant caps, **LangSmith** observability, and an **evaluation** harness* — all for $0, with the agent's reasoning visible live. It turns the resume bullets into something a hiring manager can run.

---

## 8. Open decisions
1. Build the whole roadmap, or start with **v2.0 (multi-agent core)** and iterate?
2. Keep **Supabase** (vector+metadata+checkpointer in one) vs. split vector to Qdrant — recommend keep Supabase for simplicity.
3. Verify the existing **LangGraph engine** first (it's the foundation for all of v2).
