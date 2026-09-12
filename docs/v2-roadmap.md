# Groundscope v2 — Roadmap

**Date:** 2026-06-08
**Frame:** each milestone **lights up one harness layer** and ships something *provable*
(a trace, a UI panel, a CI badge, a resumable run). Order is dependency-driven.
**Derives from:** `docs/v2-agentic-design.md`, `HLD-groundscope-v2.md`, `LLD-groundscope-v2.md`.

A milestone is **done** only when its "Provable by" artifact exists and is visible to a
visitor or in CI — not when the code merely compiles.

---

## v2.0 — Orchestration + Quality gate (the headline increment)
**Layers:** 1 (fan-out) + 5 (evals-in-CI). Ship these together: the orchestration flex
and the DevOps flex in one move.

### 2.0a — Tool bus (Layer 2, prerequisite)
- Split native `hybrid_search` / `metadata_query` / `web_search` into the three
  Groundscope MCP servers; structured JSON returns; open vs. context wrapping; inject
  `session_id` server-side.
- **DoD / provable by:** the agent binds one MCP registry; a query still grounds in docs
  and still falls back to web — through MCP tools — with the gate intact. Smoke test green.

### 2.0b — Query fan-out (Layer 1)
- Planner decomposes complex questions into sub-queries; `Send` runs parallel
  `retrieval_worker`s; `aggregate` merges + dedupes; gate unchanged.
- **DoD / provable by:** the live trace shows **parallel branches** for a multi-part
  question (e.g. "compare X in doc A and Y on the web"); LangSmith shows one nested trace
  with concurrent spans.

### 2.0c — Evals-in-CI (Layers 5–6)
- `evals/golden.jsonl` (~15–25 cases), `run_evals.py` (faithfulness / answer-relevance /
  context-precision via Ragas/DeepEval, free-LLM judge), `.github/workflows/ci.yml`
  (lint → typecheck → test → eval → docker build).
- **DoD / provable by:** a **green CI badge** in the README; a PR that regresses grounding
  fails the eval job; `eval-report.json` artifact attached to the run.

**v2.0 verdict:** with 2.0a–c, Groundscope reads as a production agentic system — uniform
tool bus, parallel orchestration, and a quality contract enforced in CI.

---

## v2.1 — Control / safety (Layer 3)
- LangGraph **Postgres checkpointer** (`thread_id = session`); **HITL `interrupt()`**
  before a sensitive open-tool action (approve/edit/reject); **guardrails**
  (`screen_input`/`screen_output`) + per-request token/subquery budgets.
- **DoD / provable by:** a run that **pauses on a gate and resumes** after approval
  (survives a process restart); a prompt-injection input is **visibly blocked** in the
  trace.

---

## v2.2 — Observability depth + Tools UI (Layers 2, 4)
- Cost/token accounting in `llm.py` (rate table); trace events carry `tokens`/`cost_usd`;
  `GET /tools` endpoint + **Connected Tools panel**; UI reskin to warm-dark + gold.
- **DoD / provable by:** every answer shows a **cost line**; the UI lists the connected
  MCP servers and their tools, and highlights which tool fired in the trace.

---

## v2.3 — Operations / multi-tenancy (Layer 6)
- Formalize sessions as tenants: **Postgres RLS** (`app.tenant` per request) + per-tenant
  scoping/caps; PR **preview deploys**.
- **DoD / provable by:** two browser sessions cannot read each other's uploads (RLS
  enforced at the DB, not the app); a PR shows a preview URL/check.

---

## v2.4 — Meta + extensions (optional, high-signal)
- Publish `groundscope-retrieval` as a **public MCP server** (documented connection so
  another agent / Claude can call it); add `gemini_grounding` and `fetch_url`; optional
  cross-encoder rerank and agent memory (Mem0 / LangGraph store).
- **DoD / provable by:** an external MCP client (e.g. Claude Desktop) calls Groundscope's
  retrieval tool and gets grounded results — "my agent's retrieval is itself a tool."

---

## Dependency order (at a glance)
```
2.0a tool bus ─▶ 2.0b fan-out ─▶ 2.0c evals/CI ─▶ 2.1 control ─▶ 2.2 obs/UI ─▶ 2.3 tenancy ─▶ 2.4 meta
                 (Layer 1)        (Layers 5–6)     (Layer 3)      (Layers 2,4)   (Layer 6)      (Layer 2)
```

## What each milestone proves (one line each)
- **2.0a** — uniform, plug-and-play tool layer (MCP).
- **2.0b** — parallel multi-source orchestration, visible.
- **2.0c** — a quality contract enforced by CI/CD.
- **2.1** — durable, human-gated, guard-railed execution.
- **2.2** — cost-aware observability + a legible tool surface.
- **2.3** — real multi-tenant isolation at the data layer.
- **2.4** — the agent's capabilities are themselves reusable infrastructure.

## Non-goals (carried from the design doc)
Not a new agent framework, not a better answer engine, not a Google-AI-Mode clone, no
swarm agents, no paid tiers. See `v2-agentic-design.md` §5.

---

## Recommended start
Build **v2.0 (a → b → c)** as one increment. It's the smallest slice that turns
Groundscope from "a RAG demo" into "an agent harness with parallel orchestration and a
CI-enforced quality gate" — the two flexes that carry the whole story.
