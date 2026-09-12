# Changelog

All notable changes to Groundscope. Format follows Keep a Changelog; versions follow the
roadmap milestones in `docs/v2-roadmap.md`.

## [2.0.1] - 2026-09-12

### Added
- **Demo corpus.** The Bhagavad-Gita in Sir Edwin Arnold's 1885 translation (public domain,
  Project Gutenberg #2388) is seeded into the live demo as one document whose pages are its 18
  chapters, so a citation reads `bhagavad-gita.txt p.2` for chapter 2 (`scripts/fetch_gita.py`,
  `data/demo/`). A plain-text upload may use a form feed as a page break.
- **Which model answered.** The trace's synthesis line and the answer payload (`model`) name the
  model that wrote the answer; the eval report records `agent_model_used` per case.

### Changed
- **Sources list what the answer used.** Citations are filtered to the labels (or URLs) the
  answer references, with page numbers matched whole (`p.1` is not `p.10`); if the answer
  names none, every retrieved source is shown.
- **Rate limits.** A per-minute 429 (Groq's 8K tokens/minute) is waited out on the same tier
  instead of failing over; a daily cap fails over at once. The eval judge falls back to the
  Groq judge model when Gemini is rate-limited, recorded per case. Every tier failure is logged.

## [2.0.0] - 2026-09-12

The "agent harness" release: one MCP tool bus, parallel query fan-out, and an eval gate in CI.

### Added
- **MCP tool bus (2.0a).** Three Groundscope-owned MCP servers, all stdio, all in `mcp.json`:
  `groundscope-retrieval` (`hybrid_search`, `metadata_query`, structured JSON returns),
  `groundscope-web` (`web_search`), `groundscope-utils` (`calculator`, `current_datetime`).
  One registry (`app/agent/mcp_registry.py`) loads them into one `ToolBus`
  (`app/agent/toolbus.py`). Context tools receive the session id from the orchestrator; the
  ReAct worker only sees session-scoped wrappers with no tenant field in their schema.
- **Keep-alive sessions.** A server marked `"keepalive": true` in `mcp.json` runs as one
  long-lived child for the life of the app (used for retrieval); the others spawn per call.
- **Query fan-out (2.0b).** The planner returns `{route, subqueries}` (max 3); a LangGraph
  `Send` edge runs one async `retrieval_worker` per sub-query in parallel; each branch runs its
  own corrective gate (documents, then web); `aggregate` dedupes sources. Trace events carry a
  `branch` id and the UI renders branches as lanes. LangSmith shows the workers as concurrent
  spans under one run.
- **Evals in CI (2.0c).** `evals/golden.jsonl` (18 cases over `data/corpus/`), `evals/judge.py`
  (faithfulness, answer relevance, context precision via one judge call per case on a
  different free model than the agent; deterministic must-contain and citation-kind gates),
  `evals/run_evals.py` (thresholds, `eval-report.json`, non-zero exit on regression).
  `.github/workflows/ci.yml` runs ruff, pyright, pytest (unit, integration against a pgvector
  service container, real-stack smoke), the eval gate, and a docker build. README badge.
- `GET /health` now lists the loaded MCP servers and their tools (`mcp`).
- `tests/` (48 unit/integration tests plus 3 real-stack smoke tests), `requirements-dev.txt`,
  `pyproject.toml` (ruff, pyright, pytest config).
- `data/corpus/`: three fictional company briefs used by the golden set; `scripts/seed.py`
  accepts many files and is idempotent.

### Changed
- Default models: `openai/gpt-oss-120b` (agent) with `openai/gpt-oss-20b` fallback and judge.
  Groq retired `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.
- `TraceEvent` gained `branch`; settings gained `max_subqueries`, `tool_timeout_s`,
  `eval_judge_model`.
- Tool failures degrade instead of failing the request: a branch whose retrieval fails is
  treated as "no document matches" and falls back to the web; the trace shows only the
  exception type (the full error goes to the server log).

### Fixed
- MCP child processes now inherit the app environment (`DATABASE_URL`, `TAVILY_API_KEY`);
  mcp's stdio client otherwise forwards only PATH/HOME-style variables.
- The trace panel and citations no longer use `innerHTML` for server data (stored-XSS fix);
  links are only rendered as anchors for http(s) URLs.

### Known limitations
- The Supabase project behind the live deploy is paused; see `docs/BLOCKED.md`.
- The CI eval job needs the `LLM_API_KEY` and `TAVILY_API_KEY` repository secrets.

## [1.x] - 2026-06

Observable agentic RAG demo: LangGraph planner (knowledge path vs. tool-worker), hybrid
retrieval (pgvector + Postgres BM25, RRF), relevance gate with Tavily fallback, grounded-or-
refuse synthesis, tiered model router + circuit breaker, live SSE trace, LangSmith tracing,
one MCP utility server, Render deploy.
