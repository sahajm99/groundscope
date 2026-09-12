# v2.0 progress log

Resumed sessions: read this, then `docs/DECISIONS.md` and `docs/BLOCKED.md`, then the plan in
`docs/superpowers/plans/2026-09-12-v2.0-tool-bus-fanout-evals.md`. Branch: `v2.0`.

## 2.0a Tool bus: PROVEN (2026-09-12)
- What: three MCP servers (`mcp_servers/retrieval_server.py`, `web_server.py`, `util_server.py`)
  registered in `mcp.json`, loaded by one registry (`app/agent/mcp_registry.py`) into one
  `ToolBus` (`app/agent/toolbus.py`). Graph nodes call `hybrid_search` / `metadata_query` /
  `web_search` through the bus with a server-injected `session_id`; the ReAct worker only sees
  open tools plus session-scoped wrappers whose schema has no `session_id`.
- How proven:
  - `pytest tests/test_mcp_servers.py tests/test_toolbus.py` -> 15 passed (local pgvector
    `gs-pg` on :5433, seeded with `data/sample.txt`). Includes real subprocess spawning with env
    inheritance and the keep-alive retrieval session being reused across calls.
  - `pytest tests/test_smoke.py` (real Groq + Tavily + DB + MCP subprocesses) -> 3 passed:
    grounded answer cites `sample.txt` via `hybrid_search` (tool_result with score), web
    question cites web via `web_search`.
  - `ruff check .` clean; `pyright` 0 errors.
- Surprises: Groq retired `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` (404); defaults
  moved to `openai/gpt-oss-120b` / `openai/gpt-oss-20b` (decision 16). mcp's stdio client
  strips the environment, so the registry passes `os.environ` explicitly (would have broken
  only in Docker).

## 2.0b Fan-out: PROVEN (2026-09-12)
- What: planner returns `{route, subqueries}` (cap 3); `fanout` conditional edge emits
  `Send("retrieval_worker", ...)` per sub-query; async workers run a per-branch corrective gate
  (docs -> web) and write only the `branches` reducer key; `aggregate` dedupes and takes the
  best doc distance; trace events inside workers carry `branch`.
- Proven so far: `pytest tests/test_fanout.py` -> 8 passed, including a concurrency assertion
  (three workers in flight simultaneously) and one-branch-failure / timeout degradation.
  `tests/test_smoke.py::test_multi_part_question_shows_parallel_branches` passed on the real
  stack (>= 2 branches).
- LangSmith proof (2026-09-12 13:47 UTC): one nested `LangGraph` root run with two
  `retrieval_worker` child spans that start at the same instant and overlap:
  https://smith.langchain.com/o/93402020-f5cb-4b7c-945b-643d71328b5a/projects/p/69844978-28c0-4627-883d-38299a6cc82e/r/01a095df-6d42-7f12-a587-59c2ab131f59
  (worker 0: 13:47:25.435 -> 13:47:27.810; worker 1: 13:47:25.435 -> 13:47:27.048).
- UI: `static/index.html` renders branch lanes (dashed lane per branch with a
  `branch N: <sub-query>` header and per-branch accent); verified in headless Chromium with a
  mocked SSE stream (27 assertions incl. lane order, interleaving, retitling, XSS payloads
  rendered as text). The renderer no longer uses innerHTML for any data (stored-XSS fix).
- Screenshot: `docs/screenshots/fanout-lanes.png` (taken during /qa).

## 2.0c Evals in CI: in progress
- Built: `evals/golden.jsonl` (18 cases over `data/corpus/*.txt`: 13 grounded, 2 multi, 2 web,
  1 metadata), `evals/judge.py` (one combined judge call per case on `openai/gpt-oss-20b`:
  faithfulness, answer relevance, context precision; deterministic must_contain / citation-kind
  gates), `evals/run_evals.py` (thresholds, `eval-report.json`, exit 1 on regression, exit 2
  without a key), `.github/workflows/ci.yml` (pgvector service, lint, pyright, seed, pytest,
  evals, artifact; docker build job), README badge.
- Unit-proven: `pytest tests/test_evals.py tests/test_llm.py` -> 17 passed (fake judge/agent:
  a regressed grounding fails, a judge saying unfaithful fails, an agent crash fails).
- Docker: `docker build -t groundscope .` succeeds locally; the container boots, reaches the
  test DB via host.docker.internal, and `/health` lists all three MCP servers (env inheritance
  proven in the Docker environment Render uses).
- QA (quick tier, local run): 8 flows pass, 1 low issue found and fixed (refusal answers no
  longer list unused sources), report in `.gstack/qa-reports/`.
- Next: real local eval run numbers; push branch; CI run; deliberate-regression proof.


## Phase 0 (infra): DONE (2026-09-12 15:20 UTC)
- Supabase project resumed by Sahaj; `/health` on the live site reports `db_reachable: true`.
- GitHub secrets `LLM_API_KEY`, `TAVILY_API_KEY` set. No Gemini key (judge stays on Groq).

## Phase 1 (eval gate green locally): in progress
- Done: round-robin branch merge + 6 whole-chunk synthesis budget (was cutting chunks at
  1,200 chars); strict judge model recorded per case; rate-limit retry; Unicode-tolerant
  checks; OR-semantics keyword leg; metadata cases skip the judge.
- Eval runs so far: run 2 27% checks (planner mis-routing, dedupe bug), run 3 83% (Unicode
  checks), run 4 83% (chunk truncation), run 5 aborted (judge model hit Groq's ~200K
  tokens/day cap). Judge moved to `qwen/qwen3.8-27b` (own budget). Run 6 in progress.

## Phase 2 (hardening): DONE, 13 tests in `tests/test_hardening.py`
- Bus rebuild under a lock, old bus drained before close, keep-alive timeout marks the bus
  broken, `/ask` in-flight cap (503), LLM client timeouts, 16-thread pool, calculator caps,
  4 tool calls per round, `/health` reads the cached bus, no session id in trace events,
  eval hard checks (citation-kind failures fail the run), web server keep-alive, UI
  try/finally. 79 tests pass.

## Phase 3 (answer experience): 3.1 and 3.2 done, 3.3 pending a document choice
- Goal paragraph in README and design doc; sources block with title link, domain, and the
  snippet used (`docs/screenshots/sources-block.png` pending a quota-free browser run).
