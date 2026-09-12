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

## 2.0b Fan-out: code done, proof in progress
- What: planner returns `{route, subqueries}` (cap 3); `fanout` conditional edge emits
  `Send("retrieval_worker", ...)` per sub-query; async workers run a per-branch corrective gate
  (docs -> web) and write only the `branches` reducer key; `aggregate` dedupes and takes the
  best doc distance; trace events inside workers carry `branch`.
- Proven so far: `pytest tests/test_fanout.py` -> 8 passed, including a concurrency assertion
  (three workers in flight simultaneously) and one-branch-failure / timeout degradation.
  `tests/test_smoke.py::test_multi_part_question_shows_parallel_branches` passed on the real
  stack (>= 2 branches).
- Next: UI lanes in `static/index.html` (in progress), LangSmith concurrent-span proof, screenshot.

## 2.0c Evals in CI: not started
