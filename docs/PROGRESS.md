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

## Phase 1 (eval gate green locally): DONE (2026-09-12, run 8)
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

## Phase 1 result: eval run 8 PASS (2026-09-12)
- `python -m evals.run_evals --pace 6` -> exit 0. 18 cases, hard_check_failures 0,
  checks_pass_rate 1.0, faithfulness 0.8824, answer_relevance 0.9647, context_precision
  0.3443 (threshold 0.25, calibrated; DECISIONS 24), agent_errors 0, judge_errors 0.
  Agent `openai/gpt-oss-120b`, judge `gemini-3.5-flash-lite` (Gemini key added by Sahaj).
- Eight runs were needed; each of the first seven exposed a real defect (see DECISIONS 17-24).

## Phase 4: CI green + regression proof (2026-09-12)
- Branch CI green: https://github.com/sahajm99/groundscope/actions/runs/34703733261
  (lint, pyright, seed, pytest incl. real-stack smoke, evals exit 0, eval-report artifact,
  docker build). First push failed only on pyright's unresolved dormant `langfuse` import.
- Three deliberate regressions, each caught by the first CI layer able to see it:
  1. threshold forced to 0.0 (nothing grounds): run 34704049355 red at the smoke tests.
  2. `MAX_SOURCES = 1` (recall collapse): run 34704211176 red at the unit tests.
  3. prompt told to add a fabricated sentence: run 34704354251 red at the smoke tests.
- Eval-gate-specific proof (regression 3 run locally, `--only` 8 cases, faithfulness judge
  on Gemini): deterministic checks 8/8, hard failures 0, **faithfulness 0.73 < 0.85 -> FAIL**.
  Report: `docs/eval-report-regressed-prompt.json`. Only the LLM judge can see this one.
- All three regressions reverted (commits `test(ci): ...` and their reverts).
- Final branch CI green (after the multi-part synthesis fix and temperature 0):
  https://github.com/sahajm99/groundscope/actions/runs/34704978970

## Phase 5: shipped (2026-09-12)
- PR #1 merged to main (merge commit 574496e); Render served v2 within ~2 min: `/health` lists
  all three MCP servers, `db_reachable: true`.
- Live check: retrieval and web search work on Render (Zephyr chunk at distance 0.149, four
  web results), but synthesis fails with `NotFoundError`: the Render dashboard still carries
  the retired Llama model names, which override render.yaml. Needs `LLM_MODEL`,
  `LLM_FALLBACK_MODEL` (and `GEMINI_API_KEY`) set in the Render dashboard.
- The first main run failed on the multi-part smoke test: under quota pressure the planner
  ran on the weakest tier and did not split a two-part question. Hotfix on main:
  `split_questions()` decomposes explicit multi-question input deterministically (no model
  call); the smoke question is phrased as two questions.

## Main runs after the merge (2026-09-12, evening)
- Run 34707065998 (5a32095): 17/18, one flaky web case (empty Tavily result -> refusal);
  fixed on main in 2def8d4 (empty web search retried once).
- Run 34707733909 (2def8d4): red on quota, not code: Groq 120b/20b both at their 200K
  tokens/day cap, Gemini at 500 requests/day; 11 agent errors, 3 judge errors.
- Fix (7 new tests, 97 pass, ruff/pyright clean): per-minute 429s wait on the same tier,
  daily 429s fail over at once, the judge falls to the Groq judge when Gemini is capped
  (recorded per case), and every tier failure is logged. Local proof with the real
  providers: judge ran on `qwen/qwen3.8-27b`; tier log shows the TPD messages.
- Push and re-run held until the daily budgets return (see BLOCKED.md).
