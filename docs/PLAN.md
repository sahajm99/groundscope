# Groundscope: plan to v2.0 done (and the road after)

Updated 2026-09-12. This is the working plan; `docs/PROGRESS.md` records what is proven,
`docs/DECISIONS.md` why, `docs/BLOCKED.md` what needs a human.

## Goal (confirmed 2026-09-12)

Groundscope is a $0, open agent harness that answers questions with citations to real
sources, your documents, connected data, and the web, or refuses. The planner splits a
question into parallel branches, every branch grounds through one pluggable MCP tool bus
with tenant scoping the model cannot touch, and a golden-set eval gate in CI turns a change
red when answers stop being grounded. It is the public, free-tier twin of the Jarvis agent
architecture.

The grounded, cited answer is the product. The harness is what makes it trustworthy.

## Definition of done for v2.0

| Milestone | Provable by | Status |
|---|---|---|
| 2.0a tool bus (3 MCP servers, one registry, injected tenant) | `/health` lists servers; smoke test grounds and web-falls-back through MCP | proven locally |
| 2.0b fan-out (Send workers, per-branch gate, lanes) | lanes screenshot; LangSmith run with overlapping worker spans | proven locally |
| 2.0c evals in CI (golden set, judge, ci.yml, badge) | green badge on main; a regressed grounding turns the eval job red | built; eval run at 15/18 locally; CI not yet run |
| Live | groundscope.onrender.com answers one grounded and one web question | blocked on Supabase resume |
| Portfolio | `projects.ts` entry truthful and pushed | drafted, typechecks, not pushed |

## Task table

Legend: owner **You** = needs your account or decision; **Me** = autonomous. "Proof" is what
closes the task.

### Phase 0: unblock infrastructure

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 0.1 | Resume the paused Supabase project (`mrvqaztivxnyeyowskug`) | You | pending | `/health` on the live site shows `db_reachable: true` |
| 0.2 | Add repo secrets `LLM_API_KEY`, `TAVILY_API_KEY` (`gh secret set ... -R sahajm99/groundscope`) | You | pending | `gh secret list` shows both |
| 0.3 | Decide on a free Gemini key (aistudio.google.com) as third LLM tier + eval judge; if yes, put `GEMINI_API_KEY` in `.env`, Render, and repo secrets | You | pending (decision) | key present in the three places |
| 0.4 | Verify the resumed DB, list the seeded corpus, seed `data/corpus/*.txt` into production | Me | after 0.1 | `GET /documents` on the live site lists the corpus |

### Phase 1: make the eval gate green locally

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 1.1 | Round-robin merge across branches; bound synthesis to 6 whole chunks (test already written, failing) | Me | in progress | `tests/test_fanout.py::...interleaves...` passes |
| 1.2 | Judge on a separate quota (Gemini if 0.3 = yes, else gpt-oss-20b) with strict model (no silent fallback) and the model actually used recorded per case | Me | pending | eval report shows `judge_model_used` |
| 1.3 | Token-aware pacing so a run fits the free-tier per-minute cap | Me | pending | a full run with 0 agent errors and 0 judge errors |
| 1.4 | Full golden-set run passes; calibrate the context-precision threshold from the measured baseline and record the numbers | Me | pending | `python -m evals.run_evals` exits 0; numbers in PROGRESS.md and DECISIONS.md |

### Phase 2: production hardening (from the adversarial review)

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 2.1 | Lock around tool-bus rebuild; a timed-out keep-alive call marks the bus broken; old bus closed only after in-flight calls drain | Me | pending | tests: concurrent rebuild loads once; timeout marks broken; in-flight call survives |
| 2.2 | LLM call timeouts (60 s, 1 retry) on both clients; larger thread pool at startup | Me | pending | unit test on client options |
| 2.3 | Concurrency cap on `/ask` (503 when full) | Me | pending | test with a semaphore of 1 |
| 2.4 | Web server keep-alive (or a spawn semaphore) so fan-out cannot spawn 3 children at once | Me | pending | registry test |
| 2.5 | Calculator input caps (length, exponent, numeric-only constants); tool calls per ReAct round capped at 4 | Me | pending | unit tests |
| 2.6 | Session id no longer emitted in trace events; `/health` reads the cached bus instead of rebuilding | Me | pending | tests |
| 2.7 | Eval hardening: word-boundary `must_contain`, `expect_grounded`/`expect_web` as hard per-case failures | Me | pending | tests |
| 2.8 | UI: `ask()` try/finally so the button never sticks disabled | Me | pending | browser check |

### Phase 3: re-center on the answer experience

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 3.1 | Put the confirmed goal paragraph in README and `docs/v2-agentic-design.md`; drop "answer quality bounded on purpose" | Me | pending | docs diff |
| 3.2 | Web citations render as reference links with the snippet used (AI-Mode style sources) | Me | pending | screenshot |
| 3.3 | Pick a recognizable public document to seed the live demo (the fictional briefs stay for evals) | You (decision) then Me | pending | live `GET /documents` |

### Phase 4: CI and the regression proof (needs 0.2)

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 4.1 | Push `v2.0`; CI green (lint, pyright, tests, evals, docker) | Me | pending | Actions run URL |
| 4.2 | Deliberate grounding regression turns the eval job red; revert | Me | pending | two run URLs in PROGRESS.md |

### Phase 5: ship

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 5.1 | `/ship`: PR `v2.0 -> main` | Me | pending | PR URL |
| 5.2 | `/land-and-deploy`: merge, Render deploy, `/health` and one grounded + one web question live | Me | pending (needs 0.1) | live transcript |
| 5.3 | `/canary` on the live URL | Me | pending | canary report |
| 5.4 | `/document-release`: README, HLD status, CHANGELOG | Me | pending | docs commit |
| 5.5 | Portfolio entry pushed (`projects.ts` only) | Me | pending | Vercel deploy |
| 5.6 | Final report: what is live, badge URL, proofs, deferrals | Me | pending | message |

### Phase 6: after v2.0 (in roadmap order)

| # | Task | Notes |
|---|---|---|
| 6.1 | v2.1: Postgres checkpointer, HITL `interrupt()` before sensitive tools, input/output guardrails | mirrors Jarvis HITL gate |
| 6.2 | First real data connector via the MCP bus (Gmail or Drive or Notion) | the "connected data" half of the goal |
| 6.3 | v2.2: cost line per answer, Connected Tools panel | |
| 6.4 | Streaming answer with inline citations | AI-Mode feel |

## Order of execution

0.1 and 0.2 (you) -> 1.1 to 1.4 (me) -> 2.x (me) -> 3.1 to 3.2 (me) -> 4.x -> 5.x -> 6.x.
3.3 needs your pick of a document; I will propose one if you prefer.
