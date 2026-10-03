# Groundscope: plan to v2.0 done (and the road after)

Updated 2026-10-03: v2.0 is complete and the statuses below were brought up to date from
`docs/PROGRESS.md`. Written 2026-09-12 as the working plan; `docs/PROGRESS.md` records what is proven,
`docs/DECISIONS.md` why, `docs/BLOCKED.md` what needs a human.

## Status as of 2026-10-03

| Area | Done | Proof |
|---|---|---|
| 2.0a MCP tool bus | yes | live `/health` lists 3 servers, 5 tools; 135 tests without model calls |
| 2.0b Send fan-out + lanes | yes | lanes screenshot; LangSmith run 01a095df with overlapping worker spans |
| 2.0c golden set + judge + CI gate | yes | main run 34777254872 green; 3 regressions caught red, at pytest stages rather than at the eval step (see BLOCKED.md watch items) |
| Hardening (locks, timeouts, caps, XSS) | yes | tests/test_hardening.py |
| Merged to main, Render serving v2 | yes | merge 574496e; deploy 2def8d4 Live |
| Render env (model names, Gemini key) | yes (you, 18:43 UTC) | live grounded and web answers with citations |
| Live grounded question | yes | resume question answered from the PDF |
| Live web-fallback question | yes | 2025 Nobel Prize in Physics answered from four web results, cited |
| Green badge on main | yes (2026-09-13) | run 34777254872: 113 tests, eval gate pass (faithfulness 1.00, relevance 1.00, precision 0.36, checks 17/18) |
| Rate-limit fix, model in trace, cited-only sources, demo corpus, blank-answer fix, v2.0.1 | yes, pushed 2026-09-13 | main 18190ea; Render serves 2.0.1 |
| /canary, /document-release | not run | the live checks recorded in PROGRESS.md stand in for the canary; docs were updated by hand |
| Portfolio `projects.ts` pushed | yes | sahajm99/portfolio f0eeb37; sahaj-mekala.vercel.app/projects/groundscope shows the v2 entry |
| Seed a public document into prod | yes | Bhagavad-Gita (Arnold, public domain): 18 chapters as pages, 64 chunks, GLOBAL |
| Cerebras capacity | not possible (PayGo, 402) | docs/BLOCKED.md |
| LangGraph 1.2 (unblocks v2.1) | yes (2026-10-03) | PR #2, run 37097978874: 116 tests, eval gate pass, Docker build; verified live |
| Corpus packs, phase 1 | yes (2026-10-03), engine only | texts stay local (DECISIONS 33, 35); synthetic pack under CI |
| Upload cleanup, keyword-leg label | yes (2026-10-03) | DECISIONS 36, 37 |

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
| 2.0a tool bus (3 MCP servers, one registry, injected tenant) | `/health` lists servers; smoke test grounds and web-falls-back through MCP | proven in CI and live |
| 2.0b fan-out (Send workers, per-branch gate, lanes) | lanes screenshot; LangSmith run with overlapping worker spans | proven in CI and live |
| 2.0c evals in CI (golden set, judge, ci.yml, badge) | green badge on main; a regressed grounding turns the eval job red | badge green; the regressions turned CI red before the eval step, so the eval-only proof is still the local run |
| Live | groundscope.onrender.com answers one grounded and one web question | proven 2026-09-12; re-verified 2026-10-03 on LangGraph 1.2 |
| Portfolio | `projects.ts` entry truthful and pushed | pushed; engineering overview page live |

## Task table

Closed. Statuses were brought up to date on 2026-10-03 from `docs/PROGRESS.md`.

Legend: owner **You** = needs your account or decision; **Me** = autonomous. "Proof" is what
closes the task.

### Phase 0: unblock infrastructure

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 0.1 | Resume the paused Supabase project (`mrvqaztivxnyeyowskug`) | You | done | `/health` on the live site shows `db_reachable: true` |
| 0.2 | Add repo secrets `LLM_API_KEY`, `TAVILY_API_KEY` (`gh secret set ... -R sahajm99/groundscope`) | You | done | `gh secret list` shows both |
| 0.3 | Decide on a free Gemini key (aistudio.google.com) as third LLM tier + eval judge; if yes, put `GEMINI_API_KEY` in `.env`, Render, and repo secrets | You | done | key present in the three places |
| 0.4 | Verify the resumed DB, list the seeded corpus, seed `data/corpus/*.txt` into production | Me | done | `GET /documents` on the live site lists the corpus |

### Phase 1: make the eval gate green locally

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 1.1 | Round-robin merge across branches; bound synthesis to 6 whole chunks (test already written, failing) | Me | done | `tests/test_fanout.py::...interleaves...` passes |
| 1.2 | Judge on a separate quota (Gemini if 0.3 = yes, else gpt-oss-20b) with strict model (no silent fallback) and the model actually used recorded per case | Me | done | eval report shows `judge_model_used` |
| 1.3 | Token-aware pacing so a run fits the free-tier per-minute cap | Me | done | a full run with 0 agent errors and 0 judge errors |
| 1.4 | Full golden-set run passes; calibrate the context-precision threshold from the measured baseline and record the numbers | Me | done | `python -m evals.run_evals` exits 0; numbers in PROGRESS.md and DECISIONS.md |

### Phase 2: production hardening (from the adversarial review)

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 2.1 | Lock around tool-bus rebuild; a timed-out keep-alive call marks the bus broken; old bus closed only after in-flight calls drain | Me | done | tests: concurrent rebuild loads once; timeout marks broken; in-flight call survives |
| 2.2 | LLM call timeouts (60 s, 1 retry) on both clients; larger thread pool at startup | Me | done | unit test on client options |
| 2.3 | Concurrency cap on `/ask` (503 when full) | Me | done | test with a semaphore of 1 |
| 2.4 | Web server keep-alive (or a spawn semaphore) so fan-out cannot spawn 3 children at once | Me | done | registry test |
| 2.5 | Calculator input caps (length, exponent, numeric-only constants); tool calls per ReAct round capped at 4 | Me | done | unit tests |
| 2.6 | Session id no longer emitted in trace events; `/health` reads the cached bus instead of rebuilding | Me | done | tests |
| 2.7 | Eval hardening: word-boundary `must_contain`, `expect_grounded`/`expect_web` as hard per-case failures | Me | done | tests |
| 2.8 | UI: `ask()` try/finally so the button never sticks disabled | Me | done | browser check |

### Phase 3: re-center on the answer experience

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 3.1 | Put the confirmed goal paragraph in README and `docs/v2-agentic-design.md`; drop "answer quality bounded on purpose" | Me | done | docs diff |
| 3.2 | Web citations render as reference links with the snippet used (AI-Mode style sources) | Me | done | screenshot |
| 3.3 | Pick a recognizable public document to seed the live demo (the fictional briefs stay for evals) | You (decision) then Me | done | live `GET /documents` |

### Phase 4: CI and the regression proof (needs 0.2)

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 4.1 | Push `v2.0`; CI green (lint, pyright, tests, evals, docker) | Me | done | Actions run URL |
| 4.2 | Deliberate grounding regression turns the eval job red; revert | Me | partly: red at pytest stages; the eval-only proof is a local run | two run URLs in PROGRESS.md |

### Phase 5: ship

| # | Task | Owner | Status | Proof |
|---|---|---|---|---|
| 5.1 | `/ship`: PR `v2.0 -> main` | Me | done | PR URL |
| 5.2 | `/land-and-deploy`: merge, Render deploy, `/health` and one grounded + one web question live | Me | done | live transcript |
| 5.3 | `/canary` on the live URL | Me | not run | canary report |
| 5.4 | `/document-release`: README, HLD status, CHANGELOG | Me | not run as a skill; README, HLD status and CHANGELOG were updated by hand | docs commit |
| 5.5 | Portfolio entry pushed (`projects.ts` only) | Me | done | Vercel deploy |
| 5.6 | Final report: what is live, badge URL, proofs, deferrals | Me | not written | message |

### Phase 6: after v2.0 (in roadmap order)

| # | Task | Notes |
|---|---|---|
| 6.0 | LangGraph 1.2 upgrade | done 2026-10-03 (PR #2); `interrupt()` and the Postgres checkpointer are now available |
| 6.1 | v2.1: Postgres checkpointer, HITL `interrupt()` before sensitive tools, input/output guardrails | mirrors Jarvis HITL gate |
| 6.2 | First real data connector via the MCP bus (Gmail or Drive or Notion) | the "connected data" half of the goal |
| 6.3 | v2.2: cost line per answer, Connected Tools panel | |
| 6.4 | Streaming answer with inline citations | AI-Mode feel |
| 6.5 | Corpus packs, phase 2: structural tools (`get_toc`, `get_verse`), commentator filter, per-pack prompt and web-fallback switch, a pack eval gate | phase 1 shipped 2026-10-03; its order against 6.1 to 6.3 is undecided |

## Order of execution

0.1 and 0.2 (you) -> 1.1 to 1.4 (me) -> 2.x (me) -> 3.1 to 3.2 (me) -> 4.x -> 5.x -> 6.x.
Phases 0 to 5 are complete; what remains is Phase 6.
3.3 needs your pick of a document; I will propose one if you prefer.
