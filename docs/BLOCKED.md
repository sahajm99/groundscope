# Blocked items

Nothing is blocked as of 2026-09-12 15:20 UTC.

## Resolved

### Supabase project paused (resolved 2026-09-12)
- The free project behind `DATABASE_URL` had auto-paused; the pooler returned
  `FATAL: (ENOTFOUND) tenant/user ... not found` and every knowledge question on the live
  site failed. Resumed from the dashboard; `/health` reports `db_reachable: true` again.
- Free Supabase projects pause after about a week idle. The keep-warm cron in
  `scripts/keepwarm.md` (ping `/health`, which touches the DB) prevents it.

### GitHub Actions secrets (resolved 2026-09-12)
- `LLM_API_KEY` and `TAVILY_API_KEY` are set on `sahajm99/groundscope` (`gh secret list`).
  The CI eval and smoke steps need them and fail loudly without them.

### Groq retired the Llama 3.x models (resolved 2026-09-12)
- `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` return 404. Defaults moved to
  `openai/gpt-oss-120b` (agent) and `openai/gpt-oss-20b` (fallback, eval judge) in
  `app/config.py`, `render.yaml`, `.env.example`. If `LLM_MODEL` was set by hand in the
  Render dashboard, that value overrides the blueprint and must be updated there too.

## Open decisions (not blockers)
- A free Gemini API key as a third LLM tier and separate judge quota (see `docs/PLAN.md` 0.3).
- Which public document seeds the live demo (see `docs/PLAN.md` 3.3).

## Open (2026-09-12 18:30 UTC)

### Live site: Render dashboard still carries the retired Llama model names
- `groundscope.onrender.com` retrieves and web-searches fine but synthesis fails with
  `NotFoundError`, because dashboard env values override `render.yaml`. Needs, in the Render
  dashboard: `LLM_MODEL=openai/gpt-oss-120b`, `LLM_FALLBACK_MODEL=openai/gpt-oss-20b`, and
  `GEMINI_API_KEY`. Nothing to deploy: Groq hosts the models; only the names change.

### Free-tier daily budgets exhausted; main badge run red
- At 18:10 UTC Groq reported `openai/gpt-oss-120b` at 199,585 / 200,000 tokens per day and
  `openai/gpt-oss-20b` at 199,123 / 200,000 (rolling 24 h window), and Gemini
  `gemini-3.5-flash-lite` at 500 / 500 requests per day (resets 07:00 UTC). One full eval run
  costs roughly 100K agent tokens; about ten runs happened today (local calibration, the
  regression proofs, PR and main runs).
- Main runs 34707065998 (one flaky web case, fixed in 2def8d4) and 34707733909 (quota) are red.
  The per-minute/per-day 429 fix is committed; it will be pushed and the main run re-run once
  the budgets return (Gemini 07:00 UTC; the bulk of the Groq tokens from ~12:00 UTC on).
- Cerebras tried 2026-09-12 18:45 UTC: the key is valid but the account is on PayGo; every
  model (`gpt-oss-120b`, `qwen-3.8-27b`) returns 402 Payment Required. Not usable under the
  free-tier rule unless the Billing tab offers a free plan. The `.env` lines are commented out.
- Capacity option (needs you): a free Cerebras key (cloud.cerebras.ai; 1M tokens/day, hosts
  `gpt-oss-120b`, OpenAI-compatible) as the second-provider tier (`LLM_FALLBACK_BASE_URL`,
  `LLM_FALLBACK_API_KEY`) would give five times today's total budget and keep CI runs from
  competing with the live site.
