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
