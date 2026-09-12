# Blocked items

## Supabase project unreachable (live grounded-answer check)
- Observed 2026-09-12: `DATABASE_URL` (aws-1-us-east-1 session pooler) returns
  `FATAL: (ENOTFOUND) tenant/user postgres.<ref> not found`; the direct host
  `db.<ref>.supabase.co` does not resolve. This is what a **paused** (or deleted) free
  project looks like. The live site's `/health` reports `db_reachable: false` and every
  knowledge question fails with `OperationalError`.
- Not obtainable here: a Supabase personal access token (needed for the Management API
  restore endpoint) or dashboard access. No related email found in Gmail.
- **To unblock:** open https://supabase.com/dashboard, restore (unpause) the project, or
  create a new one and set `DATABASE_URL` on Render; then re-seed the corpus:
  `python -m scripts.seed data/sample.txt` (schema is created idempotently on boot).
- Everything else in v2.0 is proven without it: CI uses a hermetic pgvector container;
  the live web-fallback path keeps working because tool errors now degrade gracefully.

## GitHub Actions secrets for the eval job (needs one manual step)
- The autonomous session was not permitted to write repository secrets. The CI eval job
  (`evals` in `.github/workflows/ci.yml`) needs the Groq key and the Tavily key, and it
  fails with an explicit "missing secret" error rather than skipping.
- **To unblock (two commands, values from your local `.env`):**
  ```bash
  gh secret set LLM_API_KEY    -R sahajm99/groundscope   # paste the Groq key
  gh secret set TAVILY_API_KEY -R sahajm99/groundscope   # paste the Tavily key
  ```
  Then re-run the latest workflow (`gh run rerun --failed`) and the badge turns green.
