# Keep-warm (free-tier cold-start mitigation)

Free hosts (Render/Fly) sleep on idle, and Supabase pauses after ~7 days idle.
A recruiter clicking a cold demo would hit a 30-60s wake. Mitigate by pinging
`/health` every ~10 minutes — `/health` also runs a trivial DB query, so it keeps
**both** the host and the Supabase project warm.

## Setup (free, no code)
1. Sign up at https://cron-job.org (free).
2. New cron job:
   - URL: `https://<your-app>.onrender.com/health`
   - Schedule: every 10 minutes
3. Save.

## Watch the monthly-hours math
A never-sleeping Render free web service consumes compute-hours. If a 10-min ping
keeps it always-on and that exceeds the free monthly hours, either:
- widen the interval (e.g. 14 min), or
- accept scheduled off-hours sleep; the frontend's "waking up…" state covers the
  occasional cold click.

Confirm the chosen host's free monthly-hours budget before relying on always-on.
