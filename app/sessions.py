"""Session identity + lightweight abuse guards (in-memory; fine for a demo).

- Session = a UUID cookie. Uploads are scoped to it; purged after TTL.
- Per-IP rate limit (X-Forwarded-For first hop) + a global daily cap backstop.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response

from app.config import settings

COOKIE = "gs_session"

# session_id -> last_seen epoch
_sessions: dict[str, float] = {}
# ip -> (count, window_start)
_ip_hits: dict[str, tuple[int, float]] = {}
# (day, count)
_daily: list = [0, 0.0]


def _valid_sid(sid: str | None) -> bool:
    """A 32-char hex UUID. Trusting a returning cookie (not requiring it to be in
    the in-memory map) keeps a user's uploads accessible across server restarts."""
    return bool(sid) and len(sid) == 32 and all(c in "0123456789abcdef" for c in sid)


def get_or_create_session(request: Request, response: Response) -> str:
    sid = request.cookies.get(COOKIE)
    if not _valid_sid(sid):
        sid = uuid.uuid4().hex
        response.set_cookie(COOKIE, sid, max_age=settings.session_ttl_seconds, httponly=True, samesite="lax")
    _sessions[sid] = time.time()
    return sid


def client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limited(ip: str) -> bool:
    now = time.time()
    count, start = _ip_hits.get(ip, (0, now))
    if now - start > 60:
        _ip_hits[ip] = (1, now)
        return False
    _ip_hits[ip] = (count + 1, start)
    return count + 1 > settings.rate_limit_per_min


def daily_cap_reached() -> bool:
    day = int(time.time() // 86400)
    if _daily[1] != day:
        _daily[0], _daily[1] = 0, day
    _daily[0] += 1
    return _daily[0] > settings.global_daily_cap


def purge_expired() -> list[str]:
    """Return session_ids whose TTL elapsed (caller deletes their data)."""
    now = time.time()
    ttl = settings.session_ttl_seconds
    expired = [sid for sid, seen in _sessions.items() if now - seen > ttl]
    for sid in expired:
        _sessions.pop(sid, None)
    return expired
