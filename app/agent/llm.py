"""LLM access with a tiered model router + circuit breaker (Etech7 patterns).

Router tiers (tried in order, failing over on error):
  1. primary model (Groq, openai/gpt-oss-120b)
  2. smaller same-provider model (Groq, openai/gpt-oss-20b)
  3. optional second provider (set LLM_FALLBACK_API_KEY + LLM_FALLBACK_BASE_URL)
  4. Gemini (gemini-3.5-flash-lite) when GEMINI_API_KEY is set

A per-minute rate limit is waited out on the same tier; a per-day cap fails over
(see same_tier_wait_s). Circuit breaker: after N consecutive primary failures the
breaker opens for a cooldown, during which calls skip the primary and go straight to
the fallback, preventing retry storms against a failing provider.

Each client is wrapped for LangSmith tracing when enabled.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from app.config import settings

log = logging.getLogger(__name__)


class _Breaker:
    def __init__(self, threshold: int, cooldown: float):
        self.threshold = threshold
        self.cooldown = cooldown
        self.fails = 0
        self.open_until = 0.0

    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def record_success(self) -> None:
        self.fails = 0

    def record_failure(self) -> None:
        self.fails += 1
        if self.fails >= self.threshold:
            self.open_until = time.monotonic() + self.cooldown
            self.fails = 0


_breaker = _Breaker(settings.breaker_threshold, settings.breaker_cooldown_s)
_clients: dict = {}

# Per-endpoint request throttle. Gemini's free tier allows 15 requests/minute; when both
# Groq tiers are out of daily budget the whole agent plus the eval judge fall through to it,
# so the client itself must stay under the cap instead of collecting 429s.
_throttles: dict[str, list[float]] = {}
_RPM_CAPS = {"generativelanguage.googleapis.com": 12}
RATE_LIMIT_RETRY_PAUSE_S = 35.0


def _rpm_cap(base_url: str) -> int | None:
    for host, cap in _RPM_CAPS.items():
        if host in base_url:
            return cap
    return None


def _throttle(base_url: str, cap: int) -> None:
    """Block until fewer than `cap` requests were started on this endpoint in the last minute."""
    now = time.monotonic()
    stamps = [t for t in _throttles.get(base_url, []) if now - t < 60.0]
    if len(stamps) >= cap:
        wait = 60.0 - (now - stamps[0]) + 0.5
        if wait > 0:
            time.sleep(wait)
        now = time.monotonic()
        stamps = [t for t in stamps if now - t < 60.0]
    stamps.append(now)
    _throttles[base_url] = stamps


def _is_rate_limit(e: BaseException) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return "429" in s or "rate limit" in s or "ratelimit" in s or "resource_exhausted" in s


# Two kinds of 429 that need opposite responses. Groq's free tier caps each model at 8K
# tokens per MINUTE: a burst is refused with "try again in 1.2s" and is fine a moment later.
# A per-DAY cap ("tokens per day (TPD)", Gemini's "...PerDay..." quota id) is not; only
# another tier can help. On 2026-09-12 every per-minute refusal failed over to Gemini, which
# absorbed the spillover all day until its 500 requests/day were gone, and then 11 of 18 eval
# cases had no model at all.
_DAILY_MARKERS = ("per day", "perday", "(tpd)", "(rpd)")
MAX_SAME_TIER_WAIT_S = 30.0
_RETRY_HINT = re.compile(r"(?:try again|retry) in ((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)", re.I)


def _retry_after_s(e: BaseException) -> float | None:
    """Seconds the provider asked us to wait ('try again in 2m38.4s', 'retry in 17.41s'), or
    None when the message carries no hint."""
    m = _RETRY_HINT.search(str(e))
    if not m or not m.group(1):
        return None
    hint = m.group(1)
    h, mi, s = re.search(r"(\d+)h", hint), re.search(r"(\d+)m", hint), re.search(r"(\d+(?:\.\d+)?)s", hint)
    return (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + (float(s.group(1)) if s else 0.0)


def _is_daily_quota(e: BaseException) -> bool:
    s = str(e).lower()
    return any(k in s for k in _DAILY_MARKERS)


def same_tier_wait_s(e: BaseException) -> float | None:
    """How long to wait before retrying the SAME model, or None to fail over: a per-minute
    burst with a short hint is waited out; a daily cap, or a long or missing hint, fails over."""
    if not _is_rate_limit(e) or _is_daily_quota(e):
        return None
    w = _retry_after_s(e)
    if w is None or w > MAX_SAME_TIER_WAIT_S:
        return None
    return w + 0.5


def _client(base_url: str, api_key: str):
    if base_url not in _clients:
        from openai import OpenAI

        # A stalled provider must not hold a worker thread for the client's default 600 s.
        c = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=1)
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and os.getenv("LANGSMITH_API_KEY"):
            try:
                from langsmith.wrappers import wrap_openai

                c = wrap_openai(c)
            except Exception:  # noqa: BLE001
                pass
        _clients[base_url] = c
    return _clients[base_url]


def _tiers() -> list[tuple[str, str, str]]:
    tiers = [(settings.llm_model, settings.llm_base_url, settings.llm_api_key)]
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model:
        tiers.append((settings.llm_fallback_model, settings.llm_base_url, settings.llm_api_key))
    if settings.llm_fallback_api_key and settings.llm_fallback_base_url:
        tiers.append((settings.llm_fallback_model or settings.llm_model,
                      settings.llm_fallback_base_url, settings.llm_fallback_api_key))
    if settings.gemini_api_key:
        tiers.append((settings.gemini_model, settings.gemini_base_url, settings.gemini_api_key))
    return tiers


def complete_ex(
    system: str, user: str, temperature: float = 0.2, max_tokens: int = 700,
    model: str | None = None, strict_model: bool = False,
    base_url: str | None = None, api_key: str | None = None,
) -> tuple[str, str]:
    """Route through the tiers; return (text, model that answered).

    An explicit `model` (e.g. the eval judge) becomes the first tier, on `base_url`/`api_key`
    when given (a different provider) or on the primary provider otherwise, with the
    configured tiers as fallbacks; `strict_model=True` disables the fallbacks so a judge can
    never silently become the agent model."""
    tiers = _tiers()
    if model:
        tiers = [(model, base_url or settings.llm_base_url, api_key or settings.llm_api_key)]
        if not strict_model:
            tiers += [t for t in _tiers() if t[0] != model]
    start = 1 if (_breaker.is_open() and len(tiers) > 1 and not model) else 0
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for i in range(start, len(tiers)):
        tier_model, base, key = tiers[i]
        is_last = i == len(tiers) - 1
        waits = 0
        while True:
            cap = _rpm_cap(base)
            if cap:
                _throttle(base, cap)
            try:
                resp = _client(base, key).chat.completions.create(
                    model=tier_model, temperature=temperature, max_tokens=max_tokens, messages=messages,
                )
                if i == 0 and not model:
                    _breaker.record_success()
                return (resp.choices[0].message.content or "").strip(), tier_model
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.warning("llm tier %s failed: %s: %s", tier_model, type(e).__name__, str(e)[:300])
                if i == 0 and not model:
                    _breaker.record_failure()
                # A per-minute burst is waited out on the same tier (at most twice).
                w = same_tier_wait_s(e)
                if w is not None and waits < 2:
                    waits += 1
                    time.sleep(w)
                    continue
                # The last tier gets one more try after a rate-limit window, unless the cap
                # is a daily one that no pause fixes; earlier tiers fail over instead.
                if is_last and waits == 0 and _is_rate_limit(e) and not _is_daily_quota(e):
                    waits += 1
                    time.sleep(RATE_LIMIT_RETRY_PAUSE_S)
                    continue
                break
    raise last_err if last_err else RuntimeError("no LLM tier available")


def complete(
    system: str, user: str, temperature: float = 0.2, max_tokens: int = 700,
    model: str | None = None, strict_model: bool = False,
) -> str:
    return complete_ex(system, user, temperature, max_tokens, model=model, strict_model=strict_model)[0]


def complete_json(
    system: str, user: str, model: str | None = None, max_tokens: int = 300, strict_model: bool = False
) -> dict:
    """Ask for a JSON object back; tolerate fenced code blocks."""
    return complete_json_ex(system, user, model=model, max_tokens=max_tokens, strict_model=strict_model)[0]


def complete_json_ex(
    system: str, user: str, model: str | None = None, max_tokens: int = 300, strict_model: bool = False,
    base_url: str | None = None, api_key: str | None = None,
) -> tuple[dict, str]:
    """complete_json plus the model that answered."""
    raw, used = complete_ex(system + "\nRespond ONLY with a JSON object.", user, temperature=0.0,
                            max_tokens=max_tokens, model=model, strict_model=strict_model,
                            base_url=base_url, api_key=api_key)
    return _parse_json(raw.strip()), used


def _parse_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
