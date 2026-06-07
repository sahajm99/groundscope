"""LLM access with a tiered model router + circuit breaker (Etech7 patterns).

Router tiers (tried in order, failing over on error):
  1. primary model (Groq Llama 3.3 70B)
  2. smaller same-provider model (Llama 3.1 8B instant)
  3. optional second provider (set LLM_FALLBACK_API_KEY + LLM_FALLBACK_BASE_URL)

Circuit breaker: after N consecutive primary failures the breaker opens for a
cooldown, during which calls skip the primary and go straight to the fallback —
preventing retry storms against a failing provider.

Each client is wrapped for LangSmith tracing when enabled.
"""

from __future__ import annotations

import json
import os
import time

from app.config import settings


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


def _client(base_url: str, api_key: str):
    if base_url not in _clients:
        from openai import OpenAI

        c = OpenAI(api_key=api_key, base_url=base_url)
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
    return tiers


def complete(system: str, user: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    tiers = _tiers()
    start = 1 if (_breaker.is_open() and len(tiers) > 1) else 0
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_err: Exception | None = None
    for i in range(start, len(tiers)):
        model, base, key = tiers[i]
        try:
            resp = _client(base, key).chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_tokens, messages=messages,
            )
            if i == 0:
                _breaker.record_success()
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            if i == 0:
                _breaker.record_failure()
            continue
    raise last_err if last_err else RuntimeError("no LLM tier available")


def complete_json(system: str, user: str) -> dict:
    """Ask for a JSON object back; tolerate fenced code blocks."""
    raw = complete(system + "\nRespond ONLY with a JSON object.", user, temperature=0.0, max_tokens=300).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
