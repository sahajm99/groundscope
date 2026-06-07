"""LLM client (OpenAI-compatible, default Groq). Lazy so a missing key never
breaks import/build."""

from __future__ import annotations

import json

from app.config import settings

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def complete(system: str, user: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
    resp = _get_client().chat.completions.create(
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def complete_json(system: str, user: str) -> dict:
    """Ask for a JSON object back; tolerate fenced code blocks."""
    raw = complete(system + "\nRespond ONLY with a JSON object.", user, temperature=0.0, max_tokens=300)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start : end + 1])
        raise
