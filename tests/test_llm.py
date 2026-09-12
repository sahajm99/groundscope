"""The tiered LLM router: an explicit model override becomes the first tier."""

from __future__ import annotations

from types import SimpleNamespace

from app.agent import llm


class FakeClient:
    def __init__(self, log):
        self.log = log
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.log.append(kw["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])


def test_complete_json_uses_the_model_override_first(monkeypatch):
    log: list[str] = []
    monkeypatch.setattr(llm, "_client", lambda base, key: FakeClient(log))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    assert llm.complete_json("sys", "user", model="judge-model") == {"ok": True}
    assert log == ["judge-model"]


def test_default_tiers_unchanged_without_override(monkeypatch):
    log: list[str] = []
    monkeypatch.setattr(llm, "_client", lambda base, key: FakeClient(log))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    llm.complete("sys", "user")
    assert log == [llm.settings.llm_model]


def test_strict_model_does_not_fall_back(monkeypatch):
    """The eval judge must never silently become the agent model."""
    import pytest

    class Failing(FakeClient):
        def _create(self, **kw):
            self.log.append(kw["model"])
            raise RuntimeError("429")

    log: list[str] = []
    monkeypatch.setattr(llm, "_client", lambda base, key: Failing(log))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)  # the last tier retries once after a 429
    with pytest.raises(RuntimeError):
        llm.complete("s", "u", model="judge-model", strict_model=True)
    assert set(log) == {"judge-model"} and len(log) == 2  # retried the SAME model, never fell back


def test_complete_ex_reports_the_model_that_answered(monkeypatch):
    log: list[str] = []

    class PrimaryDown(FakeClient):
        def _create(self, **kw):
            self.log.append(kw["model"])
            if kw["model"] == llm.settings.llm_model:
                raise RuntimeError("429")
            return super()._create(**kw)

    monkeypatch.setattr(llm, "_client", lambda base, key: PrimaryDown(log))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    text, used = llm.complete_ex("s", "u")
    assert used == llm.settings.llm_fallback_model and text == '{"ok": true}'


def test_gemini_key_adds_a_third_tier_on_its_own_endpoint(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "g")
    monkeypatch.setattr(llm.settings, "llm_fallback_api_key", "")
    tiers = llm._tiers()
    assert tiers[-1] == (llm.settings.gemini_model, llm.settings.gemini_base_url, "g")
    assert len(tiers) == 3


def test_no_gemini_key_means_no_third_tier(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm.settings, "llm_fallback_api_key", "")
    assert len(llm._tiers()) == 2


def test_complete_ex_can_target_a_specific_endpoint(monkeypatch):
    seen = {}

    def client(base, key):
        seen["base"], seen["key"] = base, key
        return FakeClient([])

    monkeypatch.setattr(llm, "_client", client)
    text, used = llm.complete_ex("s", "u", model="judge-x", strict_model=True, base_url="https://j/", api_key="jk")
    assert (seen["base"], seen["key"], used) == ("https://j/", "jk", "judge-x")


def test_throttle_keeps_requests_per_minute_under_the_cap(monkeypatch):
    """Gemini's free tier allows 15 requests/minute; the 16th call in a minute must wait."""
    now = {"t": 1000.0}
    slept = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s)))
    llm._throttles.clear()
    for _ in range(12):
        llm._throttle("https://generativelanguage.googleapis.com/v1beta/openai/", 12)
    assert not slept
    llm._throttle("https://generativelanguage.googleapis.com/v1beta/openai/", 12)
    assert slept and 0 < slept[0] <= 61  # waits out the minute (+0.5 s buffer)


def test_last_tier_retries_once_after_a_rate_limit(monkeypatch):
    calls = {"n": 0}

    class Flaky(FakeClient):
        def _create(self, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("Error code: 429 - RESOURCE_EXHAUSTED")
            return super()._create(**kw)

    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "_client", lambda base, key: Flaky([]))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    text, used = llm.complete_ex("s", "u", model="only", strict_model=True)
    assert text == '{"ok": true}' and calls["n"] == 2 and slept and slept[0] >= 20


# -- Per-minute vs per-day rate limits (2026-09-12 CI failure) ---------------------------
# Groq's free tier caps each model at 8K tokens/minute. A burst 429 says "try again in 1.2s";
# failing over on it spilled every burst to Gemini, whose 500 requests/day were gone by the
# afternoon, and 11 of 18 eval cases then had no model at all.


def test_retry_after_parses_groq_and_gemini_messages():
    import pytest

    p = llm._retry_after_s
    assert p(RuntimeError("Error code: 429 - Rate limit reached ... Please try again in 1.234s.")) == pytest.approx(1.234)
    assert p(RuntimeError("Error code: 429 - ... Please try again in 2m38.4s. Need more?")) == pytest.approx(158.4)
    assert p(RuntimeError("Error code: 429 - ... Please try again in 3h2m1s.")) == pytest.approx(10921.0)
    assert p(RuntimeError("Error code: 429 - ... Please retry in 17.41s.")) == pytest.approx(17.41)
    assert p(RuntimeError("Error code: 429 - no hint at all")) is None


def test_daily_quota_is_recognized_from_the_message():
    d = llm._is_daily_quota
    assert d(RuntimeError("429 ... 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' ... retry in 17s"))
    assert d(RuntimeError("429 Rate limit reached for model x on tokens per day (TPD): Limit 200000. try again in 3h"))
    assert not d(RuntimeError("429 Rate limit reached for model x on tokens per minute (TPM): Limit 8000. try again in 1.2s"))


def test_short_rate_limit_waits_and_retries_the_same_tier(monkeypatch):
    calls: list[str] = []

    class Burst(FakeClient):
        def _create(self, **kw):
            calls.append(kw["model"])
            if len(calls) == 1:
                raise RuntimeError("Error code: 429 - Rate limit reached for model x on tokens per minute (TPM): "
                                   "Limit 8000, Used 7000, Requested 3000. Please try again in 1.2s.")
            return super()._create(**kw)

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "_client", lambda base, key: Burst([]))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    text, used = llm.complete_ex("s", "u")
    assert used == llm.settings.llm_model and calls == [llm.settings.llm_model] * 2
    assert slept and 1.2 <= slept[0] <= 5


def test_daily_quota_fails_over_immediately_without_sleeping(monkeypatch):
    calls: list[str] = []

    class Daily(FakeClient):
        def _create(self, **kw):
            calls.append(kw["model"])
            if kw["model"] == llm.settings.llm_model:
                raise RuntimeError("Error code: 429 - Rate limit reached for model x on tokens per day (TPD): "
                                   "Limit 200000, Used 199000, Requested 3000. Please try again in 3h2m1s.")
            return super()._create(**kw)

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "_client", lambda base, key: Daily([]))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    text, used = llm.complete_ex("s", "u")
    assert used == llm.settings.llm_fallback_model and not slept


def test_last_tier_does_not_retry_a_daily_quota(monkeypatch):
    """Gemini's daily-cap 429 says 'retry in 17s' but means midnight; a 35 s pause is wasted."""
    import pytest

    class Gone(FakeClient):
        def _create(self, **kw):
            raise RuntimeError("Error code: 429 - GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 17s.")

    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "_client", lambda base, key: Gone([]))
    with pytest.raises(RuntimeError):
        llm.complete_ex("s", "u", model="only", strict_model=True)
    assert not slept


def test_each_tier_failure_is_logged_with_the_provider_message(monkeypatch, caplog):
    """Silent failover hid a day of Groq refusals; the reason each tier failed must be visible."""
    import logging

    class PrimaryDown(FakeClient):
        def _create(self, **kw):
            if kw["model"] == llm.settings.llm_model:
                raise RuntimeError("Error code: 429 - on tokens per day (TPD): Limit 200000, Used 199990")
            return super()._create(**kw)

    monkeypatch.setattr(llm, "_client", lambda base, key: PrimaryDown([]))
    monkeypatch.setattr(llm.settings, "llm_api_key", "k")
    with caplog.at_level(logging.WARNING, logger="app.agent.llm"):
        llm.complete_ex("s", "u")
    assert any(llm.settings.llm_model in r.message and "TPD" in r.message for r in caplog.records)
