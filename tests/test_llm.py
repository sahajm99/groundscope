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
    with pytest.raises(RuntimeError):
        llm.complete("s", "u", model="judge-model", strict_model=True)
    assert log == ["judge-model"]


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
