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
