"""Upload cleanup: a visitor's uploads die with their session, even across a restart."""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app import main, sessions, storage


def test_purge_once_sweeps_uploads_with_the_session_ttl_and_forgets_idle_sessions(monkeypatch):
    seen: list[int] = []
    monkeypatch.setattr(main.settings, "database_url", "postgresql://configured")
    monkeypatch.setattr(storage, "purge_expired_uploads", lambda ttl: seen.append(ttl) or 3)
    idle, live = uuid.uuid4().hex, uuid.uuid4().hex
    monkeypatch.setitem(sessions._sessions, idle, time.time() - 2 * main.settings.session_ttl_seconds)
    monkeypatch.setitem(sessions._sessions, live, time.time())

    assert main.purge_once() == 3
    assert seen == [main.settings.session_ttl_seconds]
    assert idle not in sessions._sessions and live in sessions._sessions


def test_purge_once_survives_a_database_error(monkeypatch):
    """A paused database must not take the cleanup loop, or startup, down with it."""

    def boom(ttl):
        raise RuntimeError("tenant/user not found")

    monkeypatch.setattr(main.settings, "database_url", "postgresql://configured")
    monkeypatch.setattr(storage, "purge_expired_uploads", boom)
    assert main.purge_once() == 0


async def test_the_cleanup_loop_keeps_running_at_its_interval(monkeypatch):
    runs: list[int] = []
    monkeypatch.setattr(main, "purge_once", lambda: runs.append(1) or 0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(main._cleanup_loop(interval_s=0.01), timeout=0.2)
    assert len(runs) >= 3


def test_startup_starts_the_cleanup_and_shutdown_stops_it(monkeypatch):
    """The purge functions existed for months and nothing called them. This pins the wiring."""

    async def fake_get_bus():
        class Bus:
            def servers(self):
                return {}

        return Bus()

    monkeypatch.setattr("app.main.get_bus", fake_get_bus)
    monkeypatch.setattr(main, "purge_once", lambda: 0)
    with TestClient(main.app):
        task = main._cleanup_task
        assert task is not None and not task.done()
    assert task.done()
