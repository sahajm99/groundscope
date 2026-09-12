"""Shared fixtures.

Integration tests need the pgvector container:
  docker run -d --name gs-pg -p 5433:5432 -e POSTGRES_PASSWORD=gs -e POSTGRES_USER=gs \
    -e POSTGRES_DB=groundscope pgvector/pgvector:pg16
Locally an unreachable DB skips those tests. In CI (CI=true) it FAILS: never skip silently.
"""

from __future__ import annotations

import os

import pytest

DEFAULT_TEST_DB = "postgresql://gs:gs@localhost:5433/groundscope"


def pytest_configure(config):
    # Must run before app.config loads .env (load_dotenv never overrides existing vars).
    os.environ["DATABASE_URL"] = os.environ.get("GS_TEST_DATABASE_URL", os.environ.get("DATABASE_URL", DEFAULT_TEST_DB))
    os.environ.setdefault("AGENT_ENGINE", "langgraph")


def _in_ci() -> bool:
    return os.environ.get("CI", "").lower() == "true"


@pytest.fixture(scope="session")
def db_url() -> str:
    return os.environ["DATABASE_URL"]


@pytest.fixture(scope="session")
def db(db_url: str) -> str:
    """Ensure the schema exists and the corpus is seeded. Returns the URL."""
    import psycopg

    try:
        with psycopg.connect(db_url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as e:  # noqa: BLE001
        if _in_ci():
            pytest.fail(f"CI requires the test database at {db_url}: {e}")
        pytest.skip(f"test database unreachable at {db_url}: {e}")

    from app import storage

    storage.init_schema()
    if not any(d["file_name"] == "sample.txt" for d in storage.list_documents("__conftest__")):
        from scripts.seed import main as seed

        seed("data/sample.txt")
    return db_url


@pytest.fixture
def llm_keys() -> None:
    from app.config import settings

    if not settings.llm_configured:
        if _in_ci():
            pytest.fail("CI requires LLM_API_KEY (repository secret)")
        pytest.skip("LLM_API_KEY not set")
