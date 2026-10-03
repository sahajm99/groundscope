"""Groundscope — observable agentic RAG demo. FastAPI entrypoint.

Single deployable: serves the API and the same-origin static frontend, so the
SSE trace stream never crosses origins.
"""

import asyncio
import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent.toolbus import close_bus, get_bus, peek_bus
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("groundscope")

app = FastAPI(title="Groundscope", version="2.0.1")

STATIC_DIR = Path(__file__).parent.parent / "static"


@app.on_event("startup")
def _startup():
    from app.ingestion.embedder import get_embedder

    get_embedder()  # validates embedding dim vs VECTOR_DIM
    if settings.db_configured:
        try:
            from app.storage import init_schema

            init_schema()
        except Exception as e:  # noqa: BLE001
            logger.warning("Schema init skipped: %s", e)


@app.on_event("startup")
async def _warm_tool_bus():
    """Start the MCP servers (keepalive retrieval child) before the first question, and give
    the loop a thread pool that fits fan-out (default is 5 threads on a 1-vCPU host)."""
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=16, thread_name_prefix="gs"))
    try:
        bus = await get_bus()
        logger.info("MCP tool bus: %s", bus.servers())
    except Exception as e:  # noqa: BLE001
        logger.warning("MCP tool bus failed to load: %s", e)


@app.on_event("shutdown")
async def _close_tool_bus():
    await close_bus()


CLEANUP_INTERVAL_S = 600
_cleanup_task: asyncio.Task | None = None


def purge_once() -> int:
    """Forget idle sessions and delete uploads past the session TTL. Returns the number of
    documents removed. Never raises: a paused database must not take the loop or startup down."""
    from app import sessions, storage

    sessions.purge_expired()
    if not settings.db_configured:
        return 0
    try:
        return storage.purge_expired_uploads(settings.session_ttl_seconds)
    except Exception as e:  # noqa: BLE001
        logger.warning("Upload cleanup skipped: %s: %s", type(e).__name__, e)
        return 0


async def _cleanup_loop(interval_s: float = CLEANUP_INTERVAL_S) -> None:
    """Sweep now and then every interval. On a free host that sleeps when idle, 'now' is each
    wake-up, so an upload is gone the first time the service runs after its hour is over."""
    while True:
        removed = await asyncio.to_thread(purge_once)
        if removed:
            logger.info("Purged %d expired upload(s).", removed)
        await asyncio.sleep(interval_s)


@app.on_event("startup")
async def _start_cleanup():
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop(), name="upload-cleanup")


@app.on_event("shutdown")
async def _stop_cleanup():
    global _cleanup_task
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task
        _cleanup_task = None


from app.api import ask as ask_api  # noqa: E402
from app.api import ingest as ingest_api  # noqa: E402

app.include_router(ask_api.router)
app.include_router(ingest_api.router)


@app.get("/health")
async def health():
    """Liveness + subsystem readiness. Also pinged by the keep-warm cron;
    touches the DB so Supabase doesn't pause on idle."""
    db_ok = False
    if settings.db_configured:
        try:
            from app.storage import ping_db

            db_ok = ping_db()
        except Exception as e:  # noqa: BLE001
            logger.warning("DB ping failed: %s", e)
    # Read the cached bus only: a health probe must never trigger a (slow) rebuild.
    bus = peek_bus()
    mcp = bus.servers() if bus is not None else {}
    return {
        "status": "ok",
        "llm_configured": settings.llm_configured,
        "db_configured": settings.db_configured,
        "db_reachable": db_ok,
        "web_search_configured": settings.web_search_configured,
        "embed_provider": settings.embed_provider,
        "vector_dim": settings.vector_dim,
        "mcp": mcp,
    }


@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        # no-store so iterating on the frontend never serves stale JS
        return FileResponse(idx, headers={"Cache-Control": "no-store"})
    return {"service": "groundscope", "hint": "frontend not built yet — see /health"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
