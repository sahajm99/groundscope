"""Supabase Postgres storage: pgvector chunks + documents metadata.

Both retrieval tools are SESSION-SCOPED: a query only sees the caller's own
uploads plus the one global, read-only seeded corpus (session_id = 'GLOBAL').
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def _vec(values: list[float]) -> "np.ndarray":
    """pgvector's psycopg adapter maps numpy float32 arrays to the vector type
    (a plain Python list is sent as float8[], which the <=> operator rejects)."""
    return np.asarray(values, dtype=np.float32)

GLOBAL_SESSION = "GLOBAL"  # the seeded sample book every session can query


def _connect(register: bool = True) -> psycopg.Connection:
    """Connect. register=True adapts the pgvector type (requires the extension
    to already exist) — use register=False for bootstrap/health connections."""
    conn = psycopg.connect(settings.database_url, autocommit=True)
    if register:
        register_vector(conn)
    return conn


def ping_db() -> bool:
    with _connect(register=False) as conn:
        conn.execute("SELECT 1")
    return True


def init_schema() -> None:
    """Create the extension + tables + indexes. Idempotent."""
    dim = settings.vector_dim
    with _connect(register=False) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          bigserial PRIMARY KEY,
                session_id  text NOT NULL,
                doc_id      text NOT NULL,
                file_name   text NOT NULL,
                page_number int  NOT NULL,
                chunk_index int  NOT NULL,
                text        text NOT NULL,
                embedding   vector({dim}) NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id       text PRIMARY KEY,
                session_id   text NOT NULL,
                file_name    text NOT NULL,
                pages        int  NOT NULL,
                chunk_count  int  NOT NULL,
                uploaded_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS chunks_session_idx ON chunks (session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS docs_session_idx ON documents (session_id)")


@dataclass
class Hit:
    text: str
    file_name: str
    page_number: int
    distance: float  # cosine distance: lower = closer


def add_document(
    session_id: str,
    doc_id: str,
    file_name: str,
    pages: int,
    rows: list[tuple[int, int, str, list[float]]],
) -> int:
    """Insert chunks + a documents row. rows = (page_number, chunk_index, text, embedding)."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (session_id, doc_id, file_name, page_number, chunk_index, text, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(session_id, doc_id, file_name, pg, ci, txt, _vec(emb)) for (pg, ci, txt, emb) in rows],
            )
        conn.execute(
            "INSERT INTO documents (doc_id, session_id, file_name, pages, chunk_count)"
            " VALUES (%s, %s, %s, %s, %s) ON CONFLICT (doc_id) DO UPDATE SET chunk_count = EXCLUDED.chunk_count",
            (doc_id, session_id, file_name, pages, len(rows)),
        )
    return len(rows)


def vector_search(session_id: str, query_embedding: list[float], limit: int = 6) -> list[Hit]:
    """Cosine-distance search scoped to this session + the global corpus."""
    qv = _vec(query_embedding)
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT text, file_name, page_number, (embedding <=> %s) AS distance
            FROM chunks
            WHERE session_id IN (%s, %s)
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qv, session_id, GLOBAL_SESSION, qv, limit),
        )
        return [Hit(text=r[0], file_name=r[1], page_number=r[2], distance=float(r[3])) for r in cur.fetchall()]


def list_documents(session_id: str) -> list[dict]:
    """Metadata for this session + the global corpus."""
    with _connect() as conn:
        cur = conn.execute(
            "SELECT file_name, pages, chunk_count, uploaded_at FROM documents"
            " WHERE session_id IN (%s, %s) ORDER BY uploaded_at",
            (session_id, GLOBAL_SESSION),
        )
        return [
            {"file_name": r[0], "pages": r[1], "chunk_count": r[2], "uploaded_at": str(r[3])}
            for r in cur.fetchall()
        ]


def purge_session(session_id: str) -> None:
    """Delete a session's uploads (never the global corpus)."""
    if session_id == GLOBAL_SESSION:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE session_id = %s", (session_id,))
        conn.execute("DELETE FROM documents WHERE session_id = %s", (session_id,))
