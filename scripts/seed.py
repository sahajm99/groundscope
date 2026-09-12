"""Seed the global, read-only sample corpus every session can query.

Usage:  python -m scripts.seed path/to/book.pdf [more files...]
Ingests each file under session_id = GLOBAL so the demo works with zero upload.
A file already seeded (same name under GLOBAL) is skipped, so the script is idempotent.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from app import storage
from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import get_embedder
from app.ingestion.extract import extract_pages


def main(path: str) -> None:
    if not settings.db_configured:
        raise SystemExit("DATABASE_URL not set.")
    storage.init_schema()

    data = Path(path).read_bytes()
    name = Path(path).name
    if any(d["file_name"] == name for d in storage.list_documents(storage.GLOBAL_SESSION)):
        print(f"Already seeded: '{name}' (skipped).")
        return
    pages = extract_pages(data, name)
    chunks = chunk_pages(pages, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    embeddings = get_embedder().embed([c.text for c in chunks])
    rows = [(c.page_number, c.chunk_index, c.text, embeddings[i]) for i, c in enumerate(chunks)]

    storage.add_document(storage.GLOBAL_SESSION, uuid.uuid4().hex, name, len(pages), rows)
    print(f"Seeded '{name}' as GLOBAL: {len(pages)} pages, {len(chunks)} chunks.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.seed path/to/file.pdf [more files...]")
    for _p in sys.argv[1:]:
        main(_p)
