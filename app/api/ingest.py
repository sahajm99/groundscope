"""POST /ingest — upload a document into the caller's session.

Extract -> chunk -> embed -> store (chunks + metadata), all scoped to the
session UUID. Enforces size + page caps for the public demo.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, Response, UploadFile, File, HTTPException

from app.config import settings
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import get_embedder
from app.ingestion.extract import extract_pages, NoTextError
from app import sessions, storage

router = APIRouter()


@router.post("/ingest")
async def ingest(request: Request, response: Response, file: UploadFile = File(...)):
    if not settings.db_configured:
        raise HTTPException(503, "Storage isn't configured yet.")

    sid = sessions.get_or_create_session(request, response)
    data = await file.read()

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {settings.max_upload_mb} MB).")

    try:
        pages = extract_pages(data, file.filename or "upload")
    except NoTextError as e:
        raise HTTPException(422, str(e))

    if len(pages) > settings.max_pages:
        raise HTTPException(413, f"Too many pages (max {settings.max_pages}).")

    chunks = chunk_pages(pages, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    if not chunks:
        raise HTTPException(422, "No text chunks produced from this document.")

    embeddings = get_embedder().embed([c.text for c in chunks])
    rows = [(c.page_number, c.chunk_index, c.text, embeddings[i]) for i, c in enumerate(chunks)]

    doc_id = uuid.uuid4().hex
    storage.add_document(sid, doc_id, file.filename or "upload", len(pages), rows)

    return {
        "doc_id": doc_id,
        "file_name": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
    }


@router.get("/documents")
def documents(request: Request, response: Response):
    """List the documents available to this session (uploads + seeded corpus)."""
    if not settings.db_configured:
        return {"documents": []}
    sid = sessions.get_or_create_session(request, response)
    return {"documents": storage.list_documents(sid)}
