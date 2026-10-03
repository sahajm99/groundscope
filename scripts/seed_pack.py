"""Seed a corpus pack (data/packs/<pack>) into the database.

Usage:  python -m scripts.seed_pack data/packs/vedanta [--session GLOBAL] [--only isha.aurobindo ...]

Every canonical file becomes one `documents` row with a deterministic doc_id
(`pack:<pack>:<doc_slug>`), so re-running replaces that file's chunks instead of duplicating
them. Each record from the build is one chunk: the text stored (and embedded) is the record's
contextual header followed by its text, and the record's provenance goes into `metadata` so
retrieval can cite "Isha 2 · Commentary · Shankara, tr. Swami Gambhirananda". `page_number`
is the unit's ordinal within the file, so the legacy "p.N" citation still means something.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

from app import storage
from app.config import settings
from app.ingestion.canonical import build_pack, load_manifest
from app.ingestion.embedder import get_embedder

_META_FIELDS = (
    "id", "pack", "text_id", "text_title", "ref", "unit", "unit_label", "unit_meta", "part", "part_label",
    "layer", "language", "school", "commentator", "translator", "doc_slug", "source_file", "citation",
    "crossrefs", "chunk_index", "chunk_count",
)


def seed(pack_dir: str, session_id: str | None = None, only: set[str] | None = None) -> dict:
    if not settings.db_configured:
        raise SystemExit("DATABASE_URL not set.")
    pack = Path(pack_dir)
    manifest = load_manifest(pack)
    pack_id = manifest["pack"]["id"]
    session_id = session_id or manifest["pack"].get("seed_session", storage.GLOBAL_SESSION)
    storage.init_schema()

    records = build_pack(pack)
    by_doc: OrderedDict[str, list[dict]] = OrderedDict()
    for r in records:
        by_doc.setdefault(r["doc_slug"], []).append(r)

    embedder = get_embedder()
    result: dict = {"pack": pack_id, "session": session_id, "docs": {}}
    for slug, recs in by_doc.items():
        if only and slug not in only:
            continue
        doc_id = f"pack:{pack_id}:{slug}"
        storage.delete_document(doc_id)
        unit_page: dict[str, int] = {}
        for r in recs:
            unit_page.setdefault(r["ref"], len(unit_page) + 1)
        texts = [f"{r['header']}\n\n{r['text']}" for r in recs]
        embeddings = embedder.embed(texts)
        rows = [
            (unit_page[r["ref"]], i, texts[i], embeddings[i], {k: r.get(k) for k in _META_FIELDS})
            for i, r in enumerate(recs)
        ]
        storage.add_document(session_id, doc_id, f"{slug}.md", len(unit_page), rows)
        result["docs"][slug] = {"doc_id": doc_id, "records": len(rows), "units": len(unit_page)}
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pack_dir")
    ap.add_argument("--session", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    out = seed(args.pack_dir, args.session, set(args.only) if args.only else None)
    print(f"pack '{out['pack']}' seeded as session {out['session']}:")
    for slug, d in out["docs"].items():
        print(f"  {slug:40s} {d['records']:4d} chunks, {d['units']:3d} units  ({d['doc_id']})")
