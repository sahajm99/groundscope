"""Hybrid retrieval against the local pgvector: the keyword leg must contribute."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# See tests/test_canonical.py: the Vedanta texts are private and absent from a clone.
needs_pack_texts = pytest.mark.skipif(
    not Path("data/packs/vedanta/texts/isha").is_dir(),
    reason="the Vedanta pack's texts are not distributed with the repository",
)


def test_keyword_leg_matches_on_any_term_not_all(db):
    """plainto_tsquery ANDs every word, so a long question with one unmatched word made the
    BM25 leg return nothing and the fused ranking degrade to dense-only (found by the evals:
    Meridian facts were missed). Terms are ORed; rank still rewards more matches."""
    from app import storage
    from app.ingestion.embedder import get_embedder

    q = "readmission rate zzzunmatchedwordzzz"
    # An unrelated embedding so only the keyword leg can surface the chunk.
    emb = get_embedder().embed(["orbital mechanics of Jupiter's moons"])[0]
    hits, _ = storage.hybrid_search("nosuch", emb, q, limit=3)
    assert any("readmission" in h.text.lower() for h in hits), [h.file_name for h in hits]


def test_keyword_only_query_survives_punctuation(db):
    from app import storage
    from app.ingestion.embedder import get_embedder

    q = "What's Meridian's 'Compass' program? (30-day readmissions!)"
    emb = get_embedder().embed(["orbital mechanics of Jupiter's moons"])[0]
    hits, _ = storage.hybrid_search("nosuch", emb, q, limit=3)
    assert any("Compass" in h.text for h in hits)


def test_chunk_metadata_roundtrips_through_hybrid_search(db):
    """Corpus-pack chunks carry a metadata dict; plain uploads leave it None. Both legs of the
    hybrid search must return it so the retrieval tool can cite by unit instead of page."""
    from app import storage
    from app.ingestion.embedder import get_embedder

    text = "Isha Upanishad | Verse 2 | Commentary | Shankara\n\nKurvan eva iha, verily by doing here zzmetaprobe."
    emb = get_embedder().embed([text])[0]
    meta = {"citation": "Isha 2 · Commentary · Shankara, tr. Swami Gambhirananda", "ref": "isha.2", "part": "commentary"}
    storage.delete_document("test:meta:doc")
    storage.add_document("__meta_test__", "test:meta:doc", "isha.test.md", 1, [(1, 0, text, emb, meta)])
    try:
        # dense leg
        hits, _ = storage.hybrid_search("__meta_test__", emb, "kurvan eva iha", limit=3)
        hit = next(h for h in hits if h.metadata and h.metadata.get("ref") == "isha.2")
        assert (hit.metadata or {})["citation"].startswith("Isha 2")
        # keyword-only leg (unrelated embedding)
        other = get_embedder().embed(["orbital mechanics of Jupiter's moons"])[0]
        hits, _ = storage.hybrid_search("__meta_test__", other, "zzmetaprobe", limit=3)
        assert any(h.metadata and h.metadata.get("ref") == "isha.2" for h in hits)
        # legacy 4-tuple rows still insert, with NULL metadata
        storage.add_document("__meta_test__", "test:meta:doc", "isha.test.md", 1, [(1, 1, "plain chunk zzplainprobe", emb)])
        hits, _ = storage.hybrid_search("__meta_test__", other, "zzplainprobe", limit=3)
        assert any(h.metadata is None and "zzplainprobe" in h.text for h in hits)
    finally:
        storage.delete_document("test:meta:doc")


@needs_pack_texts
def test_seed_pack_makes_a_verse_findable_by_citation(db):
    """Seeding the Vedanta pack's smallest file: a query in Aurobindo's wording must come back
    labelled with its verse, not a page number."""
    from app import storage
    from app.ingestion.embedder import get_embedder
    from scripts.seed_pack import seed

    out = seed("data/packs/vedanta", session_id="__pack_test__", only={"isha.aurobindo"})
    try:
        assert out["docs"]["isha.aurobindo"]["units"] == 18
        q = "action cleaves not to a man"
        hits, _ = storage.hybrid_search("__pack_test__", get_embedder().embed([q])[0], q, limit=5)
        labels = [h.metadata.get("citation", "") for h in hits if h.metadata]
        assert any(label.startswith("Isha 2 · Translation") for label in labels), labels
        # re-seeding replaces rather than duplicates
        again = seed("data/packs/vedanta", session_id="__pack_test__", only={"isha.aurobindo"})
        docs = [d for d in storage.list_documents("__pack_test__") if d["file_name"] == "isha.aurobindo.md"]
        assert len(docs) == 1 and docs[0]["chunk_count"] == again["docs"]["isha.aurobindo"]["records"]
    finally:
        storage.purge_session("__pack_test__")


def test_seeding_a_pack_cites_units_and_reseeding_replaces(db, demo_pack):
    """The seed path on the synthetic demo pack, so it is covered without the private texts."""
    from app import storage
    from app.ingestion.embedder import get_embedder
    from scripts.seed_pack import seed

    out = seed(str(demo_pack))
    try:
        assert out["session"] == "__demo_pack__"  # the manifest's seed_session
        assert out["docs"]["demo.shankara.x"]["units"] == 2
        q = "zzdemoprobe"
        hits, _ = storage.hybrid_search("__demo_pack__", get_embedder().embed([q])[0], q, limit=5)
        labels = [h.metadata["citation"] for h in hits if h.metadata]
        assert any(label.startswith("Demo 1 · Translation") for label in labels), labels
        again = seed(str(demo_pack))
        docs = [d for d in storage.list_documents("__demo_pack__") if d["file_name"] == "demo.shankara.x.md"]
        assert len(docs) == 1 and docs[0]["chunk_count"] == again["docs"]["demo.shankara.x"]["records"]
    finally:
        storage.purge_session("__demo_pack__")


def test_uploads_past_the_session_ttl_are_purged_and_everything_else_is_kept(db):
    """A visitor's upload does not outlive its session cookie: once it is older than the TTL it
    is deleted, chunks included. A fresh upload, the global corpus and named (non-cookie)
    sessions, such as a pack seeded locally, are left alone."""
    import uuid

    from app import storage
    from app.ingestion.embedder import get_embedder

    old_sid, new_sid = uuid.uuid4().hex, uuid.uuid4().hex
    emb = get_embedder().embed(["zzttlprobe"])[0]
    storage.add_document(old_sid, "test:ttl:old", "old.txt", 1, [(1, 0, "zzttlprobe old upload", emb)])
    storage.add_document(new_sid, "test:ttl:new", "new.txt", 1, [(1, 0, "zzttlprobe new upload", emb)])
    storage.add_document("__ttl_named__", "test:ttl:named", "named.txt", 1, [(1, 0, "zzttlprobe named session", emb)])
    try:
        with storage._connect() as conn:
            conn.execute(
                "UPDATE documents SET uploaded_at = now() - interval '2 hours' WHERE doc_id IN (%s, %s)",
                ("test:ttl:old", "test:ttl:named"),
            )
        assert storage.purge_expired_uploads(3600) >= 1

        def texts(sid: str) -> set[str]:
            hits, _ = storage.hybrid_search(sid, emb, "zzttlprobe", limit=10)
            return {h.text for h in hits}

        assert "zzttlprobe old upload" not in texts(old_sid)  # the chunks are gone, not just the row
        assert "old.txt" not in {d["file_name"] for d in storage.list_documents(old_sid)}
        assert "zzttlprobe new upload" in texts(new_sid)  # still inside its TTL
        assert "zzttlprobe named session" in texts("__ttl_named__")  # not a cookie session
        assert "sample.txt" in {d["file_name"] for d in storage.list_documents(old_sid)}  # the global corpus
    finally:
        for sid in (old_sid, new_sid, "__ttl_named__"):
            storage.purge_session(sid)
