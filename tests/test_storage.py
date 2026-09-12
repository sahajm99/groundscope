"""Hybrid retrieval against the local pgvector: the keyword leg must contribute."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


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
