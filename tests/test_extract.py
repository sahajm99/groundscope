"""Text extraction: a plain-text file can carry pages, so a seeded book cites by chapter."""

from __future__ import annotations

from app.ingestion.extract import extract_pages


def test_form_feeds_split_a_text_file_into_pages():
    """The demo corpus is one .txt with the Gita's 18 chapters separated by form feeds, so a
    citation 'bhagavad-gita.txt p.2' means chapter 2 rather than every chunk saying p.1."""
    data = b"Chapter one text.\fChapter two text.\f\fChapter three text.\n"
    pages = extract_pages(data, "book.txt")
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert [p.text.strip() for p in pages] == ["Chapter one text.", "Chapter two text.", "Chapter three text."]


def test_text_file_without_form_feeds_is_one_page():
    pages = extract_pages(b"just a note\n", "note.txt")
    assert len(pages) == 1 and pages[0].page_number == 1
