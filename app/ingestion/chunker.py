"""Word-based chunking with overlap, preserving page numbers.

Ported (simplified) from talk-to-your-data's chunk_text. The visual-grounding
SemanticChunker is intentionally left for v1.1.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.extract import Page


@dataclass
class Chunk:
    chunk_index: int
    page_number: int
    text: str


def chunk_pages(
    pages: list[Page], max_tokens: int = 400, overlap_tokens: int = 50
) -> list[Chunk]:
    """Chunk pages into overlapping word windows, tagging each with its page.

    Tokens are approximated by whitespace-split words. Each chunk records the
    page it started on so answers can cite "p.N".
    """
    chunks: list[Chunk] = []
    index = 0
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        if len(words) <= max_tokens:
            chunks.append(Chunk(index, page.page_number, page.text.strip()))
            index += 1
            continue

        start = 0
        step = max(1, max_tokens - overlap_tokens)
        while start < len(words):
            window = words[start : start + max_tokens]
            chunks.append(Chunk(index, page.page_number, " ".join(window)))
            index += 1
            start += step
    return chunks
