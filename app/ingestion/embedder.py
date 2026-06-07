"""Embedding provider abstraction.

Default: local fastembed (ONNX, open-source, no API key, no rate limit).
Swap to any OpenAI-compatible embedding API (Gemini/OpenAI) via env.
The chosen provider's dimension is asserted against settings.vector_dim at
startup, because the pgvector column dimension is fixed at table creation.
"""

from __future__ import annotations

import time
from typing import Protocol

from app.config import settings


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...


class LocalEmbedder:
    """fastembed ONNX model — runs in-process, no network, no key."""

    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        # bge-small-en-v1.5 = 384; derive empirically so dim is always truthful.
        self._dim = len(next(iter(self._model.embed(["dim probe"]))))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]


class OpenAICompatEmbedder:
    """Any OpenAI-compatible embeddings endpoint (Gemini, OpenAI). Batched + backoff."""

    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._dim = dim
        self._batch = 64

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = texts[i : i + self._batch]
            delay = 1.0
            for attempt in range(3):
                try:
                    resp = self._client.embeddings.create(model=self._model, input=batch)
                    out.extend(item.embedding for item in resp.data)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(delay)
                    delay *= 2
        return out


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Singleton embedder, validated against the configured vector dimension."""
    global _embedder
    if _embedder is not None:
        return _embedder

    if settings.embed_provider == "openai":
        emb: Embedder = OpenAICompatEmbedder(
            api_key=settings.embed_api_key,
            base_url=settings.embed_base_url,
            model=settings.embed_model,
            dim=settings.vector_dim,
        )
    else:
        emb = LocalEmbedder(model=settings.embed_model)

    if emb.dim != settings.vector_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: provider returns {emb.dim} but VECTOR_DIM="
            f"{settings.vector_dim}. The pgvector column is fixed at that dim — "
            f"set VECTOR_DIM={emb.dim} (and recreate the table) or change the model."
        )
    _embedder = emb
    return emb
