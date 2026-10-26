"""In-memory FAISS search over normalized document chunk embeddings."""

from collections.abc import Sequence

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.document_intelligence.models import DocumentChunk, SearchResult


class FaissSearchIndex:
    """Search chunks by cosine similarity using a flat inner-product index.

    Since vectors are L2-normalized before insertion and search, the inner
    product is cosine similarity: larger scores are better, 1.0 is identical
    direction, 0.0 is orthogonal, and -1.0 is opposite direction.
    """

    def __init__(self, embedding_dimension: int) -> None:
        if (
            isinstance(embedding_dimension, bool)
            or not isinstance(embedding_dimension, int)
            or embedding_dimension <= 0
        ):
            raise ValueError("embedding_dimension must be a positive integer")

        self.embedding_dimension = embedding_dimension
        self._index = faiss.IndexFlatIP(embedding_dimension)
        self._chunks: list[DocumentChunk] = []

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: ArrayLike,
    ) -> None:
        """Add chunks and their matching vectors in the same order.

        Embeddings are normalized here as well, so direct index users get the
        same cosine-similarity behavior as callers using ``EmbeddingEncoder``.
        """
        if any(not isinstance(chunk, DocumentChunk) for chunk in chunks):
            raise TypeError("chunks must contain only DocumentChunk instances")

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional matrix")
        if vectors.shape != (len(chunks), self.embedding_dimension):
            raise ValueError(
                "The number of embedding rows must match chunks and each "
                f"embedding must have dimension {self.embedding_dimension}; "
                f"received shape {vectors.shape} for {len(chunks)} chunks."
            )
        if not np.isfinite(vectors).all():
            raise ValueError("embeddings must contain only finite values")
        if not chunks:
            return

        normalized = _normalize_rows(vectors, "embeddings")
        self._index.add(normalized)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: ArrayLike,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return up to ``top_k`` chunks ordered from best cosine score down."""
        _validate_top_k(top_k)
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1 or query.size == 0:
            raise ValueError("query_embedding must be a non-empty vector")
        if query.shape[0] != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension must be {self.embedding_dimension}; "
                f"received {query.shape[0]}."
            )
        if not np.isfinite(query).all():
            raise ValueError("query_embedding must contain only finite values")
        normalized_query = _normalize_rows(query.reshape(1, -1), "query_embedding")

        if self.size == 0:
            return []

        scores, indices = self._index.search(normalized_query, top_k)
        results = []
        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue
            chunk = self._chunks[int(index_position)]
            results.append(SearchResult(chunk=chunk, score=float(score)))
        return results


def _normalize_rows(
    vectors: NDArray[np.float32],
    name: str,
) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} cannot contain zero-length vectors")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
