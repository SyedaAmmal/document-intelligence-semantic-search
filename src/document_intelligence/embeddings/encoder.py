"""Lazy, normalized local embeddings using Sentence Transformers."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.document_intelligence.models import DocumentChunk


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EmbeddingMatrix = NDArray[np.float32]


class SentenceEmbeddingModel(Protocol):
    """Minimal model interface, allowing a fake model to be injected in tests."""

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object: ...


class EmbeddingEncoder:
    """Encode documents and queries with one locally loaded model.

    The default MiniLM model is loaded only when an embedding is first
    requested. All output vectors are L2-normalized so inner product can be
    used as cosine similarity by the search index.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: SentenceEmbeddingModel | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    def encode_texts(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Return a float32 matrix with one normalized vector per input text.

        An empty sequence returns a matrix shaped ``(0, 0)`` without loading
        the model. Blank individual text inputs are rejected.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every text to encode must be a string")
        if any(not text.strip() for text in texts):
            raise ValueError("text inputs cannot be empty or whitespace-only")

        raw_embeddings = self._get_model().encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(
                "The embedding model must return a two-dimensional matrix."
            )
        if embeddings.shape[0] != len(texts) or embeddings.shape[1] == 0:
            raise ValueError(
                "The embedding model returned an unexpected embedding shape."
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("The embedding model returned non-finite values.")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("The embedding model returned a zero-length vector.")
        return np.ascontiguousarray(embeddings / norms, dtype=np.float32)

    def encode_chunks(self, chunks: Sequence[DocumentChunk]) -> EmbeddingMatrix:
        """Encode the text of document chunks in their given order."""
        return self.encode_texts([chunk.text for chunk in chunks])

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Encode one non-blank search query using the same model and settings."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty or whitespace-only")
        return self.encode_texts([query])[0]

    def _get_model(self) -> SentenceEmbeddingModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
