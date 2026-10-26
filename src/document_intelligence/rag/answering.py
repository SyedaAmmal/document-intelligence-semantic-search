"""Extractive, source-cited answers built only from retrieved document text."""

from dataclasses import dataclass
import re
from typing import Protocol

from numpy.typing import NDArray

from src.document_intelligence.models import SearchResult


DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.3
REFUSAL_MESSAGE = (
    "The available documents do not contain enough relevant information "
    "to answer confidently."
)
EXTRACTIVE_NOTICE = (
    "Extractive evidence from retrieved documents (not a generated answer):"
)


class QueryEncoder(Protocol):
    """The query-encoding operation required by the retrieval workflow."""

    def encode_query(self, query: str) -> NDArray: ...


class ChunkSearchIndex(Protocol):
    """The search operation required by the retrieval workflow."""

    def search(
        self,
        query_embedding: NDArray,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class AnswerEvidence:
    """An extractive passage with the citation and score of its source chunk."""

    text: str
    source_filename: str
    source_page_numbers: tuple[int, ...]
    similarity_score: float

    @property
    def citation(self) -> str:
        """Format the source filename and, when known, PDF page references."""
        if self.source_page_numbers:
            pages = ", ".join(str(page) for page in self.source_page_numbers)
            return f"{self.source_filename}, page(s) {pages}"
        return self.source_filename


@dataclass(frozen=True)
class RAGResponse:
    """Question, retrieved matches, and source-grounded extractive response."""

    question: str
    results: tuple[SearchResult, ...]
    evidence: tuple[AnswerEvidence, ...]
    answer: str
    refused: bool

    @property
    def source_filenames(self) -> tuple[str, ...]:
        """Return unique filenames from retrieved matches in rank order."""
        return tuple(dict.fromkeys(result.source_filename for result in self.results))

    @property
    def source_page_references(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        """Return filename and available PDF page references from matches."""
        return tuple(
            dict.fromkeys(
                (result.source_filename, result.source_page_numbers)
                for result in self.results
            )
        )


class ExtractiveRAG:
    """Retrieve relevant chunks and quote sentences as cited evidence.

    The workflow does not generate or paraphrase an answer. It returns sentence
    excerpts from retrieved chunks and refuses when no excerpt meets the
    configured cosine-similarity threshold.
    """

    def __init__(
        self,
        encoder: QueryEncoder,
        search_index: ChunkSearchIndex,
    ) -> None:
        self.encoder = encoder
        self.search_index = search_index

    def answer(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> RAGResponse:
        """Retrieve evidence for a question or return a clear refusal.

        ``top_k`` controls the number of chunks retrieved. ``min_similarity``
        is a cosine-similarity cutoff in the inclusive range [-1.0, 1.0];
        chunks scoring below it are not presented as evidence.
        """
        _validate_question(question)
        _validate_top_k(top_k)
        _validate_threshold(min_similarity)

        query_embedding = self.encoder.encode_query(question)
        results = tuple(self.search_index.search(query_embedding, top_k=top_k))
        evidence = tuple(
            AnswerEvidence(
                text=sentence,
                source_filename=result.source_filename,
                source_page_numbers=result.source_page_numbers,
                similarity_score=result.score,
            )
            for result in results
            if result.score >= min_similarity
            for sentence in _extract_passages(result.chunk.text)
        )

        if not evidence:
            return RAGResponse(
                question=question,
                results=results,
                evidence=(),
                answer=REFUSAL_MESSAGE,
                refused=True,
            )

        cited_evidence = "\n".join(
            f"- {item.text} [{item.citation}]" for item in evidence
        )
        return RAGResponse(
            question=question,
            results=results,
            evidence=evidence,
            answer=f"{EXTRACTIVE_NOTICE}\n{cited_evidence}",
            refused=False,
        )


def _extract_passages(text: str) -> tuple[str, ...]:
    """Split chunk text into non-empty sentence-like passages without rewriting."""
    return tuple(
        passage
        for passage in (part.strip() for part in re.split(r"(?<=[.!?])\s+", text))
        if passage
    )


def _validate_question(question: str) -> None:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    if not question.strip():
        raise ValueError("question cannot be empty or whitespace-only")


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")


def _validate_threshold(min_similarity: float) -> None:
    if isinstance(min_similarity, bool) or not isinstance(
        min_similarity, (int, float)
    ):
        raise TypeError("min_similarity must be a number")
    if not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be between -1.0 and 1.0")
