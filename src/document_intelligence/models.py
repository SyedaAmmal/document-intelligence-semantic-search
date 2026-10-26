"""Shared data structures used by document processing and retrieval."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk with its original source and source PDF page references.

    ``source_page_numbers`` contains one-based PDF page numbers. It is empty
    for formats without reliable page boundaries, such as DOCX and TXT.
    """

    chunk_id: str
    text: str
    source_filename: str
    source_page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class SearchResult:
    """One ranked search match with its cosine-similarity score."""

    chunk: DocumentChunk
    score: float

    @property
    def source_filename(self) -> str:
        """Return the matched chunk's original filename."""
        return self.chunk.source_filename

    @property
    def source_page_numbers(self) -> tuple[int, ...]:
        """Return the matched chunk's known source page references."""
        return self.chunk.source_page_numbers
