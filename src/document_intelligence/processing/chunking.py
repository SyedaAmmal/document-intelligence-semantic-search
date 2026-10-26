"""Character-based chunking with deterministic source metadata."""

import hashlib

from src.document_intelligence.ingestion.extractors import ExtractedDocument
from src.document_intelligence.models import DocumentChunk
from src.document_intelligence.processing.cleaning import clean_text


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Clean and split a document into character-counted overlapping chunks.

    ``chunk_size`` and ``overlap`` are measured in Python string characters.
    Every chunk contains at most ``chunk_size`` characters, and subsequent
    chunk windows overlap by exactly ``overlap`` characters. PDF page
    references identify source pages whose text intersects the chunk; DOCX
    and TXT chunks have no page references.
    """
    _validate_chunk_settings(chunk_size, overlap)

    cleaned_pages = [
        (page.page_number, clean_text(page.text))
        for page in document.pages
    ]
    cleaned_pages = [
        (page_number, text) for page_number, text in cleaned_pages if text
    ]
    if not cleaned_pages:
        return []

    page_spans: list[tuple[int | None, int, int]] = []
    parts: list[str] = []
    offset = 0
    for page_number, text in cleaned_pages:
        if parts:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(text)
        offset += len(text)
        page_spans.append((page_number, start, offset))

    full_text = "".join(parts)
    step = chunk_size - overlap
    chunks: list[DocumentChunk] = []
    for start in range(0, len(full_text), step):
        end = min(start + chunk_size, len(full_text))
        chunk_text = full_text[start:end]
        if not chunk_text.strip():
            continue

        page_numbers = tuple(
            page_number
            for page_number, page_start, page_end in page_spans
            if page_number is not None and start < page_end and end > page_start
        )
        chunk_id = _make_chunk_id(
            document.source_filename,
            full_text,
            chunk_size,
            overlap,
            start,
            end,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_filename=document.source_filename,
                source_page_numbers=page_numbers,
            )
        )
        if end == len(full_text):
            break

    return chunks


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer number of characters")
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer number of characters")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _make_chunk_id(
    filename: str,
    full_text: str,
    chunk_size: int,
    overlap: int,
    start: int,
    end: int,
) -> str:
    stable_input = (
        f"{filename}\0{chunk_size}\0{overlap}\0{start}\0{end}\0{full_text}"
    )
    return hashlib.sha256(stable_input.encode("utf-8")).hexdigest()
