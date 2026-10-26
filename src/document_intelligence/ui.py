"""Streamlit interface and testable document-indexing helpers."""

import hashlib
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike

from src.document_intelligence.embeddings import EmbeddingEncoder
from src.document_intelligence.ingestion import (
    DocumentExtractionError,
    extract_document_content,
)
from src.document_intelligence.models import DocumentChunk
from src.document_intelligence.processing import chunk_document
from src.document_intelligence.rag import (
    DEFAULT_MIN_SIMILARITY,
    ExtractiveRAG,
    RAGResponse,
)
from src.document_intelligence.search import FaissSearchIndex


SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class ChunkEncoder(Protocol):
    """The chunk-encoding method needed by the indexing helper."""

    def encode_chunks(self, chunks: list[DocumentChunk]) -> ArrayLike: ...


def process_uploaded_file(
    filename: str,
    content: bytes,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Extract and chunk uploaded bytes using the existing document pipeline.

    Bytes are passed to the shared extractors directly, without a temporary
    file or any writes to user-selected locations.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ValueError(
            f"Unsupported file type for '{filename}'. Supported types: {supported}."
        )
    if not isinstance(content, bytes):
        raise TypeError("uploaded content must be bytes")

    try:
        document = extract_document_content(Path(filename).name, content)
        return chunk_document(document, chunk_size=chunk_size, overlap=overlap)
    except DocumentExtractionError as exc:
        raise DocumentExtractionError(f"'{filename}': {exc}") from exc


def upload_fingerprint(filename: str, content: bytes) -> str:
    """Return a stable key for a filename and its uploaded bytes."""
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8"))
    digest.update(b"\0")
    digest.update(content)
    return digest.hexdigest()


def add_chunks_to_index(
    chunks: list[DocumentChunk],
    encoder: ChunkEncoder,
    search_index: FaissSearchIndex | None = None,
) -> FaissSearchIndex | None:
    """Encode and add chunks, creating an index with the encoder's dimension.

    ``None`` is returned when there are no non-empty chunks, without loading
    an embedding model or creating an empty FAISS index.
    """
    if not chunks:
        return search_index

    embeddings = np.asarray(encoder.encode_chunks(chunks), dtype=np.float32)
    if len(embeddings.shape) != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError("The encoder returned an unexpected embedding shape.")
    if embeddings.shape[1] <= 0:
        raise ValueError("The encoder returned embeddings with no dimensions.")

    if search_index is None:
        search_index = FaissSearchIndex(int(embeddings.shape[1]))
    elif search_index.embedding_dimension != embeddings.shape[1]:
        raise ValueError(
            "Embedding dimension does not match the current search index."
        )

    search_index.add_chunks(chunks, embeddings)
    return search_index


def reset_session_state() -> None:
    """Clear all application state and the uploader widget's current value."""
    import streamlit as st

    for key in (
        "document_chunks",
        "embedding_encoder",
        "search_index",
        "rag_workflow",
        "indexed_uploads",
        "last_response",
    ):
        st.session_state.pop(key, None)
    st.session_state.pop("document_uploader", None)


def run_app() -> None:
    """Render the Streamlit upload, indexing, search, and evidence interface."""
    import streamlit as st

    st.set_page_config(
        page_title="Document Intelligence & Semantic Search",
        page_icon="📚",
        layout="wide",
    )
    st.title("Document Intelligence & Semantic Search")
    st.write(
        "Upload PDF, DOCX, or TXT documents, then search their content by meaning."
    )
    st.caption(
        "Answers are extractive evidence quoted from your documents, not generated "
        "or fact-checked answers."
    )

    st.button(
        "Clear session and reset index",
        on_click=reset_session_state,
        help="Clears uploaded-document state and search results. Does not delete files.",
    )

    st.session_state.setdefault("document_chunks", [])
    st.session_state.setdefault("embedding_encoder", EmbeddingEncoder())
    st.session_state.setdefault("search_index", None)
    st.session_state.setdefault("rag_workflow", None)
    st.session_state.setdefault("indexed_uploads", {})
    st.session_state.setdefault("last_response", None)

    uploads = st.file_uploader(
        "Choose documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="document_uploader",
        help="Supported formats: PDF, DOCX, and TXT.",
    )
    chunk_size = st.number_input(
        "Chunk size (characters)",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100,
    )
    overlap = st.number_input(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=max(0, int(chunk_size) - 1),
        value=min(200, max(0, int(chunk_size) - 1)),
        step=50,
    )

    if st.button("Index selected documents", type="primary"):
        if not uploads:
            st.info("Choose one or more supported documents to index.")
        else:
            _index_uploads(st, uploads, int(chunk_size), int(overlap))

    _show_index_status(st)

    st.subheader("Ask a question")
    with st.form("question_form"):
        question = st.text_input("Question about your indexed documents")
        top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=5)
        min_similarity = st.slider(
            "Minimum similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(DEFAULT_MIN_SIMILARITY),
            step=0.05,
            help="Lower-scoring chunks are not shown as answer evidence.",
        )
        submitted = st.form_submit_button("Search documents")

    if submitted:
        if not question.strip():
            st.session_state["last_response"] = None
            st.warning("Enter a question before searching.")
        elif st.session_state["search_index"] is None:
            st.session_state["last_response"] = None
            st.info("Index at least one document before asking a question.")
        else:
            try:
                workflow = ExtractiveRAG(
                    st.session_state["embedding_encoder"],
                    st.session_state["search_index"],
                )
                response = workflow.answer(
                    question,
                    top_k=int(top_k),
                    min_similarity=float(min_similarity),
                )
                st.session_state["rag_workflow"] = workflow
                st.session_state["last_response"] = response
            except Exception as exc:
                st.session_state["last_response"] = None
                st.error(f"Search failed: {exc}")

    response = st.session_state["last_response"]
    if response is not None:
        _show_response(st, response)


def _index_uploads(
    st: Any,
    uploads: list[Any],
    chunk_size: int,
    overlap: int,
) -> None:
    indexed_uploads: dict[str, dict[str, object]] = st.session_state[
        "indexed_uploads"
    ]
    encoder: EmbeddingEncoder = st.session_state["embedding_encoder"]
    current_index: FaissSearchIndex | None = st.session_state["search_index"]
    errors: list[str] = []
    newly_indexed = 0

    with st.spinner("Extracting, chunking, and indexing documents..."):
        for upload in uploads:
            filename = upload.name
            content = upload.getvalue()
            fingerprint = upload_fingerprint(filename, content)
            if fingerprint in indexed_uploads:
                continue

            try:
                chunks = process_uploaded_file(filename, content, chunk_size, overlap)
                updated_index = add_chunks_to_index(chunks, encoder, current_index)
            except (DocumentExtractionError, ValueError, TypeError, OSError) as exc:
                errors.append(str(exc))
                continue
            except Exception as exc:
                errors.append(f"Could not index '{filename}': {exc}")
                continue

            current_index = updated_index
            indexed_uploads[fingerprint] = {
                "filename": filename,
                "chunk_count": len(chunks),
            }
            st.session_state["document_chunks"].extend(chunks)
            newly_indexed += 1

    st.session_state["search_index"] = current_index
    st.session_state["indexed_uploads"] = indexed_uploads
    st.session_state["rag_workflow"] = None
    st.session_state["last_response"] = None

    if newly_indexed:
        st.success(f"Indexed {newly_indexed} new document(s).")
    elif not errors:
        st.info("These documents have already been indexed in this session.")
    for error in errors:
        st.error(error)


def _show_index_status(st: Any) -> None:
    indexed_uploads: dict[str, dict[str, object]] = st.session_state["indexed_uploads"]
    chunks: list[DocumentChunk] = st.session_state["document_chunks"]
    if not indexed_uploads:
        st.info("No documents have been indexed in this session yet.")
        return

    st.subheader("Session index")
    first, second = st.columns(2)
    first.metric("Indexed documents", len(indexed_uploads))
    second.metric("Text chunks", len(chunks))
    with st.expander("Indexed filenames"):
        for item in indexed_uploads.values():
            st.write(f"- {item['filename']} ({item['chunk_count']} chunks)")


def _show_response(st: Any, response: RAGResponse) -> None:
    st.subheader("Response")
    if response.refused:
        st.warning(response.answer)
    else:
        st.success("Extractive evidence found")
        st.markdown(response.answer)
    st.caption(
        "Similarity ranks potentially relevant text; it does not prove that "
        "evidence factually answers the question."
    )

    if response.results:
        st.subheader("Retrieved chunks")
        for rank, result in enumerate(response.results, start=1):
            pages = (
                f" · PDF page(s): {', '.join(map(str, result.source_page_numbers))}"
                if result.source_page_numbers
                else ""
            )
            with st.expander(
                f"{rank}. {result.source_filename}{pages} "
                f"· similarity {result.score:.3f}"
            ):
                st.write(result.chunk.text)

    if response.evidence:
        st.subheader("Cited evidence")
        for evidence in response.evidence:
            st.markdown(
                f"> {evidence.text}\n>\n> **Source:** {evidence.citation} "
                f"· similarity {evidence.similarity_score:.3f}"
            )
