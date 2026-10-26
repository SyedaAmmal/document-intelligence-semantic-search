"""Offline tests for the Streamlit application's document/indexing helpers."""

from io import BytesIO
import unittest

import numpy as np
import pymupdf
from docx import Document

from src.document_intelligence.ingestion import DocumentExtractionError
from src.document_intelligence.models import DocumentChunk
from src.document_intelligence.search import FaissSearchIndex
from src.document_intelligence.ui import (
    add_chunks_to_index,
    process_uploaded_file,
    upload_fingerprint,
)


class FakeChunkEncoder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.encoded_chunks: list[DocumentChunk] = []

    def encode_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
        self.encoded_chunks.extend(chunks)
        return self.embeddings


class UiHelperTests(unittest.TestCase):
    def test_process_uploaded_txt_uses_original_filename_and_chunk_pipeline(self) -> None:
        chunks = process_uploaded_file(
            "meeting-notes.TXT",
            b"Decision: launch on Friday.",
            chunk_size=100,
            overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_filename, "meeting-notes.TXT")
        self.assertEqual(chunks[0].text, "Decision: launch on Friday.")
        self.assertEqual(chunks[0].source_page_numbers, ())

    def test_process_uploaded_pdf_preserves_original_name_and_page_number(self) -> None:
        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "The uploaded PDF contains searchable text.")
        content = pdf.tobytes()
        pdf.close()

        chunks = process_uploaded_file(
            "source-document.pdf",
            content,
            chunk_size=100,
            overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_filename, "source-document.pdf")
        self.assertEqual(chunks[0].source_page_numbers, (1,))

    def test_process_uploaded_docx_preserves_original_filename(self) -> None:
        buffer = BytesIO()
        document = Document()
        document.add_paragraph("DOCX content for the upload pipeline.")
        document.save(buffer)

        chunks = process_uploaded_file(
            "source-document.docx",
            buffer.getvalue(),
            chunk_size=100,
            overlap=10,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_filename, "source-document.docx")
        self.assertEqual(chunks[0].source_page_numbers, ())

    def test_rejects_unsupported_upload_extension(self) -> None:
        with self.assertRaisesRegex(ValueError, "Supported types"):
            process_uploaded_file("records.csv", b"a,b")

    def test_reports_unreadable_upload_without_suppressing_extraction_error(self) -> None:
        with self.assertRaisesRegex(DocumentExtractionError, "broken.pdf"):
            process_uploaded_file("broken.pdf", b"not a PDF")

    def test_upload_fingerprint_changes_with_filename_or_content(self) -> None:
        baseline = upload_fingerprint("note.txt", b"same bytes")

        self.assertEqual(baseline, upload_fingerprint("note.txt", b"same bytes"))
        self.assertNotEqual(baseline, upload_fingerprint("other.txt", b"same bytes"))
        self.assertNotEqual(baseline, upload_fingerprint("note.txt", b"changed"))

    def test_add_chunks_creates_index_with_encoder_dimension(self) -> None:
        chunks = [
            DocumentChunk(
                chunk_id="one",
                text="test",
                source_filename="note.txt",
                source_page_numbers=(),
            )
        ]
        encoder = FakeChunkEncoder(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

        index = add_chunks_to_index(chunks, encoder)

        self.assertIsNotNone(index)
        assert index is not None
        self.assertEqual(index.embedding_dimension, 3)
        self.assertEqual(index.size, 1)
        self.assertEqual(encoder.encoded_chunks, chunks)

    def test_add_chunks_to_existing_index_and_reject_dimension_change(self) -> None:
        existing = FaissSearchIndex(embedding_dimension=2)
        chunks = [
            DocumentChunk(
                chunk_id="one",
                text="test",
                source_filename="note.txt",
                source_page_numbers=(),
            )
        ]

        updated = add_chunks_to_index(
            chunks,
            FakeChunkEncoder(np.array([[0.0, 1.0]], dtype=np.float32)),
            existing,
        )
        self.assertIs(updated, existing)
        self.assertEqual(existing.size, 1)

        with self.assertRaisesRegex(ValueError, "does not match"):
            add_chunks_to_index(
                chunks,
                FakeChunkEncoder(np.array([[0.0, 1.0, 0.0]], dtype=np.float32)),
                existing,
            )

    def test_empty_chunks_do_not_load_or_call_encoder(self) -> None:
        class FailingEncoder:
            def encode_chunks(self, chunks: list[DocumentChunk]) -> np.ndarray:
                raise AssertionError(
                    f"encoder should not be called for {len(chunks)} chunks"
                )

        self.assertIsNone(add_chunks_to_index([], FailingEncoder()))


if __name__ == "__main__":
    unittest.main()
