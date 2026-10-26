"""Tests for conservative text cleaning and source-aware chunking."""

import unittest

from src.document_intelligence.ingestion import ExtractedDocument, ExtractedPage
from src.document_intelligence.processing import chunk_document, clean_text


class CleanTextTests(unittest.TestCase):
    def test_normalizes_line_endings_and_excess_whitespace(self) -> None:
        raw = "  First   line. \r\n \t\r\n\r\n  Second\t line!   \r"

        self.assertEqual(clean_text(raw), "First line.\n\nSecond line!")

    def test_preserves_normal_text_paragraphs_and_punctuation(self) -> None:
        text = "A sentence, with punctuation!\n\nA second paragraph?"

        self.assertEqual(clean_text(text), text)

    def test_rejects_non_string_input(self) -> None:
        with self.assertRaises(TypeError):
            clean_text(None)  # type: ignore[arg-type]


class ChunkDocumentTests(unittest.TestCase):
    def test_splits_long_text_with_character_overlap(self) -> None:
        document = ExtractedDocument(
            source_filename="notes.txt",
            pages=(ExtractedPage(page_number=None, text="abcdefghijklmnopqrst"),),
        )

        chunks = chunk_document(document, chunk_size=10, overlap=3)

        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["abcdefghij", "hijklmnopq", "opqrst"],
        )
        self.assertEqual(chunks[0].text[-3:], chunks[1].text[:3])
        self.assertEqual(chunks[1].text[-3:], chunks[2].text[:3])
        self.assertTrue(all(len(chunk.text) <= 10 for chunk in chunks))

    def test_rejects_invalid_chunk_sizes_and_overlaps(self) -> None:
        document = ExtractedDocument(
            source_filename="notes.txt",
            pages=(ExtractedPage(page_number=None, text="text"),),
        )

        invalid_settings = (
            (0, 0),
            (-1, 0),
            (5, -1),
            (5, 5),
        )
        for chunk_size, overlap in invalid_settings:
            with self.subTest(chunk_size=chunk_size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_document(document, chunk_size, overlap)

        with self.assertRaises(TypeError):
            chunk_document(document, 2.5, 0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            chunk_document(document, 5, 1.5)  # type: ignore[arg-type]

    def test_empty_and_whitespace_only_documents_produce_no_chunks(self) -> None:
        documents = (
            ExtractedDocument(source_filename="empty.txt", pages=()),
            ExtractedDocument(
                source_filename="blank.pdf",
                pages=(
                    ExtractedPage(page_number=1, text=" \t\n "),
                    ExtractedPage(page_number=2, text=""),
                ),
            ),
        )

        for document in documents:
            with self.subTest(filename=document.source_filename):
                self.assertEqual(chunk_document(document, 10, 2), [])

    def test_preserves_filename_and_pdf_page_references(self) -> None:
        document = ExtractedDocument(
            source_filename="manual.pdf",
            pages=(
                ExtractedPage(page_number=1, text="abcdefghij"),
                ExtractedPage(page_number=2, text="klmnopqrst"),
            ),
        )

        chunks = chunk_document(document, chunk_size=12, overlap=2)

        self.assertEqual(chunks[0].source_filename, "manual.pdf")
        self.assertEqual(chunks[0].source_page_numbers, (1, 2))
        self.assertEqual(chunks[1].source_page_numbers, (2,))

    def test_docx_or_txt_chunks_do_not_get_page_numbers(self) -> None:
        document = ExtractedDocument(
            source_filename="report.docx",
            pages=(ExtractedPage(page_number=None, text="Some content."),),
        )

        chunks = chunk_document(document, chunk_size=20, overlap=0)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_filename, "report.docx")
        self.assertEqual(chunks[0].source_page_numbers, ())

    def test_chunk_ids_are_deterministic_for_unchanged_input_and_settings(self) -> None:
        document = ExtractedDocument(
            source_filename="notes.txt",
            pages=(ExtractedPage(page_number=None, text="A longer sample text."),),
        )

        first = chunk_document(document, chunk_size=8, overlap=2)
        second = chunk_document(document, chunk_size=8, overlap=2)

        self.assertEqual(
            [chunk.chunk_id for chunk in first],
            [chunk.chunk_id for chunk in second],
        )
        self.assertEqual(len({chunk.chunk_id for chunk in first}), len(first))


if __name__ == "__main__":
    unittest.main()
