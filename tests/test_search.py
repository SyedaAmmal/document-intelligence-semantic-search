"""Offline tests for FAISS indexing, ranking, and source metadata."""

import unittest

import numpy as np

from src.document_intelligence.models import DocumentChunk
from src.document_intelligence.search import FaissSearchIndex


def make_chunk(
    chunk_id: str,
    filename: str,
    pages: tuple[int, ...] = (),
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        source_filename=filename,
        source_page_numbers=pages,
    )


class FaissSearchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = FaissSearchIndex(embedding_dimension=2)

    def test_adds_chunks_and_reports_index_size(self) -> None:
        chunks = [make_chunk("one", "one.pdf", (1,))]

        self.index.add_chunks(chunks, np.array([[2.0, 0.0]], dtype=np.float32))

        self.assertEqual(self.index.size, 1)

    def test_ranks_known_vectors_by_cosine_similarity(self) -> None:
        chunks = [
            make_chunk("aligned", "aligned.pdf", (1,)),
            make_chunk("diagonal", "diagonal.docx"),
            make_chunk("opposite", "opposite.txt"),
        ]
        vectors = np.array(
            [[4.0, 0.0], [1.0, 1.0], [-3.0, 0.0]],
            dtype=np.float32,
        )
        self.index.add_chunks(chunks, vectors)

        results = self.index.search(np.array([1.0, 0.0]), top_k=3)

        self.assertEqual(
            [result.chunk.chunk_id for result in results],
            ["aligned", "diagonal", "opposite"],
        )
        np.testing.assert_allclose(
            [result.score for result in results],
            [1.0, np.sqrt(0.5), -1.0],
            atol=1e-6,
        )

    def test_top_k_limits_results_and_never_returns_missing_slots(self) -> None:
        self.index.add_chunks(
            [make_chunk("one", "one.txt"), make_chunk("two", "two.txt")],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

        self.assertEqual(len(self.index.search([1.0, 0.0], top_k=1)), 1)
        self.assertEqual(len(self.index.search([1.0, 0.0], top_k=10)), 2)

    def test_empty_index_returns_no_results(self) -> None:
        self.assertEqual(self.index.search([1.0, 0.0]), [])

    def test_rejects_invalid_top_k_and_empty_or_mismatched_queries(self) -> None:
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    self.index.search([1.0, 0.0], top_k=top_k)
        with self.assertRaises(TypeError):
            self.index.search([1.0, 0.0], top_k=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.index.search([])
        with self.assertRaises(ValueError):
            self.index.search([1.0, 0.0, 0.0])

    def test_rejects_mismatched_embeddings_and_zero_vectors(self) -> None:
        chunk = make_chunk("one", "one.pdf")
        with self.assertRaises(ValueError):
            self.index.add_chunks([chunk], np.array([[1.0, 0.0, 0.0]]))
        with self.assertRaises(ValueError):
            self.index.add_chunks([chunk], np.array([[1.0, 0.0], [0.0, 1.0]]))
        with self.assertRaises(ValueError):
            self.index.add_chunks([chunk], np.array([[0.0, 0.0]]))
        self.assertEqual(self.index.size, 0)

    def test_search_result_maps_metadata_to_the_retrieved_chunk(self) -> None:
        first = make_chunk("first", "first.pdf", (2, 3))
        second = make_chunk("second", "second.txt")
        self.index.add_chunks(
            [first, second],
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )

        result = self.index.search([0.0, 1.0])[0]

        self.assertIs(result.chunk, first)
        self.assertEqual(result.source_filename, "first.pdf")
        self.assertEqual(result.source_page_numbers, (2, 3))
        self.assertAlmostEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
