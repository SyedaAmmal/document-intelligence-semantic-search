"""Offline tests for source-grounded extractive retrieval."""

import unittest

import numpy as np

from src.document_intelligence.models import DocumentChunk, SearchResult
from src.document_intelligence.rag import ExtractiveRAG, REFUSAL_MESSAGE


class FakeEncoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def encode_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeSearchIndex:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[np.ndarray, int]] = []

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((query_embedding, top_k))
        return self.results[:top_k]


def make_result(
    text: str,
    filename: str,
    score: float,
    pages: tuple[int, ...] = (),
    chunk_id: str = "chunk",
) -> SearchResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source_filename=filename,
        source_page_numbers=pages,
    )
    return SearchResult(chunk=chunk, score=score)


class ExtractiveRAGTests(unittest.TestCase):
    def make_workflow(
        self,
        results: list[SearchResult],
    ) -> tuple[ExtractiveRAG, FakeEncoder, FakeSearchIndex]:
        encoder = FakeEncoder()
        index = FakeSearchIndex(results)
        return ExtractiveRAG(encoder, index), encoder, index

    def test_retrieves_results_and_extracts_only_document_passages(self) -> None:
        retrieved = make_result(
            "The library closes at 6 PM. It opens at 9 AM.",
            "hours.pdf",
            0.82,
            pages=(2,),
        )
        workflow, encoder, index = self.make_workflow([retrieved])

        response = workflow.answer("When does the library close?")

        self.assertEqual(response.question, "When does the library close?")
        self.assertEqual(response.results, (retrieved,))
        self.assertEqual(
            [item.text for item in response.evidence],
            ["The library closes at 6 PM.", "It opens at 9 AM."],
        )
        self.assertIn("Extractive evidence", response.answer)
        self.assertIn("not a generated answer", response.answer.lower())
        self.assertIn("The library closes at 6 PM.", response.answer)
        self.assertFalse(response.refused)
        self.assertEqual(encoder.queries, ["When does the library close?"])
        self.assertEqual(index.calls[0][1], 5)

    def test_preserves_filename_and_pdf_page_citations_for_each_evidence(self) -> None:
        workflow, _, _ = self.make_workflow(
            [
                make_result(
                    "The policy starts today.",
                    "policy.pdf",
                    0.9,
                    pages=(3, 4),
                )
            ]
        )

        response = workflow.answer("When does the policy start?")

        evidence = response.evidence[0]
        self.assertEqual(evidence.source_filename, "policy.pdf")
        self.assertEqual(evidence.source_page_numbers, (3, 4))
        self.assertEqual(evidence.citation, "policy.pdf, page(s) 3, 4")
        self.assertIn("[policy.pdf, page(s) 3, 4]", response.answer)
        self.assertEqual(response.source_filenames, ("policy.pdf",))
        self.assertEqual(
            response.source_page_references,
            (("policy.pdf", (3, 4)),),
        )

    def test_blank_question_is_rejected_before_encoding(self) -> None:
        workflow, encoder, _ = self.make_workflow([])

        with self.assertRaisesRegex(ValueError, "question cannot be empty"):
            workflow.answer(" \t\n")

        self.assertEqual(encoder.queries, [])

    def test_empty_index_returns_clear_refusal(self) -> None:
        workflow, _, _ = self.make_workflow([])

        response = workflow.answer("What is documented?")

        self.assertEqual(response.results, ())
        self.assertEqual(response.evidence, ())
        self.assertEqual(response.answer, REFUSAL_MESSAGE)
        self.assertTrue(response.refused)

    def test_low_similarity_results_are_not_used_as_evidence(self) -> None:
        result = make_result("The answer might be here.", "low.txt", 0.2)
        workflow, _, _ = self.make_workflow([result])

        response = workflow.answer("Question", min_similarity=0.5)

        self.assertEqual(response.results, (result,))
        self.assertEqual(response.evidence, ())
        self.assertEqual(response.answer, REFUSAL_MESSAGE)
        self.assertTrue(response.refused)

    def test_no_useful_text_does_not_create_unsupported_answer(self) -> None:
        workflow, _, _ = self.make_workflow(
            [make_result(" \t\n ", "blank.pdf", 0.95, pages=(1,))]
        )

        response = workflow.answer("Question")

        self.assertEqual(response.evidence, ())
        self.assertEqual(response.answer, REFUSAL_MESSAGE)
        self.assertTrue(response.refused)

    def test_top_k_and_threshold_are_configurable(self) -> None:
        results = [
            make_result("First evidence.", "first.txt", 0.7, chunk_id="first"),
            make_result("Second evidence.", "second.txt", 0.4, chunk_id="second"),
        ]
        workflow, _, index = self.make_workflow(results)

        response = workflow.answer("Question", top_k=1, min_similarity=0.6)

        self.assertEqual(index.calls[0][1], 1)
        self.assertEqual(len(response.results), 1)
        self.assertEqual([item.text for item in response.evidence], ["First evidence."])

        second_workflow, _, _ = self.make_workflow(results)
        response_with_lower_threshold = second_workflow.answer(
            "Question",
            top_k=2,
            min_similarity=0.3,
        )
        self.assertEqual(len(response_with_lower_threshold.evidence), 2)

    def test_invalid_top_k_and_threshold_are_rejected(self) -> None:
        workflow, _, _ = self.make_workflow([])

        with self.assertRaises(ValueError):
            workflow.answer("Question", top_k=0)
        with self.assertRaises(ValueError):
            workflow.answer("Question", min_similarity=1.1)


if __name__ == "__main__":
    unittest.main()
