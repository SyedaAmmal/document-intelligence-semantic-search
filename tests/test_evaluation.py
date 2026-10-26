"""Offline tests for retrieval metrics and benchmark error analysis."""

from pathlib import Path
import tempfile
import unittest

from src.document_intelligence.evaluation import (
    EvaluationCase,
    evaluate_retrieval,
)
from src.document_intelligence.evaluation.benchmark import (
    SYNTHETIC_BENCHMARK,
    TokenOverlapRetriever,
    run_synthetic_benchmark,
    save_benchmark_report,
)
from src.document_intelligence.models import DocumentChunk, SearchResult


def result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            chunk_id=chunk_id,
            text=f"Text for {chunk_id}.",
            source_filename=f"{chunk_id}.txt",
            source_page_numbers=(),
        ),
        score=score,
    )


class ControlledRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])[:top_k]


class EvaluateRetrievalTests(unittest.TestCase):
    def test_calculates_precision_recall_mrr_in_rank_order(self) -> None:
        retriever = ControlledRetriever(
            {
                "query": [
                    result("irrelevant", 0.95),
                    result("relevant-a", 0.8),
                    result("relevant-b", 0.7),
                ]
            }
        )
        cases = [
            EvaluationCase(
                "query",
                frozenset({"relevant-a", "relevant-b", "not-retrieved"}),
            )
        ]

        report = evaluate_retrieval(cases, retriever, k=3)

        self.assertAlmostEqual(report.precision_at_k, 2 / 3)
        self.assertAlmostEqual(report.recall_at_k, 2 / 3)
        self.assertAlmostEqual(report.mean_reciprocal_rank, 1 / 2)
        self.assertEqual(report.evaluations[0].retrieved_chunk_ids,
                         ("irrelevant", "relevant-a", "relevant-b"))
        self.assertEqual(retriever.calls, [("query", 3)])
        self.assertGreaterEqual(report.evaluations[0].latency_ms, 0)

    def test_unfilled_top_k_slots_count_as_precision_misses(self) -> None:
        report = evaluate_retrieval(
            [EvaluationCase("short", frozenset({"found"}))],
            ControlledRetriever({"short": [result("found", 0.9)]}),
            k=4,
        )

        self.assertEqual(report.precision_at_k, 0.25)
        self.assertEqual(report.recall_at_k, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)

    def test_missing_relevant_results_are_reported_as_misses(self) -> None:
        report = evaluate_retrieval(
            [EvaluationCase("missed question", frozenset({"expected"}))],
            ControlledRetriever(
                {"missed question": [result("other", 0.4)]}
            ),
            k=2,
        )

        self.assertEqual(report.missed_relevance_count, 1)
        self.assertEqual(len(report.missed_cases), 1)
        self.assertEqual(report.missed_cases[0].retrieved_chunk_ids, ("other",))
        self.assertEqual(report.missed_cases[0].expected_relevant_ids, ("expected",))

    def test_no_ground_truth_is_undefined_and_excluded_from_metric_averages(self) -> None:
        cases = [
            EvaluationCase("answerable", frozenset({"hit"})),
            EvaluationCase("unanswerable", frozenset()),
        ]
        retriever = ControlledRetriever(
            {
                "answerable": [result("hit", 0.9)],
                "unanswerable": [result("unrelated", 0.2)],
            }
        )

        report = evaluate_retrieval(cases, retriever, k=1)

        self.assertEqual(report.precision_at_k, 1.0)
        self.assertEqual(report.recall_at_k, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)
        self.assertEqual(report.answerable_question_count, 1)
        self.assertEqual(report.no_ground_truth_question_count, 1)
        self.assertIsNone(report.evaluations[1].precision_at_k)
        self.assertIsNone(report.evaluations[1].recall_at_k)
        self.assertIsNone(report.evaluations[1].reciprocal_rank)
        self.assertEqual(len(report.no_evidence_cases), 1)

    def test_report_includes_misses_no_evidence_scores_and_correctness_caveat(self) -> None:
        report = evaluate_retrieval(
            [
                EvaluationCase("unanswered question", frozenset({"expected"})),
                EvaluationCase("no evidence question", frozenset()),
            ],
            ControlledRetriever(
                {
                    "unanswered question": [result("wrong", 0.33)],
                    "no evidence question": [result("candidate", 0.21)],
                }
            ),
            k=2,
        )

        markdown = report.to_markdown()
        self.assertIn("unanswered question", markdown)
        self.assertIn("wrong (0.3300)", markdown)
        self.assertIn("expected", markdown)
        self.assertIn("no evidence question", markdown)
        self.assertIn("candidate (0.2100)", markdown)
        self.assertIn("do not establish factual correctness", markdown)

    def test_synthetic_benchmark_is_28_cases_and_reproducible_in_rank_ids(self) -> None:
        first = run_synthetic_benchmark(k=5)
        second = run_synthetic_benchmark(k=5)

        self.assertEqual(len(SYNTHETIC_BENCHMARK.cases), 28)
        self.assertEqual(first.k, 5)
        self.assertEqual(
            [
                evaluation.retrieved_chunk_ids
                for evaluation in first.evaluations
            ],
            [
                evaluation.retrieved_chunk_ids
                for evaluation in second.evaluations
            ],
        )
        self.assertEqual(first.no_ground_truth_question_count, 4)
        self.assertIn("Synthetic Retrieval Evaluation", first.to_markdown())

    def test_saves_generated_summary_and_error_analysis_to_markdown(self) -> None:
        report = run_synthetic_benchmark(k=5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            report_path = save_benchmark_report(report, temporary_directory)

            self.assertEqual(
                report_path,
                Path(temporary_directory) / "synthetic_retrieval_evaluation.md",
            )
            saved_report = report_path.read_text(encoding="utf-8")

        self.assertIn("# Synthetic Retrieval Evaluation (K=5)", saved_report)
        self.assertIn("- Questions: 28", saved_report)
        self.assertIn("- Answerable questions: 24", saved_report)
        self.assertIn("- No-ground-truth questions: 4", saved_report)
        self.assertIn(
            f"- Precision@5: {report.precision_at_k:.4f}",
            saved_report,
        )
        self.assertIn(f"- Recall@5: {report.recall_at_k:.4f}", saved_report)
        self.assertIn(f"- MRR: {report.mean_reciprocal_rank:.4f}", saved_report)
        self.assertIn(
            f"- Mean retrieval latency: {report.mean_latency_ms:.3f} ms",
            saved_report,
        )
        self.assertIn("Missed relevance", saved_report)
        self.assertIn("late_fee", saved_report)
        self.assertIn("library_hours (", saved_report)
        self.assertIn("No evidence", saved_report)
        self.assertIn("may vary between runs and machines", saved_report)

    def test_invalid_k_and_empty_case_list_are_rejected(self) -> None:
        retriever = ControlledRetriever({})
        with self.assertRaises(ValueError):
            evaluate_retrieval([EvaluationCase("q", frozenset())], retriever, k=0)
        with self.assertRaises(ValueError):
            evaluate_retrieval([], retriever, k=1)


if __name__ == "__main__":
    unittest.main()
