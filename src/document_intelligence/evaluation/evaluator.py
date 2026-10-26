"""Reproducible ranked-retrieval metrics and error analysis."""

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from src.document_intelligence.models import SearchResult


class Retriever(Protocol):
    """Minimal query retrieval interface used by evaluation."""

    def search(self, query: str, top_k: int) -> list[SearchResult]: ...


@dataclass(frozen=True)
class EvaluationCase:
    """One evaluation question with manually assigned relevant chunk IDs.

    An empty ``relevant_chunk_ids`` means the benchmark has no relevant
    evidence for that question; it is not a positive retrieval example.
    """

    question: str
    relevant_chunk_ids: frozenset[str]


@dataclass(frozen=True)
class QueryEvaluation:
    """Per-question retrieval output and metrics, when ground truth exists."""

    question: str
    expected_relevant_ids: tuple[str, ...]
    retrieved: tuple[SearchResult, ...]
    precision_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float | None
    latency_ms: float

    @property
    def retrieved_chunk_ids(self) -> tuple[str, ...]:
        """Return the retrieved chunk IDs in ranked order."""
        return tuple(result.chunk.chunk_id for result in self.retrieved)


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics, per-query outcomes, and missed/no-evidence cases."""

    k: int
    evaluations: tuple[QueryEvaluation, ...]
    precision_at_k: float | None
    recall_at_k: float | None
    mean_reciprocal_rank: float | None
    mean_latency_ms: float
    answerable_question_count: int
    no_ground_truth_question_count: int
    missed_relevance_count: int

    @property
    def missed_cases(self) -> tuple[QueryEvaluation, ...]:
        """Return answerable cases with no relevant result in the top K."""
        return tuple(
            evaluation
            for evaluation in self.evaluations
            if evaluation.expected_relevant_ids
            and not _has_relevant_result(evaluation)
        )

    @property
    def no_evidence_cases(self) -> tuple[QueryEvaluation, ...]:
        """Return cases whose ground truth declares no relevant evidence."""
        return tuple(
            evaluation
            for evaluation in self.evaluations
            if not evaluation.expected_relevant_ids
        )

    def to_markdown(self) -> str:
        """Format an auditable summary and error-analysis table."""
        lines = [
            f"# Synthetic Retrieval Evaluation (K={self.k})",
            "",
            f"- Questions: {len(self.evaluations)}",
            f"- Answerable questions: {self.answerable_question_count}",
            f"- No-ground-truth questions: {self.no_ground_truth_question_count}",
            f"- Missed-relevance cases: {self.missed_relevance_count}",
            f"- Precision@{self.k}: {_format_metric(self.precision_at_k)}",
            f"- Recall@{self.k}: {_format_metric(self.recall_at_k)}",
            f"- MRR: {_format_metric(self.mean_reciprocal_rank)}",
            f"- Mean retrieval latency: {self.mean_latency_ms:.3f} ms",
            "",
            "Metrics are averaged over answerable questions only. No-ground-truth "
            "questions have undefined per-query precision, recall, and reciprocal "
            "rank and are reported separately, never counted as successful hits.",
            "",
            "## Missed relevance and no-evidence cases",
            "",
            "Similarity scores rank retrieved chunks; they do not establish factual "
            "correctness.",
            "",
            "| Case | Question | Retrieved chunk IDs (scores) | Expected relevant IDs |",
            "|---|---|---|---|",
        ]
        error_cases = (*self.missed_cases, *self.no_evidence_cases)
        if not error_cases:
            lines.append("| None | — | — | — |")
        for case in error_cases:
            category = (
                "No evidence"
                if not case.expected_relevant_ids
                else "Missed relevance"
            )
            retrieved = ", ".join(
                f"{result.chunk.chunk_id} ({result.score:.4f})"
                for result in case.retrieved
            ) or "none"
            expected = ", ".join(case.expected_relevant_ids) or "none (unanswerable)"
            lines.append(
                f"| {category} | {_escape_table(case.question)} | "
                f"{_escape_table(retrieved)} | {_escape_table(expected)} |"
            )
        return "\n".join(lines)


def evaluate_retrieval(
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    retriever: Retriever,
    k: int = 5,
) -> EvaluationReport:
    """Evaluate ranking at K and measure per-query retrieval latency.

    Precision@K is hits divided by K (unfilled result slots count as misses).
    Recall@K is hits divided by the number of known relevant IDs.
    Reciprocal rank is 1 / the rank of the first relevant result, or zero if
    none is retrieved. Aggregate precision, recall, and MRR average answerable
    cases only. For questions with no ground-truth relevant IDs, these metrics
    are undefined and excluded from those averages; they remain in the report.
    Retrieval latency is wall-clock time around ``search`` and averages all
    benchmark questions, including no-evidence cases.
    """
    _validate_k(k)
    if not cases:
        raise ValueError("at least one evaluation case is required")

    evaluations: list[QueryEvaluation] = []
    for case in cases:
        start = perf_counter()
        retrieved = tuple(retriever.search(case.question, top_k=k))
        latency_ms = (perf_counter() - start) * 1000.0
        top_results = retrieved[:k]

        if case.relevant_chunk_ids:
            relevant_ids = case.relevant_chunk_ids
            hit_ids = {
                result.chunk.chunk_id
                for result in top_results
                if result.chunk.chunk_id in relevant_ids
            }
            precision = len(hit_ids) / k
            recall = len(hit_ids) / len(relevant_ids)
            reciprocal_rank = next(
                (
                    1.0 / rank
                    for rank, result in enumerate(top_results, start=1)
                    if result.chunk.chunk_id in relevant_ids
                ),
                0.0,
            )
        else:
            precision = None
            recall = None
            reciprocal_rank = None

        evaluations.append(
            QueryEvaluation(
                question=case.question,
                expected_relevant_ids=tuple(sorted(case.relevant_chunk_ids)),
                retrieved=top_results,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank,
                latency_ms=latency_ms,
            )
        )

    answerable = [
        item for item in evaluations if item.expected_relevant_ids
    ]
    missed_count = sum(
        not _has_relevant_result(item) for item in answerable
    )
    return EvaluationReport(
        k=k,
        evaluations=tuple(evaluations),
        precision_at_k=_mean(
            [item.precision_at_k for item in answerable]
        ),
        recall_at_k=_mean([item.recall_at_k for item in answerable]),
        mean_reciprocal_rank=_mean(
            [item.reciprocal_rank for item in answerable]
        ),
        mean_latency_ms=sum(item.latency_ms for item in evaluations)
        / len(evaluations),
        answerable_question_count=len(answerable),
        no_ground_truth_question_count=len(evaluations) - len(answerable),
        missed_relevance_count=missed_count,
    )


def _has_relevant_result(evaluation: QueryEvaluation) -> bool:
    return any(
        result.chunk.chunk_id in evaluation.expected_relevant_ids
        for result in evaluation.retrieved
    )


def _mean(values: list[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return sum(present_values) / len(present_values)


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be greater than zero")
