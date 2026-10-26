"""Small, deterministic synthetic retrieval benchmark.

This benchmark is a reproducible check of evaluation plumbing, not an
estimate of real-world embedding or semantic-search quality. Its retriever
uses token overlap and should not be substituted for production retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
import re

from src.document_intelligence.evaluation.evaluator import (
    EvaluationCase,
    EvaluationReport,
    evaluate_retrieval,
)
from src.document_intelligence.models import DocumentChunk, SearchResult


@dataclass(frozen=True)
class SyntheticBenchmark:
    """Synthetic corpus and manually labeled questions."""

    chunks: tuple[DocumentChunk, ...]
    cases: tuple[EvaluationCase, ...]


SYNTHETIC_BENCHMARK = SyntheticBenchmark(
    chunks=(
        DocumentChunk(
            "library_hours",
            "The library opens at 9 AM on weekdays and closes at 6 PM.",
            "library_guide.txt",
            (),
        ),
        DocumentChunk(
            "loan_period",
            "Library books may be borrowed for 21 days before they are due.",
            "library_guide.txt",
            (),
        ),
        DocumentChunk(
            "renewals",
            "A library book can be renewed twice if nobody else has requested it.",
            "library_guide.txt",
            (),
        ),
        DocumentChunk(
            "guest_wifi",
            "Visitors can use guest Wi-Fi by selecting the GuestNet network.",
            "visitor_guide.txt",
            (),
        ),
        DocumentChunk(
            "parking",
            "Visitor parking is free after 5 PM in the east lot.",
            "visitor_guide.txt",
            (),
        ),
        DocumentChunk(
            "museum_admission",
            "The city museum is free for children under 12 years old.",
            "museum_faq.txt",
            (),
        ),
        DocumentChunk(
            "museum_monday",
            "The city museum is closed every Monday.",
            "museum_faq.txt",
            (),
        ),
        DocumentChunk(
            "cafe_hours",
            "The campus cafe serves breakfast from 7 AM until 10 AM.",
            "campus_guide.txt",
            (),
        ),
        DocumentChunk(
            "study_room",
            "Study rooms can be reserved online for up to two hours.",
            "campus_guide.txt",
            (),
        ),
        DocumentChunk(
            "late_fee",
            "Overdue library books incur a fine of 25 cents per day.",
            "library_guide.txt",
            (),
        ),
        DocumentChunk(
            "bus_route",
            "Bus route 4 connects the campus with the central train station.",
            "transit_guide.txt",
            (),
        ),
        DocumentChunk(
            "accessible_entry",
            "The museum's north entrance has a step-free accessible ramp.",
            "museum_faq.txt",
            (),
        ),
    ),
    cases=(
        EvaluationCase("When does the library open on weekdays?", frozenset({"library_hours"})),
        EvaluationCase("What time does the library close?", frozenset({"library_hours"})),
        EvaluationCase("How long may I borrow library books?", frozenset({"loan_period"})),
        EvaluationCase("When are borrowed books due back?", frozenset({"loan_period"})),
        EvaluationCase("Can library loans be extended?", frozenset({"renewals"})),
        EvaluationCase("How often can I renew a requested book?", frozenset({"renewals"})),
        EvaluationCase("How do visitors connect to wireless internet?", frozenset({"guest_wifi"})),
        EvaluationCase("Which network provides guest Wi-Fi access?", frozenset({"guest_wifi"})),
        EvaluationCase("Is evening visitor parking free?", frozenset({"parking"})),
        EvaluationCase("Where can guests park after five?", frozenset({"parking"})),
        EvaluationCase("Do children pay to enter the museum?", frozenset({"museum_admission"})),
        EvaluationCase("What is the museum admission price for a ten year old?", frozenset({"museum_admission"})),
        EvaluationCase("Which day is the city museum closed?", frozenset({"museum_monday"})),
        EvaluationCase("Can I visit the museum on Mondays?", frozenset({"museum_monday"})),
        EvaluationCase("What are the campus cafe breakfast hours?", frozenset({"cafe_hours"})),
        EvaluationCase("Until what time is breakfast served at the cafe?", frozenset({"cafe_hours"})),
        EvaluationCase("How long may a student reserve a study room?", frozenset({"study_room"})),
        EvaluationCase("Can study spaces be booked online?", frozenset({"study_room"})),
        EvaluationCase("What is the penalty for a late library book?", frozenset({"late_fee"})),
        EvaluationCase("How much is charged for each overdue day?", frozenset({"late_fee"})),
        EvaluationCase("Which bus travels from campus to the train station?", frozenset({"bus_route"})),
        EvaluationCase("How do I get from the college to the central station?", frozenset({"bus_route"})),
        EvaluationCase("Where is the step-free entrance at the museum?", frozenset({"accessible_entry"})),
        EvaluationCase("Does the museum have an accessible ramp?", frozenset({"accessible_entry"})),
        EvaluationCase("What is the library's phone number?", frozenset()),
        EvaluationCase("Does the cafe offer vegan pastries?", frozenset()),
        EvaluationCase("What is the museum's postal address?", frozenset()),
        EvaluationCase("Can visitors borrow museum artifacts?", frozenset()),
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIRECTORY = PROJECT_ROOT / "evaluation_results"
DEFAULT_REPORT_FILENAME = "synthetic_retrieval_evaluation.md"


class TokenOverlapRetriever:
    """Deterministic lexical retriever used only by this synthetic benchmark."""

    def __init__(self, chunks: tuple[DocumentChunk, ...]) -> None:
        self.chunks = chunks

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in self.chunks:
            chunk_tokens = _tokens(chunk.text)
            shared_terms = query_tokens & chunk_tokens
            if not shared_terms:
                continue
            score = len(shared_terms) / len(query_tokens)
            scored.append(
                (
                    SearchResult(chunk=chunk, score=score),
                    len(shared_terms),
                )
            )

        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[0].chunk.chunk_id,
            )
        )
        return [result for result, _ in scored[:top_k]]


def run_synthetic_benchmark(k: int = 5) -> EvaluationReport:
    """Evaluate the bundled synthetic cases using deterministic token overlap."""
    retriever = TokenOverlapRetriever(SYNTHETIC_BENCHMARK.chunks)
    return evaluate_retrieval(SYNTHETIC_BENCHMARK.cases, retriever, k=k)


def save_benchmark_report(
    report: EvaluationReport,
    output_directory: str | Path = DEFAULT_REPORT_DIRECTORY,
) -> Path:
    """Save this run's generated summary and error analysis as Markdown."""
    report_directory = Path(output_directory)
    report_directory.mkdir(parents=True, exist_ok=True)
    report_path = report_directory / DEFAULT_REPORT_FILENAME
    report_text = report.to_markdown()
    report_text += (
        "\n\nRetrieval latency is a wall-clock measurement for this run and may "
        "vary between runs and machines."
    )
    report_path.write_text(report_text + "\n", encoding="utf-8")
    return report_path


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", text)
        if len(token) > 1
    }


if __name__ == "__main__":
    benchmark_report = run_synthetic_benchmark()
    print(benchmark_report.to_markdown())
    report_path = save_benchmark_report(benchmark_report)
    print(f"\nSaved Markdown report: {report_path}")
