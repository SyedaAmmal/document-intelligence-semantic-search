"""Offline retrieval evaluation and synthetic benchmark."""

from .evaluator import (
    EvaluationCase,
    EvaluationReport,
    QueryEvaluation,
    evaluate_retrieval,
)

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "QueryEvaluation",
    "evaluate_retrieval",
]
