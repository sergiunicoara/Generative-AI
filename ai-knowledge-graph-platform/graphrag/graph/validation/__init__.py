"""Deterministic graph validation and the ingestion publication gate.

See docs/graph-validation.md for the rule catalogue, severities and lifecycle.
"""
from graphrag.graph.validation.batch import BatchResult, RejectedRecord, validate_batch, validate_document
from graphrag.graph.validation.gate import PublicationGate, PublicationRejected, StagedBatch
from graphrag.graph.validation.report import ValidationReport, Violation
from graphrag.graph.validation.rules import RULES, PublicationState, Rule, Severity

__all__ = [
    "BatchResult", "PublicationGate", "PublicationRejected", "PublicationState", "RULES",
    "RejectedRecord", "Rule", "Severity", "StagedBatch", "ValidationReport", "Violation",
    "validate_batch", "validate_document",
]
