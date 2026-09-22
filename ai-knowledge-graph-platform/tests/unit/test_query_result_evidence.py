"""Pydantic contract tests for `CitationEvidence`/`QueryResult.evidence` —
the structured, additive sibling to `citations: list[str]` (see
docs/IMPLEMENTATION_AUDIT.md, "Structured answer evidence")."""

from __future__ import annotations

from graphrag.core.models import CitationEvidence, QueryResult


def test_citation_evidence_round_trips_through_model_dump():
    original = CitationEvidence(
        source_id="DocA", source_label="Document A", path="[DocA]",
        valid_from="2024-01-02", confidence=0.87,
    )
    reloaded = CitationEvidence(**original.model_dump())
    assert reloaded == original


def test_citation_evidence_defaults_are_none_and_empty():
    minimal = CitationEvidence(source_id="DocA", source_label="DocA")
    assert minimal.path == ""
    assert minimal.valid_from is None
    assert minimal.confidence is None


def test_query_result_evidence_defaults_to_empty_list():
    result = QueryResult(question="q", answer="a", citations=["DocA"])
    assert result.evidence == []
    assert result.citations == ["DocA"]  # untouched sibling field


def test_query_result_evidence_json_round_trips():
    result = QueryResult(
        question="q", answer="a", citations=["DocA"],
        evidence=[CitationEvidence(source_id="DocA", source_label="DocA", confidence=0.5)],
    )
    reloaded = QueryResult.model_validate_json(result.model_dump_json())
    assert reloaded.evidence == result.evidence
    assert reloaded.citations == result.citations
