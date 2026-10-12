"""Offline graph-correctness report: pure arithmetic and read-only guarantees."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph import correctness
from graphrag.graph.validation.graph_checks import assert_read_only


def test_both_queries_are_read_only():
    assert_read_only(correctness.EDGE_COUNTS)
    assert_read_only(correctness.SUPERSEDED_SOURCE_EDGES)


def test_queries_are_tenant_scoped_and_use_the_shared_validity_predicate():
    for q in (correctness.EDGE_COUNTS, correctness.SUPERSEDED_SOURCE_EDGES):
        assert "tenant: $tenant" in q and "'RETRACTED'" in q and "$as_of" in q


def test_summary_arithmetic():
    out = correctness.summarize({
        "edges": 10, "current_edges": 8, "inverted_validity": 1, "current_edges_on_quarantined": 2,
        "inferred_edges": 4, "inferred_without_premises": 1, "document_edges_without_source": 3,
    }, superseded_source_edges=2)
    m, v = out["measures"], out["violations"]
    assert m == {"current_edge_share": 0.8, "temporal_consistency": 0.9,
                 "current_edges_from_current_documents": 0.75, "inference_traceability": 0.75,
                 "provenance_coverage": 0.7}
    assert v == {"inverted_validity": 1, "superseded_source_edges": 2,
                 "current_edges_on_quarantined_entities": 2, "inferred_without_premises": 1,
                 "document_edges_without_source": 3}
    assert out["edges"] == 10 and out["current_edges"] == 8


def test_empty_graph_reports_none_not_a_fake_perfect_score():
    out = correctness.summarize({})
    assert out["edges"] == 0 and set(out["measures"].values()) == {None}
    assert all(value == 0 for value in out["violations"].values())


def test_no_inferred_edges_means_traceability_is_not_applicable():
    out = correctness.summarize({"edges": 5, "current_edges": 5})
    assert out["measures"]["inference_traceability"] is None
    assert out["measures"]["current_edge_share"] == 1.0


@pytest.mark.asyncio
async def test_report_uses_the_read_session_and_passes_as_of():
    neo = MagicMock()
    neo.run_read = AsyncMock(side_effect=[[{"edges": 2, "current_edges": 2}], [{"superseded_source_edges": 1}]])
    neo.run = AsyncMock(side_effect=AssertionError("must not use the write session"))
    report = await correctness.graph_correctness_report(neo, "acme")
    assert report["tenant"] == "acme" and report["violations"]["superseded_source_edges"] == 1
    assert neo.run_read.await_count == 2
    for call in neo.run_read.await_args_list:
        assert call.kwargs == {"tenant": "acme", "as_of": None}


@pytest.mark.asyncio
async def test_missing_rows_do_not_crash():
    neo = MagicMock()
    neo.run_read = AsyncMock(return_value=[])
    report = await correctness.graph_correctness_report(neo, "acme")
    assert report["edges"] == 0
