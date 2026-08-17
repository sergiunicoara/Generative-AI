"""Unit tests for DocumentAuthorityService.apply_authority_weights.

Focused on the `as_edge_weight` opt-in added 2026-08-17: default behavior
(False) must fold the authority multiplier into edge["confidence"], exactly
as before. When True, the multiplier is written to a separate
edge["authority_weight"] field and edge["confidence"] must be left
untouched -- see the docstring on apply_authority_weights, and
tests/unit/test_gnn_scorer.py's TestGNNScorerAuthorityAsEdgeWeight for the
consuming side (GNNScorer) of this same change.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.document_authority import DocumentAuthorityService

pytestmark = pytest.mark.asyncio


def _neo4j_with_doc(level: int, superseded: bool) -> AsyncMock:
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"id": "d1", "level": level, "superseded": superseded}])
    return neo4j


class TestApplyAuthorityWeightsDefaultBehavior:
    """as_edge_weight defaults to False -- must match pre-2026-08-17 behavior
    exactly: multiplier folded into confidence, no authority_weight field."""

    async def test_regulatory_non_superseded_confidence_unchanged(self):
        svc = DocumentAuthorityService(_neo4j_with_doc(level=1, superseded=False))
        edges = [{"source_doc_id": "d1", "confidence": 0.8}]
        result = await svc.apply_authority_weights("acme", edges)
        assert result[0]["confidence"] == pytest.approx(0.8 * 1.0)
        assert "authority_weight" not in result[0]

    async def test_superseded_applies_penalty_to_confidence(self):
        svc = DocumentAuthorityService(_neo4j_with_doc(level=1, superseded=True))
        edges = [{"source_doc_id": "d1", "confidence": 1.0}]
        result = await svc.apply_authority_weights("acme", edges)
        assert result[0]["confidence"] == pytest.approx(1.0 * 1.0 * 0.5)
        assert "authority_weight" not in result[0]

    async def test_informal_level_applies_070_multiplier(self):
        svc = DocumentAuthorityService(_neo4j_with_doc(level=4, superseded=False))
        edges = [{"source_doc_id": "d1", "confidence": 1.0}]
        result = await svc.apply_authority_weights("acme", edges)
        assert result[0]["confidence"] == pytest.approx(0.70)


class TestApplyAuthorityWeightsAsEdgeWeight:
    """as_edge_weight=True: authority_weight is a separate field, confidence
    is left completely untouched."""

    async def test_confidence_untouched_when_as_edge_weight(self):
        svc = DocumentAuthorityService(_neo4j_with_doc(level=1, superseded=False))
        edges = [{"source_doc_id": "d1", "confidence": 0.8}]
        result = await svc.apply_authority_weights("acme", edges, as_edge_weight=True)
        assert result[0]["confidence"] == 0.8   # unchanged, not re-multiplied by 1.0
        assert result[0]["authority_weight"] == pytest.approx(1.0)

    async def test_superseded_writes_authority_weight_not_confidence(self):
        """This is the whole point of the change: a superseded source's
        penalty must NOT be able to push confidence below
        gnn_edge_confidence_threshold (default 0.7) and get the edge
        silently dropped -- it should down-weight, not delete."""
        svc = DocumentAuthorityService(_neo4j_with_doc(level=1, superseded=True))
        edges = [{"source_doc_id": "d1", "confidence": 0.95}]
        result = await svc.apply_authority_weights("acme", edges, as_edge_weight=True)
        assert result[0]["confidence"] == 0.95   # untouched -- still clears the 0.7 threshold
        assert result[0]["authority_weight"] == pytest.approx(1.0 * 0.5)   # down-weighted instead

    async def test_manufacturer_spec_level_writes_095_authority_weight(self):
        svc = DocumentAuthorityService(_neo4j_with_doc(level=2, superseded=False))
        edges = [{"source_doc_id": "d1", "confidence": 1.0}]
        result = await svc.apply_authority_weights("acme", edges, as_edge_weight=True)
        assert result[0]["authority_weight"] == pytest.approx(0.95)
        assert result[0]["confidence"] == 1.0
