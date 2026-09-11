"""Unit tests for GraphEvaluator.community_modularity() — the NetworkX-based
independent cross-check on community_coherence().

Only this method is covered here; the rest of GraphEvaluator has no existing
test file and backfilling it is out of scope for this change.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.graph.graph_evaluator import GraphEvaluator


def _make_evaluator(rows: list[dict]) -> GraphEvaluator:
    neo4j = MagicMock()
    neo4j.run = AsyncMock(return_value=rows)
    return GraphEvaluator(neo4j)


class TestCommunityModularity:
    async def test_two_well_separated_communities_score_high(self) -> None:
        """Two dense communities with no cross-community edges should score
        close to the maximum modularity for a 2-community partition."""
        rows = [
            {"source": "a1", "community_id": "c1", "targets": ["a2", "a3"]},
            {"source": "a2", "community_id": "c1", "targets": ["a1", "a3"]},
            {"source": "a3", "community_id": "c1", "targets": ["a1", "a2"]},
            {"source": "b1", "community_id": "c2", "targets": ["b2", "b3"]},
            {"source": "b2", "community_id": "c2", "targets": ["b1", "b3"]},
            {"source": "b3", "community_id": "c2", "targets": ["b1", "b2"]},
        ]
        evaluator = _make_evaluator(rows)
        result = await evaluator.community_modularity(tenant="acme")

        assert result["community_count"] == 2
        assert result["modularity"] > 0.3  # well-separated cliques score well

    async def test_empty_graph_returns_empty_dict(self) -> None:
        evaluator = _make_evaluator([])
        result = await evaluator.community_modularity(tenant="acme")
        assert result == {}

    async def test_single_community_returns_empty_dict(self) -> None:
        """Modularity is undefined for a single partition — must not raise."""
        rows = [
            {"source": "a1", "community_id": "c1", "targets": ["a2"]},
            {"source": "a2", "community_id": "c1", "targets": ["a1"]},
        ]
        evaluator = _make_evaluator(rows)
        result = await evaluator.community_modularity(tenant="acme")
        assert result == {}

    async def test_no_edges_returns_empty_dict(self) -> None:
        """Communities exist but no RELATES_TO edges connect anyone —
        modularity has nothing to measure."""
        rows = [
            {"source": "a1", "community_id": "c1", "targets": []},
            {"source": "b1", "community_id": "c2", "targets": []},
        ]
        evaluator = _make_evaluator(rows)
        result = await evaluator.community_modularity(tenant="acme")
        assert result == {}

    async def test_hierarchical_memberships_are_reduced_to_one_partition(self) -> None:
        """Leiden levels overlap by design; NetworkX modularity must not see them all."""
        rows = [
            {"source": "a1", "community_id": "level-0-a", "community_level": 0, "targets": ["a2"]},
            {"source": "a2", "community_id": "level-0-a", "community_level": 0, "targets": ["a1"]},
            {"source": "b1", "community_id": "level-0-b", "community_level": 0, "targets": ["b2"]},
            {"source": "b2", "community_id": "level-0-b", "community_level": 0, "targets": ["b1"]},
            {"source": "a1", "community_id": "level-1-all", "community_level": 1, "targets": ["a2"]},
            {"source": "b1", "community_id": "level-1-all", "community_level": 1, "targets": ["b2"]},
        ]
        evaluator = _make_evaluator(rows)

        result = await evaluator.community_modularity(tenant="acme")

        assert result["community_count"] == 2
        assert result["modularity"] > 0

    async def test_result_included_in_full_report(self) -> None:
        """full_report() must call community_modularity and thread its
        result through under the 'community_modularity' key.

        The other five metric methods issue a varying number of neo4j.run()
        calls internally (e.g. orphan_growth_rate issues two), so a
        positional side_effect list across the whole call chain is fragile.
        Patch them directly instead — only community_modularity's real
        behavior is under test here."""
        rows = [
            {"source": "a1", "community_id": "c1", "targets": ["a2"]},
            {"source": "a2", "community_id": "c1", "targets": ["a1"]},
            {"source": "b1", "community_id": "c2", "targets": ["b2"]},
            {"source": "b2", "community_id": "c2", "targets": ["b1"]},
        ]
        neo4j = MagicMock()
        neo4j.run = AsyncMock(return_value=rows)
        evaluator = GraphEvaluator(neo4j)

        with (
            patch.object(evaluator, "entity_resolution_quality", AsyncMock(return_value={})),
            patch.object(evaluator, "relation_precision", AsyncMock(return_value={})),
            patch.object(evaluator, "contradiction_rate", AsyncMock(return_value={})),
            patch.object(evaluator, "orphan_growth_rate", AsyncMock(return_value={})),
            patch.object(evaluator, "merge_split_error_proxy", AsyncMock(return_value={})),
            patch.object(evaluator, "community_coherence", AsyncMock(return_value={})),
        ):
            report = await evaluator.full_report(tenant="acme")

        assert "community_modularity" in report
        assert report["community_modularity"]["community_count"] == 2
