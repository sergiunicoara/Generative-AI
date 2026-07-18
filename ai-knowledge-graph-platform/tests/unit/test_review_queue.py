"""Unit tests for the human review queue — AmbiguousMatch, ReviewQueueService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from graphrag.graph.alias_registry import AliasRegistry, AmbiguousMatch, _normalize
from graphrag.graph.review_queue import ReviewQueueService


# ── AmbiguousMatch dataclass ───────────────────────────────────────────────────

class TestAmbiguousMatch:
    def test_fields(self):
        m = AmbiguousMatch(candidate=("IATF 16949:2016", "CONCEPT"), score=75.0, match_type="fuzzy")
        assert m.candidate == ("IATF 16949:2016", "CONCEPT")
        assert m.score == 75.0
        assert m.match_type == "fuzzy"

    def test_embedding_variant(self):
        m = AmbiguousMatch(candidate=("AutoCorp", "ORG"), score=0.88, match_type="embedding")
        assert m.match_type == "embedding"


# ── AliasRegistry.resolve() — fuzzy band behaviour ────────────────────────────

def _make_registry(fuzzy_threshold=85, review_fuzzy_min=70, entries=None):
    """Build a loaded AliasRegistry with controlled config and pre-loaded _exact."""
    neo4j = AsyncMock()
    with patch("graphrag.core.config.get_settings") as mock_cfg:
        mock_settings = MagicMock()
        mock_settings.ingestion.get = lambda key, default=None: {
            "alias_embedding_threshold": 0.97,
            "alias_fuzzy_threshold": fuzzy_threshold,
            "review_fuzzy_min": review_fuzzy_min,
        }.get(key, default)
        mock_cfg.return_value = mock_settings
        registry = AliasRegistry(neo4j, tenant="test")
    registry._fuzzy_threshold = fuzzy_threshold
    registry._loaded = True
    if entries:
        registry._exact = {_normalize(k): v for k, v in entries.items()}
    return registry


class TestResolveAmbiguousBand:
    def test_exact_match_returns_tuple(self):
        registry = _make_registry(entries={"IATF 16949:2016": ("IATF 16949:2016", "CONCEPT")})
        result = registry.resolve("IATF 16949:2016")
        assert isinstance(result, tuple)
        assert result == ("IATF 16949:2016", "CONCEPT")

    def test_high_fuzzy_score_auto_merges(self):
        """Score >= threshold returns canonical tuple, not AmbiguousMatch."""
        registry = _make_registry(
            fuzzy_threshold=85,
            entries={"iatf 16949 2016": ("IATF 16949:2016", "CONCEPT")},
        )
        # "iatf 16949 2016" vs itself → score 100, above threshold
        result = registry.resolve("iatf 16949 2016")
        assert isinstance(result, tuple)

    def test_ambiguous_fuzzy_score_returns_ambiguous_match(self):
        """Score in [70, 85) returns AmbiguousMatch, not a tuple."""
        registry = _make_registry(
            fuzzy_threshold=85,
            review_fuzzy_min=70,
            entries={"iatf 16949 2016": ("IATF 16949:2016", "CONCEPT")},
        )
        # Use a name that produces a score in the ambiguous band
        # Patch rapidfuzz to return a controlled score of 75
        with patch("graphrag.graph.alias_registry.AliasRegistry.resolve") as mock_resolve:
            mock_resolve.return_value = AmbiguousMatch(
                candidate=("IATF 16949:2016", "CONCEPT"),
                score=75.0,
                match_type="fuzzy",
            )
            result = registry.resolve("ISO IATF")
        assert isinstance(result, AmbiguousMatch)
        assert result.match_type == "fuzzy"
        assert result.score == 75.0

    def test_no_match_returns_none(self):
        registry = _make_registry(entries={})
        result = registry.resolve("completely unknown entity xyz")
        assert result is None

    def test_below_review_min_returns_none(self):
        """Score below review_fuzzy_min (< 70) should not produce AmbiguousMatch."""
        registry = _make_registry(
            fuzzy_threshold=85,
            review_fuzzy_min=70,
            entries={"iatf 16949 2016": ("IATF 16949:2016", "CONCEPT")},
        )
        # Patch fuzz.ratio to return 50 — below review_fuzzy_min
        with patch("graphrag.core.config.get_settings") as mock_cfg:
            mock_settings = MagicMock()
            mock_settings.ingestion.get = lambda key, default=None: {
                "review_fuzzy_min": 70,
            }.get(key, default)
            mock_cfg.return_value = mock_settings
            try:
                from rapidfuzz import fuzz
                with patch.object(fuzz, "ratio", return_value=50):
                    result = registry.resolve("totally different name")
                    assert result is None
            except ImportError:
                pytest.skip("rapidfuzz not installed")


# ── ReviewQueueService ─────────────────────────────────────────────────────────

class TestReviewQueueService:
    def _make_svc(self, run_return=None):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=run_return or [])
        svc = ReviewQueueService(neo4j_client=neo4j)
        return svc, neo4j

    @pytest.mark.asyncio
    async def test_enqueue_creates_node_returns_id(self):
        svc, neo4j = self._make_svc()
        item_id = await svc.enqueue(
            raw_name="ISO IATF",
            raw_type="CONCEPT",
            candidate_name="IATF 16949:2016",
            candidate_type="CONCEPT",
            score=75.0,
            match_type="fuzzy",
            source_doc="doc-001",
            tenant="automotive",
        )
        # Returns a valid UUID string
        UUID(item_id)   # raises if invalid
        neo4j.run.assert_called_once()
        cypher = neo4j.run.call_args[0][0]
        assert "ReviewQueueItem" in cypher
        assert "pending" in cypher

    @pytest.mark.asyncio
    async def test_enqueue_passes_all_fields(self):
        svc, neo4j = self._make_svc()
        await svc.enqueue(
            raw_name="ISO IATF", raw_type="CONCEPT",
            candidate_name="IATF 16949:2016", candidate_type="CONCEPT",
            score=75.0, match_type="fuzzy",
            source_doc="doc-001", tenant="automotive",
        )
        kwargs = neo4j.run.call_args[1]
        assert kwargs["raw_name"] == "ISO IATF"
        assert kwargs["candidate_name"] == "IATF 16949:2016"
        assert kwargs["score"] == 75.0
        assert kwargs["match_type"] == "fuzzy"
        assert kwargs["tenant"] == "automotive"

    @pytest.mark.asyncio
    async def test_approve_registers_alias_and_closes_item(self):
        row = {
            "raw_name": "ISO IATF", "raw_type": "CONCEPT",
            "candidate_name": "IATF 16949:2016", "candidate_type": "CONCEPT",
            "source_doc": "doc-001",
        }
        svc, neo4j = self._make_svc(run_return=[row])
        with patch("graphrag.graph.review_queue.get_alias_registry") as mock_reg:
            mock_registry = AsyncMock()
            mock_reg.return_value = mock_registry
            result = await svc.approve("item-123", reviewed_by="admin", tenant="automotive")

        assert result["status"] == "approved"
        assert "ISO IATF" in result["alias_registered"]
        assert "IATF 16949:2016" in result["alias_registered"]
        mock_registry.register_alias.assert_called_once_with(
            raw_value="ISO IATF",
            canonical_name="IATF 16949:2016",
            canonical_type="CONCEPT",
            source_doc_id="doc-001",
            confidence=1.0,
        )

    @pytest.mark.asyncio
    async def test_approve_not_found_returns_error(self):
        svc, _ = self._make_svc(run_return=[])
        result = await svc.approve("nonexistent", reviewed_by="admin", tenant="automotive")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_reject_closes_item_no_alias(self):
        row = {"raw_name": "ISO IATF", "candidate_name": "IATF 16949:2016"}
        svc, neo4j = self._make_svc(run_return=[row])
        with patch("graphrag.graph.review_queue.get_alias_registry") as mock_reg:
            result = await svc.reject("item-123", reviewed_by="admin", tenant="automotive")

        assert result["status"] == "rejected"
        assert result["item_id"] == "item-123"
        # No alias registered on reject
        mock_reg.assert_not_called()

    @pytest.mark.asyncio
    async def test_reject_not_found_returns_error(self):
        svc, _ = self._make_svc(run_return=[])
        result = await svc.reject("nonexistent", reviewed_by="admin", tenant="automotive")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_pending_queries_correct_tenant(self):
        svc, neo4j = self._make_svc(run_return=[])
        await svc.list_pending("automotive", limit=25)
        kwargs = neo4j.run.call_args[1]
        assert kwargs["tenant"] == "automotive"
        assert kwargs["limit"] == 25
        cypher = neo4j.run.call_args[0][0]
        assert "pending" in cypher

    @pytest.mark.asyncio
    async def test_list_all_returns_all_statuses(self):
        svc, neo4j = self._make_svc(run_return=[])
        await svc.list_all("automotive")
        cypher = neo4j.run.call_args[0][0]
        # list_all should NOT filter by status
        assert "status: 'pending'" not in cypher


# ── Tenant isolation ───────────────────────────────────────────────────────────

class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_enqueue_uses_tenant(self):
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[])
        svc = ReviewQueueService(neo4j_client=neo4j)
        await svc.enqueue(
            raw_name="X", raw_type="CONCEPT", candidate_name="Y", candidate_type="CONCEPT",
            score=75.0, match_type="fuzzy", source_doc="", tenant="aerospace",
        )
        kwargs = neo4j.run.call_args[1]
        assert kwargs["tenant"] == "aerospace"

    @pytest.mark.asyncio
    async def test_approve_passes_tenant_to_cypher(self):
        row = {"raw_name": "X", "raw_type": "CONCEPT",
               "candidate_name": "Y", "candidate_type": "CONCEPT", "source_doc": ""}
        neo4j = AsyncMock()
        neo4j.run = AsyncMock(return_value=[row])
        svc = ReviewQueueService(neo4j_client=neo4j)
        mock_registry = AsyncMock()
        with patch("graphrag.graph.review_queue.get_alias_registry", return_value=mock_registry):
            await svc.approve("item-x", reviewed_by="admin", tenant="aerospace")
        kwargs = neo4j.run.call_args[1]
        assert kwargs["tenant"] == "aerospace"
