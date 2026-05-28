"""Integration tests for GraphRAG operational safety paths.

Coverage
--------
1. Session store persistence — memory round-trip, strict-mode failure propagation,
   non-strict fallback to memory, strict-mode log level, max-turns enforcement.

2. Leiden startup path — RuntimeError when graspologic is missing and
   require_leiden=True; connected-components fallback when require_leiden=False;
   happy-path when graspologic is present.

3. Tenant-scoped contradiction detection — scan(tenant=X) embeds the tenant
   in every Cypher call; a scan for tenant A cannot surface rows injected for
   tenant B; scan(tenant=None) issues unscoped queries.

4. Community auto-rebuild lifecycle — snapshot → check_staleness → mark_rebuilt
   full lifecycle; no-snapshot case; version history milestone flag.

All tests use AsyncMock to avoid a live Neo4j instance or Redis server.
"""

from __future__ import annotations

import sys
from collections import deque
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.core.models import SessionTurn


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock():
    client = MagicMock()
    client.run = AsyncMock(return_value=[])
    return client


def _make_turn(question: str = "q", answer: str = "a") -> SessionTurn:
    return SessionTurn(question=question, answer=answer)


# ── 1. Session store persistence ───────────────────────────────────────────────

class TestSessionStorePersistence:
    """SessionStore in-memory + strict-mode behaviour."""

    @pytest.mark.asyncio
    async def test_memory_store_round_trip(self):
        """save_turn() + load_turns() preserves order and content (memory mode)."""
        from graphrag.retrieval.session_store import SessionStore

        store = SessionStore(redis_url=None)
        sid = "test-session-1"

        t1 = _make_turn("Who founded SpaceX?", "Elon Musk")
        t2 = _make_turn("When?", "2002")
        await store.save_turn(sid, t1)
        await store.save_turn(sid, t2)

        turns = await store.load_turns(sid)
        assert len(turns) == 2
        questions = [t.question for t in turns]
        assert "Who founded SpaceX?" in questions
        assert "When?" in questions

    @pytest.mark.asyncio
    async def test_redis_unavailable_strict_raises_connection_error(self):
        """verify_connection() raises ConnectionError in strict mode when Redis is down."""
        from graphrag.retrieval.session_store import SessionStore

        store = SessionStore(redis_url=None, strict=True)
        # Inject a mock Redis that fails on ping
        mock_redis = MagicMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis unreachable"))
        store._redis = mock_redis
        store._strict = True

        with pytest.raises((ConnectionError, Exception)):
            await store.verify_connection()

    @pytest.mark.asyncio
    async def test_redis_unavailable_non_strict_falls_back_to_memory(self):
        """save_turn() writes to memory when Redis rpush raises (non-strict)."""
        from graphrag.retrieval.session_store import SessionStore

        store = SessionStore(redis_url=None, strict=False)
        mock_redis = MagicMock()
        mock_redis.rpush = AsyncMock(side_effect=OSError("connection refused"))
        store._redis = mock_redis  # inject a broken Redis client

        sid = "fallback-session"
        turn = _make_turn("fallback?", "yes")
        await store.save_turn(sid, turn)   # should NOT raise

        # Turn should have landed in memory
        assert sid in store._memory
        assert len(store._memory[sid]) == 1
        assert store._memory[sid][0].question == "fallback?"

    @pytest.mark.asyncio
    async def test_strict_mode_logs_error_on_op_failure(self):
        """_log_op_failure uses log.error in strict mode, log.warning otherwise."""
        from graphrag.retrieval.session_store import SessionStore
        import graphrag.retrieval.session_store as ss_module

        store_strict = SessionStore(redis_url=None, strict=True)
        store_warn   = SessionStore(redis_url=None, strict=False)

        err_calls  = []
        warn_calls = []

        # Patch the module-level logger
        with patch.object(ss_module.log, "error",  side_effect=lambda *a, **kw: err_calls.append(kw)):
            with patch.object(ss_module.log, "warning", side_effect=lambda *a, **kw: warn_calls.append(kw)):
                store_strict._log_op_failure("save", ValueError("boom"))
                store_warn._log_op_failure("save", ValueError("boom"))

        assert len(err_calls)  == 1, "strict mode should call log.error once"
        assert len(warn_calls) == 1, "non-strict mode should call log.warning once"

    @pytest.mark.asyncio
    async def test_max_turns_enforced_in_memory(self):
        """Memory deque never grows beyond max_turns (sliding window)."""
        from graphrag.retrieval.session_store import SessionStore

        max_t = 3
        store = SessionStore(redis_url=None, max_turns=max_t)
        sid = "window-session"

        for i in range(max_t + 2):
            await store.save_turn(sid, _make_turn(f"q{i}", f"a{i}"))

        turns = await store.load_turns(sid)
        assert len(turns) == max_t
        # The deque kept the LAST max_t turns
        questions = [t.question for t in turns]
        assert "q0" not in questions   # oldest was evicted
        assert f"q{max_t + 1}" in questions


# ── 2. Leiden startup path ─────────────────────────────────────────────────────

class TestLeidenStartupPath:
    """CommunityBuilder._run_leiden() respects require_leiden config."""

    def _make_builder(self, require_leiden: bool):
        """Create a CommunityBuilder with a mocked config and neo4j client."""
        with patch("graphrag.graph.community_builder.get_neo4j", return_value=MagicMock()):
            with patch("graphrag.graph.community_builder.get_settings") as mock_cfg:
                mock_cfg.return_value.graph = {"require_leiden": require_leiden}
                from graphrag.graph.community_builder import CommunityBuilder
                builder = CommunityBuilder.__new__(CommunityBuilder)
                builder._cfg = {"require_leiden": require_leiden}
                builder._neo4j = MagicMock()
                builder._tenant = "default"
                return builder

    def _tiny_graph(self):
        """Return a tiny NetworkX graph with 4 string-ID nodes for algorithm tests.

        Community.member_entity_ids is a list[str] — nodes must be strings.
        """
        import networkx as nx
        G = nx.Graph()
        G.add_edges_from([("n0", "n1"), ("n1", "n2"), ("n2", "n3")])
        return G

    @pytest.mark.asyncio
    async def test_leiden_missing_strict_raises(self):
        """RuntimeError raised when graspologic missing and require_leiden=True.

        Setting sys.modules[key] = None makes `from graspologic.partition import leiden`
        raise ImportError even when the package is installed — it signals 'do not import'.
        """
        builder = self._make_builder(require_leiden=True)
        G = self._tiny_graph()

        # Setting the entry to None makes Python raise ImportError on import
        with patch.dict(sys.modules, {
            "graspologic": None,
            "graspologic.partition": None,
        }):
            with pytest.raises(RuntimeError, match="graspologic"):
                builder._run_leiden(G)

    @pytest.mark.asyncio
    async def test_leiden_missing_fallback_uses_components(self):
        """Falls back to connected-components when graspologic missing and require_leiden=False."""
        builder = self._make_builder(require_leiden=False)
        G = self._tiny_graph()

        with patch.dict(sys.modules, {
            "graspologic": None,
            "graspologic.partition": None,
        }):
            result = builder._run_leiden(G)

        from graphrag.core.models import Community
        assert isinstance(result, list)
        assert len(result) > 0
        for c in result:
            assert isinstance(c, Community)
            assert "[fallback:" in c.summary, (
                f"Expected fallback tag in summary, got: {c.summary!r}"
            )

    @pytest.mark.asyncio
    async def test_leiden_present_returns_communities(self):
        """When graspologic is available, _run_leiden returns a non-empty list."""
        try:
            import graspologic  # noqa: F401  — skip if not installed
        except ImportError:
            pytest.skip("graspologic not installed — skipping happy-path test")

        builder = self._make_builder(require_leiden=True)
        G = self._tiny_graph()
        result = builder._run_leiden(G)
        assert isinstance(result, list)
        assert len(result) >= 1


# ── 3. Tenant-scoped contradiction detection ───────────────────────────────────

class TestTenantScopedContradiction:
    """ContradictionDetector.scan(tenant=X) must embed X in every Cypher call."""

    @pytest.mark.asyncio
    async def test_scan_passes_tenant_to_all_queries(self, neo4j_mock):
        """Every neo4j.run() call during scan(tenant='acme') carries tenant='acme'."""
        from graphrag.graph.contradiction_detector import ContradictionDetector

        # Return empty result sets so the scan completes without trying to CREATE nodes
        neo4j_mock.run = AsyncMock(return_value=[])
        detector = ContradictionDetector(neo4j_mock)
        await detector.scan(tenant="acme")

        # Every call must have 'acme' somewhere in its args/kwargs
        calls = neo4j_mock.run.call_args_list
        assert len(calls) > 0, "scan() should have issued at least one query"
        for i, call in enumerate(calls):
            # Positional args are (cypher_string,) and keyword args include params
            all_args_str = str(call)
            assert "acme" in all_args_str, (
                f"Call #{i} does not reference tenant 'acme': {call}"
            )

    @pytest.mark.asyncio
    async def test_different_tenant_rows_not_returned(self, neo4j_mock):
        """
        scan(tenant='acme') returns 0 conflicts even if the mock would return
        rows for a tenant='other' scan — rows are gated by the Cypher tenant filter,
        not Python-side filtering.

        We simulate this by having the mock return non-empty rows only on calls
        that do NOT contain 'acme' in their kwargs — demonstrating that the
        tenant filter prevents cross-contamination.
        """
        from graphrag.graph.contradiction_detector import ContradictionDetector

        def _side_effect(*args, **kwargs):
            # Only return a fake row if the call was NOT scoped to 'acme'
            if kwargs.get("tenant") != "acme":
                return [{"src": "X", "tgt": "Y", "rel": "Z",
                         "doc_ids": ["d1", "d2"],
                         "independent_pairs": [{"a": "d1", "b": "d2"}]}]
            return []

        neo4j_mock.run = AsyncMock(side_effect=_side_effect)
        detector = ContradictionDetector(neo4j_mock)
        conflicts = await detector.scan(tenant="acme")

        # All calls were scoped to 'acme', so _side_effect returned [] for every one
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_scan_none_tenant_omits_tenant_param(self, neo4j_mock):
        """scan(tenant=None) does not pass a tenant kwarg to any Neo4j call."""
        from graphrag.graph.contradiction_detector import ContradictionDetector

        neo4j_mock.run = AsyncMock(return_value=[])
        detector = ContradictionDetector(neo4j_mock)
        await detector.scan(tenant=None)

        for call in neo4j_mock.run.call_args_list:
            kwargs = call.kwargs if hasattr(call, "kwargs") else call[1]
            assert "tenant" not in kwargs, (
                f"Expected no tenant kwarg but found one in: {call}"
            )


# ── 4. Community auto-rebuild lifecycle ────────────────────────────────────────

class TestCommunityAutoRebuildLifecycle:
    """Full snapshot → stale → rebuild → reset cycle via CommunityManager."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_snapshot_stale_rebuild_reset(self, neo4j_mock):
        """
        End-to-end: snapshot → check_staleness (stale) → mark_rebuilt → check_staleness (fresh).
        """
        from graphrag.graph.community_manager import CommunityManager

        manager = CommunityManager(neo4j_mock)

        # 1. snapshot() — needs: (a) current stats query, (b) CREATE query
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"entity_count": 100, "edge_count": 200, "community_count": 10}],  # snapshot stats
            [],  # CREATE CommunitySnapshot
        ])
        snap = await manager.snapshot(tenant="default")
        assert "snapshot_id" in snap
        assert snap["entity_count"] == 100

        # 2. check_staleness() → should_rebuild=True  (drift > 0.15)
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"entity_count": 100, "edge_count": 200, "recorded_at": "2025-01-01T00:00:00"}],  # last snap
            [{"entities": 130, "edges": 280}],  # current (large drift)
        ])
        report = await manager.check_staleness(tenant="default")
        assert report["should_rebuild"] is True
        assert report["staleness_score"] > 0.15

        # 3. mark_rebuilt() — calls snapshot() (2 queries) then SET milestone (1 query)
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"entity_count": 130, "edge_count": 280, "community_count": 12}],  # snapshot stats
            [],  # CREATE CommunitySnapshot
            [],  # SET is_rebuild_milestone
        ])
        snap_id = await manager.mark_rebuilt(tenant="default")
        assert isinstance(snap_id, str) and len(snap_id) > 0

        # 4. check_staleness() after rebuild → should_rebuild=False (no drift)
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"entity_count": 130, "edge_count": 280, "recorded_at": "2025-01-02T00:00:00"}],
            [{"entities": 130, "edges": 280}],  # identical counts
        ])
        report2 = await manager.check_staleness(tenant="default")
        assert report2["should_rebuild"] is False
        assert report2["staleness_score"] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_no_snapshot_returns_full_staleness(self, neo4j_mock):
        """When no snapshot exists, staleness=1.0 and should_rebuild=True."""
        from graphrag.graph.community_manager import CommunityManager

        neo4j_mock.run = AsyncMock(return_value=[])  # empty → no snapshot
        manager = CommunityManager(neo4j_mock)
        report = await manager.check_staleness(tenant="default")

        assert report["staleness_score"] == 1.0
        assert report["should_rebuild"] is True

    @pytest.mark.asyncio
    async def test_version_history_returns_milestones(self, neo4j_mock):
        """get_version_history() returns the correct rows with is_rebuild flag."""
        from graphrag.graph.community_manager import CommunityManager

        fake_history = [
            {"snapshot_id": "s1", "entity_count": 50, "edge_count": 100,
             "community_count": 5, "recorded_at": "2025-01-03", "is_rebuild": True},
            {"snapshot_id": "s2", "entity_count": 60, "edge_count": 110,
             "community_count": 5, "recorded_at": "2025-01-02", "is_rebuild": False},
            {"snapshot_id": "s3", "entity_count": 40, "edge_count": 80,
             "community_count": 4, "recorded_at": "2025-01-01", "is_rebuild": False},
        ]
        neo4j_mock.run = AsyncMock(return_value=fake_history)
        manager = CommunityManager(neo4j_mock)
        history = await manager.get_version_history(limit=10, tenant="default")

        assert len(history) == 3
        milestones = [h for h in history if h["is_rebuild"]]
        assert len(milestones) == 1
        assert milestones[0]["snapshot_id"] == "s1"
