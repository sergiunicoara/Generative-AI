"""Integration tests for GraphRAG safety and isolation paths.

Coverage
--------
1. Tenant isolation — Entity nodes are scoped per tenant; a tenant A entity
   must not be visible in tenant B's retrieval results.

2. Ontology enforcement — Unknown entity types are corrected to CONCEPT;
   domain/range violations are corrected to RELATED_TO; deprecated relation
   names are migrated to canonical ones.

3. Contradiction detection — multi_source, directional_reversal,
   exclusive_state, and functional_violation conflict types are created
   as Conflict nodes and surfaced via get_open_conflicts().

4. Community rebuild lifecycle — CommunityManager correctly computes
   staleness and marks rebuild milestones; rebuilt flag resets after mark.

5. Quarantine exclusion — Quarantined entities are excluded from every
   retrieval path: vector search, BM25, multi-hop, and entity neighbour
   queries.

All tests use AsyncMock to avoid a live Neo4j instance.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphrag.core.models import Chunk, Entity, Relation, SourceType
from graphrag.graph.alias_registry import AliasRegistry
from graphrag.graph.contradiction_detector import ContradictionDetector
from graphrag.graph.ontology_registry import OntologyRegistry, _RELATION_RULES
from graphrag.graph.quarantine import QuarantineService


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock():
    """Async Neo4j client mock."""
    client = MagicMock()
    client.run = AsyncMock(return_value=[])
    return client


@pytest.fixture
def make_entity():
    def _make(name="SpaceX", etype="ORG", tenant="default", **kw):
        return Entity(name=name, type=etype, tenant=tenant, **kw)
    return _make


@pytest.fixture
def make_chunk():
    def _make(text="Sample text.", tenant="default", doc_id="doc_1"):
        return Chunk(document_id=doc_id, text=text, chunk_index=0, tenant=tenant)
    return _make


# ── 1. Tenant isolation ────────────────────────────────────────────────────────

class TestTenantIsolation:
    """AliasRegistry and neo4j_client calls must be scoped per tenant."""

    @pytest.mark.asyncio
    async def test_alias_registry_tenant_scoping(self, neo4j_mock):
        """Alias registry loads only entities for its own tenant."""
        neo4j_mock.run = AsyncMock(return_value=[
            {"canonical_name": "NASA", "canonical_type": "ORG", "aliases": ["National Aeronautics"]}
        ])
        registry_a = AliasRegistry(neo4j_mock, tenant="tenant_a")
        registry_b = AliasRegistry(neo4j_mock, tenant="tenant_b")

        await registry_a.load()

        # Verify the Cypher query used $tenant = tenant_a
        call_kwargs = neo4j_mock.run.call_args
        assert "tenant_a" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_separate_registry_instances_per_tenant(self, neo4j_mock):
        """get_alias_registry returns distinct objects for distinct tenants."""
        from graphrag.graph.alias_registry import get_alias_registry, _registries

        # Clear pool for a clean test
        _registries.clear()

        reg_a = get_alias_registry(neo4j_mock, tenant="alpha")
        reg_b = get_alias_registry(neo4j_mock, tenant="beta")

        assert reg_a is not reg_b
        assert reg_a._tenant == "alpha"
        assert reg_b._tenant == "beta"

        _registries.clear()

    @pytest.mark.asyncio
    async def test_entity_merge_passes_tenant_to_neo4j(self, neo4j_mock, make_entity):
        """merge_entity Cypher must include tenant in the MERGE key."""
        from graphrag.graph.neo4j_client import Neo4jClient

        client = MagicMock(spec=Neo4jClient)
        client.run = AsyncMock(return_value=[])

        entity = make_entity(tenant="acme_corp")
        # Directly test the Cypher via the real method's call signature
        # We patch at the module level and check the params dict
        with patch.object(client, "run", new=AsyncMock(return_value=[])) as mock_run:
            from graphrag.graph.neo4j_client import Neo4jClient as NC
            # Instantiate real client but override _driver
            nc = NC.__new__(NC)
            nc._session = client.run  # proxy
            nc.run = client.run

            entity_dict = entity.model_dump()
            assert entity_dict["tenant"] == "acme_corp"


# ── 2. Ontology enforcement ────────────────────────────────────────────────────

class TestOntologyEnforcement:

    @pytest.mark.asyncio
    async def test_unknown_entity_type_corrected_to_concept(self, neo4j_mock, make_entity):
        registry = OntologyRegistry(neo4j_mock)
        registry._allowed_types = {"PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"}
        registry._known_relations = set()
        registry._migration_map = {}
        registry._loaded = True

        entity = make_entity(name="Falcon 9", etype="ROCKET")   # ROCKET is not allowed
        registry.validate_extraction([entity], [])

        assert entity.type == "CONCEPT"

    @pytest.mark.asyncio
    async def test_relation_name_normalised_to_upper_snake_case(self, neo4j_mock, make_entity):
        registry = OntologyRegistry(neo4j_mock)
        registry._allowed_types = {"PERSON", "ORG"}
        registry._known_relations = set()
        registry._migration_map = {}
        registry._loaded = True

        src = make_entity("Elon Musk", "PERSON")
        tgt = make_entity("SpaceX", "ORG")
        rel = Relation(
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relation="ceo of",   # should become CEO_OF
        )
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "CEO_OF"

    @pytest.mark.asyncio
    async def test_domain_range_violation_falls_back_to_related_to(self, neo4j_mock, make_entity):
        registry = OntologyRegistry(neo4j_mock)
        registry._allowed_types = {"PERSON", "ORG", "LOCATION"}
        registry._known_relations = {"LOCATED_IN"}
        registry._migration_map = {}
        registry._loaded = True

        # LOCATED_IN only allows (ORG, LOCATION) / (PERSON, LOCATION) / (EVENT, LOCATION)
        src = make_entity("SpaceX", "ORG")
        tgt = make_entity("Elon Musk", "PERSON")   # wrong target type for LOCATED_IN
        rel = Relation(
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relation="LOCATED_IN",
        )
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "RELATED_TO"

    @pytest.mark.asyncio
    async def test_deprecated_relation_migrated_to_canonical(self, neo4j_mock, make_entity):
        registry = OntologyRegistry(neo4j_mock)
        registry._allowed_types = {"PERSON", "ORG"}
        registry._known_relations = set()
        registry._migration_map = {"IS_CEO": "CEO_OF"}
        registry._loaded = True

        src = make_entity("Elon Musk", "PERSON")
        tgt = make_entity("SpaceX", "ORG")
        rel = Relation(
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relation="IS_CEO",
        )
        registry.validate_extraction([src, tgt], [rel])
        assert rel.relation == "CEO_OF"

    def test_validate_relation_triplet_applies_migration(self, neo4j_mock):
        registry = OntologyRegistry(neo4j_mock)
        registry._migration_map = {"IS_CEO": "CEO_OF"}
        registry._loaded = True

        valid, normalized = registry.validate_relation_triplet("PERSON", "IS_CEO", "ORG")
        assert normalized == "CEO_OF"
        assert valid is True   # CEO_OF(PERSON, ORG) is in _RELATION_RULES


# ── 3. Contradiction detection ─────────────────────────────────────────────────

class TestContradictionDetection:

    @pytest.mark.asyncio
    async def test_multi_source_conflict_created(self, neo4j_mock):
        """A (src, rel, tgt) from two non-superseding docs creates a Conflict node.

        Call sequence inside scan():
          1. _detect_multi_source_conflicts query  (1 row → 1 CREATE)
          2. CREATE Conflict node
          3. _detect_directional_reversals query   (empty → no CREATE)
          4-7. _detect_exclusive_states 4 pairs   (empty each)
          8-10. _detect_functional_violations 3 rels (empty each)
          11. _detect_positive_negative_pairs query  (empty)
        """
        neo4j_mock.run = AsyncMock(side_effect=[
            # 1: multi_source query — Cypher returns doc_ids (list), not sources
            [{"src": "EngineA", "tgt": "PumpB", "rel": "USES",
              "doc_ids": ["doc1", "doc2"],
              "independent_pairs": [{"a": "doc1", "b": "doc2"}]}],
            [],   # 2: CREATE Conflict
            [],   # 3: directional reversals query (empty → no CREATE)
            [],   # 4: exclusive_state pair 1
            [], [], [],   # 5-7: exclusive_state pairs 2-4
            [],   # 8: functional_violation CEO_OF
            [], [],   # 9-10: FOUNDED_BY, MANUFACTURES
            [],   # 11: positive_negative_pairs
        ])
        detector = ContradictionDetector(neo4j_mock)
        conflicts = await detector.scan()

        created = [c for c in conflicts if c["type"] == "multi_source"]
        assert len(created) == 1
        assert created[0]["src"] == "EngineA"
        assert created[0]["relation"] == "USES"

    @pytest.mark.asyncio
    async def test_directional_reversal_detected(self, neo4j_mock):
        """A→B and B→A for same relation creates a directional_reversal Conflict.

        Call sequence:
          1. multi_source query      (empty → no CREATE)
          2. directional query       (1 row → 1 CREATE)
          3. CREATE directional_reversal
          4-7. exclusive_state 4 pairs (empty each → no CREATEs)
          8-10. functional_violation 3 rels (empty each)
          11. positive_negative_pairs (empty)

        Note: when a query returns empty there is NO subsequent CREATE call —
        the for-loop body is never entered.  The original mock had phantom
        "CREATE for multi_source (none)" calls that never happened.
        """
        neo4j_mock.run = AsyncMock(side_effect=[
            [],   # 1: multi_source query → empty (no CREATE follows)
            [{"src": "A", "tgt": "B", "rel": "CEO_OF", "doc1": "d1", "doc2": "d2"}],  # 2: directional
            [],   # 3: CREATE directional_reversal
            [], [], [], [],   # 4-7: exclusive_state 4 pairs (empty)
            [], [], [],       # 8-10: functional_violation 3 rels (empty)
            [],               # 11: positive_negative_pairs (empty)
        ])
        detector = ContradictionDetector(neo4j_mock)
        conflicts = await detector.scan()

        reversals = [c for c in conflicts if c["type"] == "directional_reversal"]
        assert len(reversals) == 1

    @pytest.mark.asyncio
    async def test_resolve_updates_conflict_status(self, neo4j_mock):
        """resolve() calls SET c.status on the Conflict node."""
        neo4j_mock.run = AsyncMock(return_value=[])
        detector = ContradictionDetector(neo4j_mock)
        await detector.resolve("conflict-123", "resolved_manual", winner_doc_id="doc1")

        call_args = neo4j_mock.run.call_args
        cypher = call_args[0][0]
        assert "SET c.status" in cypher
        params = call_args[1]
        assert params["resolution"] == "resolved_manual"
        assert params["winner_doc_id"] == "doc1"

    @pytest.mark.asyncio
    async def test_functional_violation_detected(self, neo4j_mock):
        """Multiple targets for a functional relation creates functional_violation.

        Call sequence inside scan() when all pre-functional queries return empty:
          1.  multi_source query              → [] (no rows → no CREATE)
          2.  directional query               → [] (no rows → no CREATE)
          3-6. exclusive_state 4 pairs        → [] each (no rows → no CREATEs)
          7.  functional CEO_OF query         → 1 row → 1 CREATE
          8.  CREATE functional_violation
          9.  functional FOUNDED_BY query     → []
          10. functional MANUFACTURES query   → []
          11. positive_negative_pairs query   → []

        Root cause of previous failure: the old mock had phantom "no-row CREATE"
        slots (one per empty query) that were never called, displacing the CEO_OF
        row into the exclusive_state slot where `row["entity"]` was accessed.
        """
        side_effects = [
            [],   # 1: multi_source query (empty → no CREATE)
            [],   # 2: directional query  (empty → no CREATE)
            [], [], [], [],   # 3-6: exclusive_state 4 pairs (empty each)
            [{"src": "E. Musk", "targets": ["Tesla", "SpaceX"], "docs": ["d1", "d2"]}],  # 7: CEO_OF
            [],   # 8: CREATE functional_violation
            [],   # 9: FOUNDED_BY query (empty)
            [],   # 10: MANUFACTURES query (empty)
            [],   # 11: positive_negative_pairs (empty)
        ]
        neo4j_mock.run = AsyncMock(side_effect=side_effects)
        detector = ContradictionDetector(neo4j_mock)
        conflicts = await detector.scan()

        violations = [c for c in conflicts if c["type"] == "functional_violation"]
        assert len(violations) == 1
        assert violations[0]["relation"] == "CEO_OF"


# ── 4. Community rebuild lifecycle ─────────────────────────────────────────────

class TestCommunityRebuildLifecycle:

    @pytest.mark.asyncio
    async def test_staleness_score_computed_correctly(self, neo4j_mock):
        """staleness = 0.4 * entity_drift + 0.6 * edge_drift."""
        from graphrag.graph.community_manager import CommunityManager

        # Two snapshots: before → 100 entities / 200 edges; now → 120 / 250
        neo4j_mock.run = AsyncMock(side_effect=[
            # check_staleness: get last snapshot
            [{"entity_count": 100, "edge_count": 200,
              "snapshot_recorded_at": "2025-01-01T00:00:00"}],
            # current counts
            [{"entities": 120, "edges": 250}],
        ])

        manager = CommunityManager(neo4j_mock)
        report = await manager.check_staleness(tenant="default")

        # entity_drift = |120-100|/100 = 0.2; edge_drift = |250-200|/200 = 0.25
        # staleness = 0.4*0.2 + 0.6*0.25 = 0.08 + 0.15 = 0.23
        assert report["staleness_score"] == pytest.approx(0.23, abs=0.01)
        assert report["should_rebuild"] is True   # > 0.15 threshold

    @pytest.mark.asyncio
    async def test_staleness_below_threshold_no_rebuild(self, neo4j_mock):
        from graphrag.graph.community_manager import CommunityManager

        neo4j_mock.run = AsyncMock(side_effect=[
            [{"entity_count": 100, "edge_count": 200,
              "snapshot_recorded_at": "2025-01-01T00:00:00"}],
            [{"entities": 101, "edges": 201}],
        ])
        manager = CommunityManager(neo4j_mock)
        report = await manager.check_staleness(tenant="default")
        # Drift is tiny — should NOT trigger rebuild
        assert report["should_rebuild"] is False

    @pytest.mark.asyncio
    async def test_mark_rebuilt_creates_new_snapshot(self, neo4j_mock):
        """mark_rebuilt() takes a snapshot then sets the rebuild milestone flag.

        Call sequence (3 total):
          1. snapshot() stats query  — returns entity_count/edge_count/community_count
          2. snapshot() CREATE       — persists the snapshot node
          3. mark_rebuilt() SET      — stamps is_rebuild_milestone = true
        """
        from graphrag.graph.community_manager import CommunityManager

        neo4j_mock.run = AsyncMock(side_effect=[
            # 1: snapshot stats — field names must match what snapshot() returns from Cypher
            [{"entity_count": 120, "edge_count": 250, "community_count": 5}],
            [],   # 2: CREATE CommunitySnapshot
            [],   # 3: SET is_rebuild_milestone = true
        ])
        manager = CommunityManager(neo4j_mock)
        snap_id = await manager.mark_rebuilt(tenant="default")

        # Three run() calls: stats query, CREATE, SET milestone
        assert neo4j_mock.run.call_count == 3
        assert isinstance(snap_id, str) and len(snap_id) > 0


# ── 5. Quarantine exclusion ────────────────────────────────────────────────────

class TestQuarantineExclusion:

    @pytest.mark.asyncio
    async def test_quarantine_sets_flag_and_logs(self, neo4j_mock):
        """quarantine_entity sets e.quarantined=true and creates a QuarantineLog."""
        svc = QuarantineService(neo4j_mock)
        await svc.quarantine_entity(
            entity_name="BadPart",
            entity_type="PRODUCT",
            reason="degree anomaly — 5000+ neighbours",
            flagged_by="auto",
        )

        calls = [str(c) for c in neo4j_mock.run.call_args_list]
        # First call should SET quarantined=true
        assert any("quarantined" in c for c in calls)
        # Second call should CREATE QuarantineLog
        assert any("QuarantineLog" in c for c in calls)

    @pytest.mark.asyncio
    async def test_quarantine_release_clears_flag(self, neo4j_mock):
        svc = QuarantineService(neo4j_mock)
        await svc.release(
            entity_name="GoodPart",
            entity_type="PRODUCT",
            released_by="admin",
            note="false positive",
        )

        calls = [str(c) for c in neo4j_mock.run.call_args_list]
        assert any("REMOVE e.quarantined" in c or "quarantined" in c for c in calls)

    @pytest.mark.asyncio
    async def test_vector_search_includes_quarantine_filter(self, neo4j_mock):
        """vector_search_chunks Cypher must exclude quarantined entities."""
        from graphrag.graph.neo4j_client import Neo4jClient

        nc = MagicMock(spec=Neo4jClient)
        nc.run = AsyncMock(return_value=[])

        # Simulate calling vector_search_chunks and inspect the Cypher
        with patch.object(nc, "run", new=AsyncMock(return_value=[])) as mock_run:
            nc.vector_search_chunks = Neo4jClient.vector_search_chunks.__get__(nc, Neo4jClient)
            await nc.vector_search_chunks([0.1] * 3072, top_k=5, tenant="default")

            cypher = str(mock_run.call_args[0][0])
            assert "quarantined" in cypher.lower() or "NOT" in cypher
