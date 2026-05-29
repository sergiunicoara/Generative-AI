"""Unit tests for GraphWriter — entity writing with alias resolution and dedup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.core.models import Chunk, Entity, Relation


def _make_entity(name: str, etype: str = "ORG") -> Entity:
    return Entity(name=name, type=etype, source_doc_id="doc1")


def _make_chunk(tenant: str = "default") -> Chunk:
    return Chunk(document_id="doc1", text="some text", chunk_index=0, tenant=tenant)


def _build_writer():
    """GraphWriter with all Neo4j/service dependencies mocked."""
    with (
        patch("graphrag.ingestion.graph_writer.get_neo4j"),
        patch("graphrag.ingestion.graph_writer.AuditTrail"),
        patch("graphrag.ingestion.graph_writer.IngestionValidator"),
        patch("graphrag.ingestion.graph_writer.CycleDetector"),
        patch("graphrag.ingestion.graph_writer.QuarantineService"),
        patch("graphrag.ingestion.graph_writer.ContradictionDetector"),
        patch("graphrag.ingestion.graph_writer.get_ontology_registry"),
        patch("graphrag.ingestion.graph_writer.get_settings"),
        patch("graphrag.ingestion.graph_writer.get_alias_registry"),
    ):
        from graphrag.ingestion.graph_writer import GraphWriter
        writer = GraphWriter.__new__(GraphWriter)
        writer._neo4j = AsyncMock()
        writer._audit = AsyncMock()
        writer._validator = AsyncMock()
        writer._cycle_detector = AsyncMock()
        writer._quarantine = AsyncMock()
        writer._contradiction = AsyncMock()
        writer._changed_by = "test"
        writer._registry_loaded_tenants = set()
        writer._ontology_loaded = True
        writer._cfg = MagicMock()
        writer._cfg.graph = {}
    return writer


class TestWriteEntities:
    async def test_alias_resolved_entity_not_written_to_neo4j(self):
        """When alias resolves to a canonical name, merge_entity must NOT be called."""
        writer = _build_writer()
        chunk = _make_chunk()

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=("SpaceX", "ORG"))
        mock_registry.register_alias = AsyncMock()
        writer._neo4j.merge_mentions = AsyncMock()

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            entity = _make_entity("Space Exploration Technologies")
            result = await writer.write_entities([entity], chunk)

        writer._neo4j.merge_entity.assert_not_called()
        assert result == []   # alias-resolved entities are not in the written list

    async def test_new_entity_is_written(self):
        """An entity with no alias and no embedding duplicate is written to Neo4j."""
        writer = _build_writer()
        chunk = _make_chunk()

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)   # no alias
        mock_registry.find_duplicate_by_embedding = AsyncMock(return_value=None)
        mock_registry._exact = {}

        writer._neo4j.entity_exists = AsyncMock(return_value=False)
        writer._neo4j.merge_entity = AsyncMock()
        writer._neo4j.merge_mentions = AsyncMock()
        writer._audit.log_entity_change = AsyncMock()

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            entity = _make_entity("SpaceX")
            result = await writer.write_entities([entity], chunk)

        writer._neo4j.merge_entity.assert_called_once()
        assert len(result) == 1
        assert result[0].name == "SpaceX"

    async def test_embedding_duplicate_not_written(self):
        """Entity whose embedding matches an existing entity is not duplicated."""
        writer = _build_writer()
        chunk = _make_chunk()

        entity = _make_entity("Tesla Inc")
        entity.embedding = [0.1] * 768

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)
        # Duplicate found via embedding
        mock_registry.find_duplicate_by_embedding = AsyncMock(
            return_value=("Tesla", "ORG", 0.96)
        )
        mock_registry.register_alias = AsyncMock()
        writer._neo4j.merge_mentions = AsyncMock()

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            result = await writer.write_entities([entity], chunk)

        writer._neo4j.merge_entity.assert_not_called()
        assert result == []

    async def test_tenant_propagated_to_entity(self):
        """chunk.tenant must be applied to the entity before merge."""
        writer = _build_writer()
        chunk = _make_chunk(tenant="finance")

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)
        mock_registry.find_duplicate_by_embedding = AsyncMock(return_value=None)
        mock_registry._exact = {}

        writer._neo4j.entity_exists = AsyncMock(return_value=False)
        writer._neo4j.merge_entity = AsyncMock()
        writer._neo4j.merge_mentions = AsyncMock()
        writer._audit.log_entity_change = AsyncMock()

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            entity = _make_entity("Acme Corp")
            await writer.write_entities([entity], chunk)

        assert entity.tenant == "finance"


class TestWriteRelations:
    async def test_invalid_ontology_relation_skipped(self):
        """Relations that fail ontology validation must not be written."""
        writer = _build_writer()

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)

        # Ontology rejects the relation
        mock_ontology = MagicMock()
        mock_ontology.validate_relation_triplet = MagicMock(return_value=(False, "RELATED_TO"))
        mock_ontology.record_schema_event = AsyncMock()
        writer._ontology = mock_ontology

        src = _make_entity("A", "PERSON")
        tgt = _make_entity("B", "ORG")
        rel = Relation(
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relation="INVALID_REL",
        )
        entity_map = {src.id: src, tgt.id: tgt}

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            await writer.write_relations([rel], entity_map, doc_id="doc1", tenant="default")

        writer._neo4j.merge_relation.assert_not_called()

    async def test_valid_relation_written(self):
        """Relations that pass ontology validation are written to Neo4j."""
        writer = _build_writer()

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)

        mock_ontology = MagicMock()
        mock_ontology.validate_relation_triplet = MagicMock(return_value=(True, "WORKS_AT"))
        writer._ontology = mock_ontology

        writer._neo4j.merge_relation = AsyncMock()
        writer._audit.log_relation_change = AsyncMock()

        src = _make_entity("Alice", "PERSON")
        tgt = _make_entity("Acme", "ORG")
        rel = Relation(
            source_entity_id=src.id,
            target_entity_id=tgt.id,
            relation="WORKS_AT",
        )
        entity_map = {src.id: src, tgt.id: tgt}

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            await writer.write_relations([rel], entity_map, doc_id="doc1", tenant="default")

        writer._neo4j.merge_relation.assert_called_once()

    async def test_missing_entity_in_map_skips_relation(self):
        """Relations whose src or tgt is not in entity_map are silently skipped."""
        writer = _build_writer()

        mock_registry = MagicMock()
        mock_registry.resolve = MagicMock(return_value=None)
        writer._ontology = MagicMock()

        rel = Relation(
            source_entity_id="unknown_src",
            target_entity_id="unknown_tgt",
            relation="USES",
        )

        with patch.object(writer, "_get_registry", return_value=mock_registry), \
             patch.object(writer, "_ensure_registry", AsyncMock()):
            await writer.write_relations([rel], {}, doc_id="doc1")

        writer._neo4j.merge_relation.assert_not_called()
