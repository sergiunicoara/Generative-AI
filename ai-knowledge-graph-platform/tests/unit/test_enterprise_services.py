from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.enterprise.lineage import LineageService
from graphrag.enterprise.metadata_governance import MetadataGovernanceService
from graphrag.enterprise.models import (
    ACLState,
    CollectionSchema,
    DocumentAccessPolicy,
    LineageAssertion,
    LineageRelation,
    MetadataEnvelope,
    SyncChange,
    SyncChangeType,
)
from graphrag.enterprise.sync import ContentSyncService


class _Neo4j:
    def __init__(self, responses=None):
        self.run = AsyncMock(side_effect=responses or [])


@pytest.mark.asyncio
async def test_metadata_validation_rejects_missing_required_field() -> None:
    neo4j = _Neo4j([[{"required_fields": ["jurisdiction"], "allowed_fields": ["jurisdiction"]}]])
    service = MetadataGovernanceService(neo4j)
    envelope = MetadataEnvelope(collection="contracts", schema_version="v1")

    with pytest.raises(ValueError, match="jurisdiction"):
        await service.validate(envelope, "tenant-a")


@pytest.mark.asyncio
async def test_metadata_schema_is_tenant_scoped() -> None:
    neo4j = _Neo4j([[{"id": "schema-1", "collection": "contracts", "version": "v1", "status": "active"}]])
    result = await MetadataGovernanceService(neo4j).register_schema(CollectionSchema(
        collection="contracts", version="v1", status="active", tenant="tenant-a",
    ))

    assert result["id"] == "schema-1"
    assert neo4j.run.call_args.kwargs["tenant"] == "tenant-a"


@pytest.mark.asyncio
async def test_sync_change_uses_normal_ingestion_pipeline() -> None:
    neo4j = _Neo4j([[], [], []])
    publish = AsyncMock(return_value="job-1")
    service = ContentSyncService(neo4j, publisher=publish)
    change = SyncChange(
        change_type=SyncChangeType.UPSERT,
        external_id="sharepoint-item-42",
        filename="contract.txt",
        text="The supplier shall deliver monthly reports.",
        metadata=MetadataEnvelope(
            collection="contracts", source_system="sharepoint", external_id="sharepoint-item-42",
        ),
        access_policy=DocumentAccessPolicy(
            mode="restricted", state=ACLState.KNOWN, allow_principals=["group:legal"],
            requires_group_resolution=True,
        ),
    )

    result = await service.apply_changes("sharepoint-contracts", [change], "tenant-a", cursor="delta-2")

    published = publish.await_args.args[0]
    assert result["queued"] == 1
    assert published.source_id == "sharepoint-contracts"
    assert published.metadata_envelope.external_id == "sharepoint-item-42"
    assert published.access_policy.requires_group_resolution is True


@pytest.mark.asyncio
async def test_lineage_submission_requires_source_backed_evidence() -> None:
    neo4j = _Neo4j([[{"review_id": "review-1", "status": "pending"}]])
    service = LineageService(neo4j)
    assertion = LineageAssertion(
        relation=LineageRelation.AMENDS,
        target_document_id="doc-old",
        evidence_chunk_id="chunk-1",
        evidence_quote="This amendment changes section 4.",
        confidence=0.93,
    )

    result = await service.submit_lineage("doc-new", assertion, "tenant-a")

    assert result == {"review_id": "review-1", "status": "pending"}
    cypher = neo4j.run.call_args.args[0]
    assert "MATCH (chunk:Chunk" in cypher
    assert "SUPPORTED_BY" in cypher
