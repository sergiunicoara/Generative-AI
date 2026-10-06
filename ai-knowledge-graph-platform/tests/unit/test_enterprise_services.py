from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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
async def test_sync_delete_invalidates_answers_before_advancing_cursor() -> None:
    neo4j = _Neo4j([[], [{"tombstoned": 1}], []])
    neo4j.begin_corpus_update = AsyncMock()
    neo4j.complete_corpus_update = AsyncMock(return_value=12)
    calls = MagicMock()
    calls.attach_mock(neo4j.run, "run")
    calls.attach_mock(neo4j.begin_corpus_update, "begin")
    calls.attach_mock(neo4j.complete_corpus_update, "complete")
    service = ContentSyncService(neo4j, publisher=AsyncMock())

    result = await service.apply_changes(
        "sharepoint-contracts",
        [SyncChange(change_type=SyncChangeType.DELETE, external_id="item-42")],
        "tenant-a",
        cursor="delta-3",
    )

    assert result["tombstoned"] == 1
    neo4j.begin_corpus_update.assert_awaited_once_with(
        "tenant-a", reason="content_sync_delete"
    )
    neo4j.complete_corpus_update.assert_awaited_once_with(
        "tenant-a", reason="content_sync_delete", outcome="completed"
    )
    assert neo4j.run.await_args_list[1].kwargs == {
        "source_id": "sharepoint-contracts",
        "tenant": "tenant-a",
        "external_ids": ["item-42"],
    }
    assert [call[0] for call in calls.mock_calls] == [
        "run", "begin", "run", "complete", "run",
    ]


@pytest.mark.asyncio
async def test_sync_reconcile_invalidates_answers_for_tombstones() -> None:
    neo4j = _Neo4j([[], [{"tombstoned": 2}], []])
    neo4j.begin_corpus_update = AsyncMock()
    neo4j.complete_corpus_update = AsyncMock(return_value=13)
    calls = MagicMock()
    calls.attach_mock(neo4j.run, "run")
    calls.attach_mock(neo4j.begin_corpus_update, "begin")
    calls.attach_mock(neo4j.complete_corpus_update, "complete")
    service = ContentSyncService(neo4j)

    result = await service.reconcile(
        "sharepoint-contracts", ["item-1", "item-1"], "tenant-a"
    )

    assert result == {"source_id": "sharepoint-contracts", "tombstoned": 2}
    assert neo4j.run.await_args_list[1].kwargs["discovered_external_ids"] == ["item-1"]
    neo4j.begin_corpus_update.assert_awaited_once_with(
        "tenant-a", reason="content_sync_reconcile"
    )
    neo4j.complete_corpus_update.assert_awaited_once_with(
        "tenant-a", reason="content_sync_reconcile", outcome="completed"
    )
    assert [call[0] for call in calls.mock_calls] == [
        "run", "begin", "run", "complete", "run",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "reconcile"])
async def test_sync_tombstone_failure_finalizes_corpus_update(operation: str) -> None:
    neo4j = _Neo4j([[], RuntimeError("tombstone failed")])
    neo4j.begin_corpus_update = AsyncMock()
    neo4j.complete_corpus_update = AsyncMock(return_value=14)
    service = ContentSyncService(neo4j)

    with pytest.raises(RuntimeError, match="tombstone failed"):
        if operation == "delete":
            await service.apply_changes(
                "sharepoint-contracts",
                [SyncChange(change_type=SyncChangeType.DELETE, external_id="item-42")],
                "tenant-a",
            )
        else:
            await service.reconcile("sharepoint-contracts", [], "tenant-a")

    neo4j.complete_corpus_update.assert_awaited_once_with(
        "tenant-a", reason=f"content_sync_{operation}", outcome="failed"
    )


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
