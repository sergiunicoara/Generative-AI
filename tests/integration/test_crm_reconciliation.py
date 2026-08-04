"""§16 P2 exit criterion — 'Identical, changed, merged, converted, archived, and
deleted fixtures behave correctly.' End to end through SalesforceAdapter ->
CrmIngestionPipeline -> reconciliation -> repositories, against the real Neo4j
container (docker-compose's neo4j service, host port 7688).
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.domain.identity import crm_entity_id
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.salesforce import SalesforceAdapter
from src.ingestion.pipeline import CrmIngestionPipeline
from src.ingestion.reconciliation import ReconciliationOutcome

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T1 = datetime(2026, 1, 2, tzinfo=timezone.utc)
_T2 = datetime(2026, 1, 3, tzinfo=timezone.utc)


def _pipeline(executor) -> tuple[CrmIngestionPipeline, CrmRepository, SourceRepository]:
    crm_repo = CrmRepository(executor)
    source_repo = SourceRepository(executor)
    pipeline = CrmIngestionPipeline(crm_repo, source_repo, SalesforceAdapter())
    return pipeline, crm_repo, source_repo


def _ws() -> str:
    return f"ws-crm-{uuid4().hex[:8]}"


async def test_identical_reingest_is_a_no_op(executor):
    workspace_id = _ws()
    pipeline, crm_repo, source_repo = _pipeline(executor)
    account_raw = {
        "Id": "001xx0000004C92", "Name": "Acme Corp", "Website": "acme.com",
        "IsDeleted": False, "MasterRecordId": None,
    }

    first = await pipeline.ingest_accounts(workspace_id, [account_raw], ingestion_run_id="run-1", observed_at=_T0)
    second = await pipeline.ingest_accounts(workspace_id, [account_raw], ingestion_run_id="run-2", observed_at=_T1)

    assert first[0].outcome == ReconciliationOutcome.CREATED
    assert second[0].outcome == ReconciliationOutcome.NO_OP

    account_id = crm_entity_id(workspace_id, "salesforce", "Account", account_raw["Id"])
    source_record_id = account_id
    snapshots = await source_repo.list_snapshots(workspace_id, source_record_id)
    assert len(snapshots) == 1  # identical re-ingest created zero additional snapshots

    account = await crm_repo.get_account(workspace_id, account_id)
    assert account.name == "Acme Corp"


async def test_changed_content_supersedes_and_preserves_prior_snapshot(executor):
    workspace_id = _ws()
    pipeline, crm_repo, source_repo = _pipeline(executor)
    account_v1 = {
        "Id": "001xx0000004C93", "Name": "Acme Corp", "Website": "acme.com",
        "IsDeleted": False, "MasterRecordId": None,
    }
    account_v2 = {**account_v1, "Name": "Acme Corporation"}

    await pipeline.ingest_accounts(workspace_id, [account_v1], ingestion_run_id="run-1", observed_at=_T0)
    results = await pipeline.ingest_accounts(workspace_id, [account_v2], ingestion_run_id="run-2", observed_at=_T1)

    assert results[0].outcome == ReconciliationOutcome.SUPERSEDED

    account_id = crm_entity_id(workspace_id, "salesforce", "Account", account_v1["Id"])
    account = await crm_repo.get_account(workspace_id, account_id)
    assert account.name == "Acme Corporation"  # entity reflects the new content

    snapshots = await source_repo.list_snapshots(workspace_id, account_id)
    assert len(snapshots) == 2
    by_version = {s.source_version: s for s in snapshots}
    assert by_version[1].superseded is True  # prior snapshot preserved, marked superseded
    assert by_version[1].content_hash != by_version[2].content_hash
    assert by_version[2].superseded is False


async def test_account_merge_sets_merged_into_without_affecting_survivor(executor):
    workspace_id = _ws()
    pipeline, crm_repo, _ = _pipeline(executor)
    survivor_raw = {
        "Id": "001xxSURVIVOR", "Name": "Global Acme", "Website": "acme.com",
        "IsDeleted": False, "MasterRecordId": None,
    }
    merged_raw = {
        "Id": "001xxMERGED", "Name": "Acme West", "Website": "acme-west.com",
        "IsDeleted": False, "MasterRecordId": "001xxSURVIVOR",
    }

    await pipeline.ingest_accounts(workspace_id, [survivor_raw, merged_raw], ingestion_run_id="run-1", observed_at=_T0)

    survivor_id = crm_entity_id(workspace_id, "salesforce", "Account", survivor_raw["Id"])
    merged_id = crm_entity_id(workspace_id, "salesforce", "Account", merged_raw["Id"])

    survivor = await crm_repo.get_account(workspace_id, survivor_id)
    merged = await crm_repo.get_account(workspace_id, merged_id)

    assert merged.merged_into_account_id == survivor_id
    assert survivor.merged_into_account_id is None


async def test_lead_conversion_sets_converted_fields(executor):
    workspace_id = _ws()
    pipeline, _, _ = _pipeline(executor)
    lead_open = {
        "Id": "00Qxx0000004C92", "Name": "Jane Prospect", "Email": "jane@prospect.com",
        "IsConverted": False, "IsDeleted": False,
    }
    lead_converted = {
        **lead_open, "IsConverted": True, "ConvertedContactId": "003xxCONTACT",
    }

    await pipeline.ingest_leads(workspace_id, [lead_open], ingestion_run_id="run-1", observed_at=_T0)
    results = await pipeline.ingest_leads(workspace_id, [lead_converted], ingestion_run_id="run-2", observed_at=_T1)
    assert results[0].outcome == ReconciliationOutcome.SUPERSEDED

    from src.graph.repositories.crm_repository import CrmRepository as _CrmRepository
    crm_repo = _CrmRepository(executor)
    lead_id = crm_entity_id(workspace_id, "salesforce", "Lead", lead_open["Id"])
    lead = await crm_repo.get_lead(workspace_id, lead_id)

    assert lead.converted_to_type == "Contact"
    assert lead.converted_to_id == crm_entity_id(workspace_id, "salesforce", "Contact", "003xxCONTACT")


async def test_opportunity_archived_transition_is_reflected(executor):
    workspace_id = _ws()
    pipeline, crm_repo, _ = _pipeline(executor)
    opp_open = {
        "Id": "006xxOPP1", "Name": "Acme Renewal", "AccountId": "001xxACC",
        "OwnerId": "005xxOWNER", "StageName": "Negotiation", "IsClosed": False, "IsDeleted": False,
    }
    opp_closed = {**opp_open, "StageName": "Closed Lost", "IsClosed": True}

    await pipeline.ingest_opportunities(workspace_id, [opp_open], ingestion_run_id="run-1", observed_at=_T0)
    results = await pipeline.ingest_opportunities(workspace_id, [opp_closed], ingestion_run_id="run-2", observed_at=_T1)
    assert results[0].outcome == ReconciliationOutcome.SUPERSEDED

    opportunity_id = crm_entity_id(workspace_id, "salesforce", "Opportunity", opp_open["Id"])
    opportunity = await crm_repo.get_opportunity(workspace_id, opportunity_id)
    assert opportunity.is_open is False
    assert opportunity.stage == "Closed Lost"


async def test_deletion_tombstones_the_source_record(executor):
    workspace_id = _ws()
    pipeline, crm_repo, source_repo = _pipeline(executor)
    account_raw = {
        "Id": "001xxDELETE", "Name": "Acme To Delete", "Website": "acme.com",
        "IsDeleted": False, "MasterRecordId": None,
    }
    account_deleted_raw = {**account_raw, "IsDeleted": True}

    await pipeline.ingest_accounts(workspace_id, [account_raw], ingestion_run_id="run-1", observed_at=_T0)
    results = await pipeline.ingest_accounts(
        workspace_id, [account_deleted_raw], ingestion_run_id="run-2", observed_at=_T1
    )
    assert results[0].outcome == ReconciliationOutcome.TOMBSTONED

    account_id = crm_entity_id(workspace_id, "salesforce", "Account", account_raw["Id"])
    record = await source_repo.get_source_record(workspace_id, account_id)
    assert record.source_status.value == "DELETED"

    # the underlying Account entity node is not silently dropped — its last
    # known content remains readable even though the source is now inactive.
    account = await crm_repo.get_account(workspace_id, account_id)
    assert account is not None
    assert account.name == "Acme To Delete"


async def test_deletion_without_adapter_support_is_refused():
    """Trustworthiness of the deletion signal is a per-adapter declaration, not
    an assumption reconciliation makes for itself (§6)."""
    from src.ingestion.reconciliation import reconcile_deletion

    class _UntrustworthyAdapter:
        supports_deletion_signal = False

    with pytest.raises(ValueError, match="trustworthy deletion signal"):
        await reconcile_deletion(
            SourceRepository(),
            workspace_id="ws-x",
            source_system="untrustworthy",
            object_type="Account",
            external_id="001",
            observed_at=_T0,
            adapter_supports_deletion_signal=_UntrustworthyAdapter.supports_deletion_signal,
        )
