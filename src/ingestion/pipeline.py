"""§11 — the NORMALIZING -> PERSISTING portion of the ingestion state machine, for
CRM and content-asset sources specifically (transcript-specific EXTRACTING/
RESOLVING states are Increment 5/6's src/extraction and src/resolution).

CrmIngestionPipeline is adapter-agnostic — it depends only on the adapter's
`parse_*`/`supports_deletion_signal` surface (structurally, not by inheritance),
so adding a second CRM provider (Dynamics, HubSpot, ...) means writing a new
adapter under src/ingestion/adapters/, never touching this file (§4).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable

from src.graph.repositories.content_repository import ContentRepository
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.base import ParsedRecord
from src.ingestion.reconciliation import (
    ReconciliationOutcome,
    reconcile_deletion,
    reconcile_source_record,
)


@dataclass(frozen=True)
class IngestionItemResult:
    object_type: str
    external_id: str
    outcome: ReconciliationOutcome


class CrmIngestionPipeline:
    def __init__(
        self,
        crm_repo: CrmRepository,
        source_repo: SourceRepository,
        adapter,  # SalesforceAdapter-shaped: parse_account/parse_contact/parse_lead/parse_opportunity, source_system, supports_deletion_signal
    ):
        self._crm_repo = crm_repo
        self._source_repo = source_repo
        self._adapter = adapter

    async def _ingest(
        self,
        workspace_id: str,
        raw_records: list[dict],
        *,
        parse: Callable[[str, dict], ParsedRecord],
        upsert: Callable[..., Awaitable[None]],
        ingestion_run_id: str,
        observed_at: datetime,
    ) -> list[IngestionItemResult]:
        results = []
        for raw in raw_records:
            parsed = parse(workspace_id, raw)
            reconciliation = await reconcile_source_record(
                self._source_repo,
                workspace_id=workspace_id,
                source_system=self._adapter.source_system,
                object_type=parsed.object_type,
                external_id=parsed.external_id,
                content_hash=parsed.content_hash,
                ingestion_run_id=ingestion_run_id,
                observed_at=observed_at,
            )
            if reconciliation.outcome in (ReconciliationOutcome.CREATED, ReconciliationOutcome.SUPERSEDED):
                await upsert(parsed.entity)
            if parsed.extra and parsed.extra.get("is_deleted"):
                reconciliation = await reconcile_deletion(
                    self._source_repo,
                    workspace_id=workspace_id,
                    source_system=self._adapter.source_system,
                    object_type=parsed.object_type,
                    external_id=parsed.external_id,
                    observed_at=observed_at,
                    adapter_supports_deletion_signal=self._adapter.supports_deletion_signal,
                )
            results.append(IngestionItemResult(parsed.object_type, parsed.external_id, reconciliation.outcome))
        return results

    async def ingest_accounts(
        self, workspace_id: str, raw_records: list[dict], *, ingestion_run_id: str, observed_at: datetime
    ) -> list[IngestionItemResult]:
        return await self._ingest(
            workspace_id, raw_records,
            parse=self._adapter.parse_account, upsert=self._crm_repo.upsert_account,
            ingestion_run_id=ingestion_run_id, observed_at=observed_at,
        )

    async def ingest_contacts(
        self, workspace_id: str, raw_records: list[dict], *, ingestion_run_id: str, observed_at: datetime
    ) -> list[IngestionItemResult]:
        return await self._ingest(
            workspace_id, raw_records,
            parse=self._adapter.parse_contact, upsert=self._crm_repo.upsert_contact,
            ingestion_run_id=ingestion_run_id, observed_at=observed_at,
        )

    async def ingest_leads(
        self, workspace_id: str, raw_records: list[dict], *, ingestion_run_id: str, observed_at: datetime
    ) -> list[IngestionItemResult]:
        return await self._ingest(
            workspace_id, raw_records,
            parse=self._adapter.parse_lead, upsert=self._crm_repo.upsert_lead,
            ingestion_run_id=ingestion_run_id, observed_at=observed_at,
        )

    async def ingest_opportunities(
        self, workspace_id: str, raw_records: list[dict], *, ingestion_run_id: str, observed_at: datetime
    ) -> list[IngestionItemResult]:
        return await self._ingest(
            workspace_id, raw_records,
            parse=self._adapter.parse_opportunity, upsert=self._crm_repo.upsert_opportunity,
            ingestion_run_id=ingestion_run_id, observed_at=observed_at,
        )

    async def ingest_deletion(
        self, workspace_id: str, object_type: str, external_id: str, *, observed_at: datetime
    ) -> IngestionItemResult:
        reconciliation = await reconcile_deletion(
            self._source_repo,
            workspace_id=workspace_id,
            source_system=self._adapter.source_system,
            object_type=object_type,
            external_id=external_id,
            observed_at=observed_at,
            adapter_supports_deletion_signal=self._adapter.supports_deletion_signal,
        )
        return IngestionItemResult(object_type, external_id, reconciliation.outcome)


class ContentIngestionPipeline:
    def __init__(self, content_repo: ContentRepository, source_repo: SourceRepository, adapter):
        self._content_repo = content_repo
        self._source_repo = source_repo
        self._adapter = adapter

    async def ingest_content_assets(
        self,
        workspace_id: str,
        raw_records: list[dict],
        *,
        division_id: str | None,
        ingestion_run_id: str,
        observed_at: datetime,
    ) -> list[IngestionItemResult]:
        results = []
        for raw in raw_records:
            parsed = self._adapter.parse_content_asset(workspace_id, raw, division_id=division_id)
            reconciliation = await reconcile_source_record(
                self._source_repo,
                workspace_id=workspace_id,
                source_system=self._adapter.source_system,
                object_type=parsed.object_type,
                external_id=parsed.external_id,
                content_hash=parsed.content_hash,
                ingestion_run_id=ingestion_run_id,
                observed_at=observed_at,
            )
            if reconciliation.outcome in (ReconciliationOutcome.CREATED, ReconciliationOutcome.SUPERSEDED):
                await self._content_repo.upsert_content_asset(parsed.entity)
            results.append(IngestionItemResult(parsed.object_type, parsed.external_id, reconciliation.outcome))
        return results
