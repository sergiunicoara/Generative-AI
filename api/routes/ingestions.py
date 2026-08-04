"""§11 required API: POST /api/v1/ingestions/crm, POST /api/v1/ingestions/
content-assets, GET /api/v1/ingestions/{id}.

The API returns an ingestion id rather than holding a request open — true even
though this MVP's pipeline currently runs synchronously in-process within the
request (§11 explicitly permits an in-process bounded worker for the MVP). The
job's state is still recorded through ACCEPTED -> PERSISTING -> COMPLETED /
FAILED_PERMANENT in api/state.py's store, so GET .../{id} works the same way it
would against a real async worker — swapping the synchronous call below for a
queued background task later doesn't change this route's contract.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.dependencies import get_workspace_id
from api.state import IngestionJob, InMemoryIngestionStore
from src.domain.enums import IngestionState
from src.extraction.fixture_provider import FixtureExtractionProvider
from src.graph.execution import GraphExecutor
from src.graph.repositories.claim_repository import ClaimRepository
from src.graph.repositories.content_repository import ContentRepository
from src.graph.repositories.conversation_repository import ConversationRepository
from src.graph.repositories.crm_repository import CrmRepository
from src.graph.repositories.source_repository import SourceRepository
from src.ingestion.adapters.gong import GongAdapter
from src.ingestion.adapters.salesforce import SalesforceAdapter
from src.ingestion.adapters.showpad import ShowpadAdapter
from src.ingestion.pipeline import ContentIngestionPipeline, CrmIngestionPipeline
from src.ingestion.transcript_pipeline import TranscriptIngestionPipeline

router = APIRouter(prefix="/api/v1/ingestions", tags=["ingestions"])
_store = InMemoryIngestionStore()


class CrmIngestionRequest(BaseModel):
    accounts: list[dict] = []
    contacts: list[dict] = []
    leads: list[dict] = []
    opportunities: list[dict] = []


class ContentAssetIngestionRequest(BaseModel):
    division_id: str | None = None
    content_assets: list[dict] = []


class TranscriptIngestionRequest(BaseModel):
    calls: list[dict] = []
    opportunity_id: str | None = None
    account_id: str | None = None
    email_to_contact_id: dict[str, str] = {}
    email_to_seller_id: dict[str, str] = {}


@router.post("/crm", status_code=202)
async def ingest_crm(body: CrmIngestionRequest, workspace_id: str = Depends(get_workspace_id)) -> dict:
    ingestion_id = str(uuid4())
    now = datetime.now(timezone.utc)
    job = IngestionJob(
        ingestion_id=ingestion_id, workspace_id=workspace_id, kind="crm",
        state=IngestionState.ACCEPTED, created_at=now, updated_at=now,
    )
    _store.put(job)

    executor = GraphExecutor()
    pipeline = CrmIngestionPipeline(CrmRepository(executor), SourceRepository(executor), SalesforceAdapter())

    job.state = IngestionState.PERSISTING
    try:
        results = []
        results += await pipeline.ingest_accounts(
            workspace_id, body.accounts, ingestion_run_id=ingestion_id, observed_at=now
        )
        results += await pipeline.ingest_contacts(
            workspace_id, body.contacts, ingestion_run_id=ingestion_id, observed_at=now
        )
        results += await pipeline.ingest_leads(
            workspace_id, body.leads, ingestion_run_id=ingestion_id, observed_at=now
        )
        results += await pipeline.ingest_opportunities(
            workspace_id, body.opportunities, ingestion_run_id=ingestion_id, observed_at=now
        )
        job.item_results = [asdict(r) for r in results]
        job.state = IngestionState.COMPLETED
    except Exception as exc:
        job.state = IngestionState.FAILED_PERMANENT
        job.error = str(exc)
    job.updated_at = datetime.now(timezone.utc)
    _store.put(job)

    return {"ingestion_id": ingestion_id, "state": job.state.value}


@router.post("/content-assets", status_code=202)
async def ingest_content_assets(
    body: ContentAssetIngestionRequest, workspace_id: str = Depends(get_workspace_id)
) -> dict:
    ingestion_id = str(uuid4())
    now = datetime.now(timezone.utc)
    job = IngestionJob(
        ingestion_id=ingestion_id, workspace_id=workspace_id, kind="content-assets",
        state=IngestionState.ACCEPTED, created_at=now, updated_at=now,
    )
    _store.put(job)

    executor = GraphExecutor()
    pipeline = ContentIngestionPipeline(ContentRepository(executor), SourceRepository(executor), ShowpadAdapter())

    job.state = IngestionState.PERSISTING
    try:
        results = await pipeline.ingest_content_assets(
            workspace_id, body.content_assets,
            division_id=body.division_id, ingestion_run_id=ingestion_id, observed_at=now,
        )
        job.item_results = [asdict(r) for r in results]
        job.state = IngestionState.COMPLETED
    except Exception as exc:
        job.state = IngestionState.FAILED_PERMANENT
        job.error = str(exc)
    job.updated_at = datetime.now(timezone.utc)
    _store.put(job)

    return {"ingestion_id": ingestion_id, "state": job.state.value}


@router.post("/transcripts", status_code=202)
async def ingest_transcripts(
    body: TranscriptIngestionRequest, workspace_id: str = Depends(get_workspace_id)
) -> dict:
    ingestion_id = str(uuid4())
    now = datetime.now(timezone.utc)
    job = IngestionJob(
        ingestion_id=ingestion_id, workspace_id=workspace_id, kind="transcripts",
        state=IngestionState.ACCEPTED, created_at=now, updated_at=now,
    )
    _store.put(job)

    executor = GraphExecutor()
    # Fixture extractor by default — pyproject.toml's own open item notes no
    # LLM provider is pinned yet. Swapping in LlmExtractionProvider later
    # doesn't change this route, only which provider is constructed here.
    pipeline = TranscriptIngestionPipeline(
        ConversationRepository(executor), SourceRepository(executor), ClaimRepository(executor),
        GongAdapter(), FixtureExtractionProvider(),
    )

    job.state = IngestionState.EXTRACTING
    try:
        results = []
        for raw_call in body.calls:
            result = await pipeline.ingest_call(
                workspace_id, raw_call,
                ingestion_run_id=ingestion_id, observed_at=now,
                opportunity_id=body.opportunity_id, account_id=body.account_id,
                email_to_contact_id=body.email_to_contact_id, email_to_seller_id=body.email_to_seller_id,
            )
            results.append({
                "conversation_id": result.conversation_id,
                "outcome": result.outcome.value,
                "claims_created": result.claims_created,
            })
        job.item_results = results
        job.state = IngestionState.COMPLETED
    except Exception as exc:
        job.state = IngestionState.FAILED_PERMANENT
        job.error = str(exc)
    job.updated_at = datetime.now(timezone.utc)
    _store.put(job)

    return {"ingestion_id": ingestion_id, "state": job.state.value}


@router.get("/{ingestion_id}")
async def get_ingestion(ingestion_id: str, workspace_id: str = Depends(get_workspace_id)) -> dict:
    job = _store.get(ingestion_id)
    if job is None or job.workspace_id != workspace_id:
        # Same 404 whether the job never existed or belongs to another
        # workspace — a 403 would confirm the id's existence to a caller who
        # doesn't own it.
        raise HTTPException(status_code=404, detail="ingestion not found")
    return {
        "ingestion_id": job.ingestion_id,
        "kind": job.kind,
        "state": job.state.value,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "item_results": job.item_results,
        "error": job.error,
    }
