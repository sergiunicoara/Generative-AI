"""Enterprise content governance, sync and lineage API surface."""

from __future__ import annotations

from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth.dependencies import (
    assert_request_tenant,
    get_current_user,
    get_tenant,
    require_scope,
)
from graphrag.enterprise.lineage import LineageService
from graphrag.enterprise.metadata_governance import MetadataGovernanceService
from graphrag.enterprise.models import CollectionSchema, SyncChange
from graphrag.enterprise.sync import ContentSyncService
from graphrag.enterprise.sharepoint import SharePointSyncConnector

router = APIRouter()


class SyncChangesRequest(BaseModel):
    changes: list[SyncChange] = Field(default_factory=list, max_length=1_000)
    cursor: str = Field(default="", max_length=2_048)
    trigger: Literal["webhook", "delta", "manual"] = "delta"


class ReconciliationRequest(BaseModel):
    discovered_external_ids: list[str] = Field(default_factory=list, max_length=100_000)


@router.post("/governance/schemas", dependencies=[Depends(require_scope("write"))])
async def register_metadata_schema(
    schema: CollectionSchema, tenant: str = Depends(get_tenant),
):
    assert_request_tenant(schema.tenant, tenant)
    return await MetadataGovernanceService().register_schema(schema.model_copy(update={"tenant": tenant}))


@router.get("/governance/coverage", dependencies=[Depends(require_scope("read"))])
async def metadata_coverage(
    collection: str | None = None, tenant: str = Depends(get_tenant),
):
    return {"collections": await MetadataGovernanceService().coverage(tenant, collection)}


@router.post("/sync/{source_id}/changes", dependencies=[Depends(require_scope("write"))])
async def apply_sync_changes(
    source_id: str,
    body: SyncChangesRequest,
    tenant: str = Depends(get_tenant),
):
    try:
        return await ContentSyncService().apply_changes(
            source_id, body.changes, tenant, cursor=body.cursor, trigger=body.trigger,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sync/{source_id}/reconcile", dependencies=[Depends(require_scope("write"))])
async def reconcile_sync_source(
    source_id: str,
    body: ReconciliationRequest,
    tenant: str = Depends(get_tenant),
):
    return await ContentSyncService().reconcile(source_id, body.discovered_external_ids, tenant)


@router.get("/sync/sources", dependencies=[Depends(require_scope("read"))])
async def list_sync_sources(tenant: str = Depends(get_tenant)):
    return {"sources": await ContentSyncService().sources(tenant)}


@router.get("/sync/due-full-reviews", dependencies=[Depends(require_scope("read"))])
async def due_full_reviews(tenant: str = Depends(get_tenant)):
    return {"sources": await ContentSyncService().due_full_reviews(tenant)}


@router.post("/sync/sharepoint/{source_id}/run", dependencies=[Depends(require_scope("write"))])
async def sync_sharepoint_source(source_id: str, tenant: str = Depends(get_tenant)):
    connector = SharePointSyncConnector.from_settings(source_id)
    if connector.config.tenant != tenant:
        raise HTTPException(status_code=403, detail="SharePoint source does not match authenticated tenant")
    try:
        return await connector.sync_once()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="SharePoint synchronization failed") from exc


@router.get("/lineage/reviews", dependencies=[Depends(require_scope("read"))])
async def lineage_reviews(
    kind: Literal["lineage", "obligation"] = "lineage",
    status: Literal["pending", "approved", "rejected", "all"] = "pending",
    tenant: str = Depends(get_tenant),
):
    return {"reviews": await LineageService().list_reviews(tenant, kind, status)}


@router.post("/lineage/reviews/{review_id}/approve", dependencies=[Depends(require_scope("write"))])
async def approve_lineage_review(
    review_id: str,
    kind: Literal["lineage", "obligation"] = "lineage",
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
):
    service = LineageService()
    reviewer = str(user.get("sub") or "unknown")
    return (
        await service.approve_lineage(review_id, reviewer, tenant)
        if kind == "lineage"
        else await service.approve_obligation(review_id, reviewer, tenant)
    )


@router.post("/lineage/reviews/{review_id}/reject", dependencies=[Depends(require_scope("write"))])
async def reject_lineage_review(
    review_id: str,
    kind: Literal["lineage", "obligation"] = "lineage",
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
):
    service = LineageService()
    reviewer = str(user.get("sub") or "unknown")
    return (
        await service.reject_lineage(review_id, reviewer, tenant)
        if kind == "lineage"
        else await service.reject_obligation(review_id, reviewer, tenant)
    )


@router.get("/obligations", dependencies=[Depends(require_scope("read"))])
async def obligations_register(
    as_of: str | None = None, tenant: str = Depends(get_tenant),
):
    return {"obligations": await LineageService().obligations(tenant, as_of)}
