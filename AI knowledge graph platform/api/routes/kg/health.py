"""Graph snapshot checkpoints, query cache, and health alert endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── Graph Snapshots ───────────────────────────────────────────────────────────

class SnapshotCreateRequest(BaseModel):
    label: str
    tenant: str = "default"
    include_health: bool = True


@router.post(
    "/snapshots/create",
    dependencies=[Depends(require_scope("write"))],
    summary="Create a named graph snapshot checkpoint",
)
async def create_snapshot(request: SnapshotCreateRequest):
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    svc = GraphSnapshotService(get_neo4j())
    snap_id = await svc.create_snapshot(
        label=request.label,
        tenant=request.tenant,
        include_health=request.include_health,
    )
    return {"snap_id": snap_id, "label": request.label}


@router.get(
    "/snapshots",
    dependencies=[Depends(require_scope("read"))],
    summary="List graph snapshots for a tenant",
)
async def list_snapshots(tenant: str = "default", limit: int = 50):
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    svc = GraphSnapshotService(get_neo4j())
    snapshots = await svc.list_snapshots(tenant=tenant, limit=limit)
    return {"snapshots": snapshots, "tenant": tenant, "count": len(snapshots)}


@router.get(
    "/snapshots/{snap_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Return stored metrics from a specific snapshot",
)
async def get_snapshot(snap_id: str, tenant: str = "default"):
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    svc = GraphSnapshotService(get_neo4j())
    result = await svc.restore_summary(snap_id=snap_id, tenant=tenant)
    if not result:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return result


@router.get(
    "/snapshots/diff/{snap_id_a}/{snap_id_b}",
    dependencies=[Depends(require_scope("read"))],
    summary="Compute statistical delta between two graph snapshots",
)
async def diff_snapshots(snap_id_a: str, snap_id_b: str, tenant: str = "default"):
    from graphrag.graph.graph_snapshots import GraphSnapshotService
    svc = GraphSnapshotService(get_neo4j())
    return await svc.diff_snapshots(
        snap_id_a=snap_id_a,
        snap_id_b=snap_id_b,
        tenant=tenant,
    )


# ── Query Cache ───────────────────────────────────────────────────────────────

class CacheInvalidateRequest(BaseModel):
    entity_names: list[str]
    tenant: str = "default"


@router.post(
    "/cache/invalidate",
    dependencies=[Depends(require_scope("write"))],
    summary="Invalidate cached query results that used any of the given entities",
)
async def cache_invalidate(request: CacheInvalidateRequest):
    from graphrag.retrieval.query_cache import get_query_cache
    cache = await get_query_cache()
    count = await cache.invalidate_for_entities(
        entity_names=request.entity_names,
        tenant=request.tenant,
    )
    return {"invalidated": count, "tenant": request.tenant}


@router.delete(
    "/cache/flush/{tenant}",
    dependencies=[Depends(require_scope("write"))],
    summary="Remove all cached results for a tenant (use after bulk re-ingestion)",
)
async def cache_flush_tenant(tenant: str):
    from graphrag.retrieval.query_cache import get_query_cache
    cache = await get_query_cache()
    count = await cache.flush_tenant(tenant=tenant)
    return {"flushed": count, "tenant": tenant}


@router.get(
    "/cache/stats",
    dependencies=[Depends(require_scope("read"))],
    summary="Return cache backend statistics",
)
async def cache_stats():
    from graphrag.retrieval.query_cache import get_query_cache
    cache = await get_query_cache()
    return await cache.stats()


# ── Health Alerts ─────────────────────────────────────────────────────────────

@router.get(
    "/health/alerts",
    dependencies=[Depends(require_scope("read"))],
    summary="Return the most recently fired threshold-breach alerts (newest first)",
)
async def get_health_alerts(limit: int = 50):
    """
    Return recently fired GraphHealthSnapshot threshold-breach alerts.

    Alerts are accumulated in memory by AlertService.fire() and keyed to the
    most recent GraphEvaluator.persist_snapshot() call.  The deque is
    process-local — for multi-worker deployments use a shared Redis list.
    """
    from graphrag.monitoring.alerts import get_recent_alerts
    return {"alerts": get_recent_alerts(limit=limit)}
