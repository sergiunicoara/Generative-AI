"""Community manager and incremental community detection endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── Incremental Community Detection ──────────────────────────────────────────

class IncrementalRebuildRequest(BaseModel):
    tenant: str = "default"
    dry_run: bool = False


@router.get(
    "/incremental-community/summary",
    dependencies=[Depends(require_scope("read"))],
    summary="Show how many entities changed since the last community build",
)
async def incremental_community_summary(tenant: str = "default"):
    from graphrag.graph.incremental_community import IncrementalCommunityDetector
    detector = IncrementalCommunityDetector(get_neo4j())
    return await detector.community_change_summary(tenant=tenant)


@router.post(
    "/incremental-community/rebuild-affected",
    dependencies=[Depends(require_scope("write"))],
    summary="Rebuild only communities containing recently changed entities",
)
async def incremental_rebuild_affected(request: IncrementalRebuildRequest):
    from graphrag.graph.incremental_community import IncrementalCommunityDetector
    detector = IncrementalCommunityDetector(get_neo4j())
    return await detector.rebuild_affected_communities(
        tenant=request.tenant,
        dry_run=request.dry_run,
    )


@router.post(
    "/incremental-community/record-rebuild-point",
    dependencies=[Depends(require_scope("write"))],
    summary="Manually record a community rebuild point (normally set automatically)",
)
async def record_rebuild_point(tenant: str = "default"):
    from graphrag.graph.incremental_community import IncrementalCommunityDetector
    detector = IncrementalCommunityDetector(get_neo4j())
    rp_id = await detector.record_rebuild_point(tenant=tenant)
    return {"rebuild_point_id": rp_id, "tenant": tenant}


# ── HDBSCAN Semantic Community Detection ─────────────────────────────────────

@router.post(
    "/semantic-communities/build",
    dependencies=[Depends(require_scope("write"))],
    summary="Build HDBSCAN semantic communities as a parallel signal to Leiden",
)
async def build_semantic_communities(tenant: str = "default"):
    from graphrag.graph.community_builder import CommunityBuilder
    builder = CommunityBuilder(tenant=tenant)
    return await builder.build_semantic_communities()
