"""POST /corrections — human correction loop for graph review and repair.

Endpoints
---------
POST /corrections/entity/split          Split an over-merged entity
POST /corrections/entity/quarantine     Quarantine a suspicious entity
POST /corrections/entity/release        Release a quarantined entity
POST /corrections/edge/reject           Delete or quarantine a specific edge
POST /corrections/edge/override         Create a MANUAL source_type override edge
POST /corrections/conflict/resolve      Mark a Conflict node as resolved
GET  /corrections/conflicts             List open conflicts
GET  /corrections/quarantined           List quarantined entities
GET  /corrections/over-merges          List over-merge candidates
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.entity_splitter import EntitySplitter
from graphrag.graph.quarantine import QuarantineService
from graphrag.graph.contradiction_detector import ContradictionDetector

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class EntitySplitRequest(BaseModel):
    entity_name: str
    entity_type: str
    doc_group_a: list[str]   # doc IDs for Entity_A
    doc_group_b: list[str]   # doc IDs for Entity_B
    tenant: str = "default"
    reviewed_by: str = "admin"


class QuarantineRequest(BaseModel):
    entity_name: str
    entity_type: str
    reason: str
    flagged_by: str = "admin"
    propagate_depth: int = 0   # 0 = single entity only, >0 = subgraph


class ReleaseRequest(BaseModel):
    entity_name: str
    entity_type: str
    released_by: str
    note: str = ""


class EdgeRejectRequest(BaseModel):
    src_entity: str
    tgt_entity: str
    relation: str
    rejected_by: str = "admin"


class EdgeOverrideRequest(BaseModel):
    src_entity: str
    tgt_entity: str
    relation: str
    confidence: float = 1.0
    override_by: str = "admin"
    note: str = ""


class ConflictResolveRequest(BaseModel):
    conflict_id: str
    resolution: str             # "resolved_manual" | "false_positive"
    winner_doc_id: str = ""
    resolved_by: str = "admin"


# ── Entity corrections ─────────────────────────────────────────────────────────

@router.post(
    "/entity/split",
    dependencies=[Depends(require_scope("write"))],
    summary="Split an over-merged entity into two separate nodes",
)
async def split_entity(request: EntitySplitRequest):
    """
    Splits entity_name into Entity_A (backed by doc_group_a) and
    Entity_B (backed by doc_group_b). Redistributes MENTIONS and
    RELATES_TO edges accordingly. Marks original as status=split.
    """
    if not request.doc_group_a or not request.doc_group_b:
        raise HTTPException(status_code=400, detail="Both doc groups must be non-empty")

    neo4j = get_neo4j()
    splitter = EntitySplitter(neo4j)
    result = await splitter.split_entity(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        doc_group_a=request.doc_group_a,
        doc_group_b=request.doc_group_b,
        tenant=request.tenant,
        split_by=request.reviewed_by,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post(
    "/entity/quarantine",
    dependencies=[Depends(require_scope("write"))],
    summary="Quarantine a suspicious entity (excludes it from retrieval)",
)
async def quarantine_entity(request: QuarantineRequest):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    if request.propagate_depth > 0:
        count = await svc.quarantine_subgraph_from(
            seed_entity_name=request.entity_name,
            seed_entity_type=request.entity_type,
            reason=request.reason,
            flagged_by=request.flagged_by,
            depth=request.propagate_depth,
        )
        return {"quarantined_count": count, "mode": "subgraph"}
    else:
        await svc.quarantine_entity(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            reason=request.reason,
            flagged_by=request.flagged_by,
        )
        return {"quarantined_count": 1, "mode": "single"}


@router.post(
    "/entity/release",
    dependencies=[Depends(require_scope("write"))],
    summary="Release a quarantined entity back into active retrieval",
)
async def release_entity(request: ReleaseRequest):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    await svc.release(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        released_by=request.released_by,
        note=request.note,
    )
    return {"status": "released", "entity": request.entity_name}


# ── Edge corrections ───────────────────────────────────────────────────────────

@router.post(
    "/edge/reject",
    dependencies=[Depends(require_scope("write"))],
    summary="Delete a specific RELATES_TO edge",
)
async def reject_edge(request: EdgeRejectRequest):
    """
    Deletes the specified edge and logs the deletion to AuditTrail.
    """
    neo4j = get_neo4j()
    rows = await neo4j.run(
        """
        MATCH (s:Entity {name: $src})-[r:RELATES_TO {relation: $rel}]->(t:Entity {name: $tgt})
        WITH r, s, t,
             r.confidence AS old_conf, r.source_doc_id AS old_doc
        DELETE r
        RETURN count(r) AS deleted,
               old_conf AS confidence,
               old_doc  AS source_doc_id
        """,
        src=request.src_entity,
        tgt=request.tgt_entity,
        rel=request.relation,
    )
    deleted = rows[0]["deleted"] if rows else 0
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Edge ({request.src_entity})-[{request.relation}]->({request.tgt_entity}) not found",
        )

    # Audit log
    from graphrag.graph.audit_trail import AuditTrail
    audit = AuditTrail(neo4j)
    await audit.log_relation_change(
        src_name=request.src_entity,
        tgt_name=request.tgt_entity,
        relation=request.relation,
        operation="delete",
        old_values={"confidence": rows[0].get("confidence"), "source_doc_id": rows[0].get("source_doc_id")},
        changed_by=request.rejected_by,
    )
    return {"status": "deleted", "edges_removed": deleted}


@router.post(
    "/edge/override",
    dependencies=[Depends(require_scope("write"))],
    summary="Create a MANUAL source_type override edge",
)
async def override_edge(request: EdgeOverrideRequest):
    """
    Creates or updates a RELATES_TO edge with source_type=manual and
    high confidence, overriding any LLM-extracted version.
    """
    from datetime import datetime, timezone
    neo4j = get_neo4j()
    await neo4j.run(
        """
        MATCH (s:Entity {name: $src})
        MATCH (t:Entity {name: $tgt})
        MERGE (s)-[r:RELATES_TO {relation: $rel}]->(t)
        SET r.confidence   = $confidence,
            r.source_type  = 'manual',
            r.override_by  = $override_by,
            r.override_note = $note,
            r.extracted_at = $now
        """,
        src=request.src_entity,
        tgt=request.tgt_entity,
        rel=request.relation,
        confidence=request.confidence,
        override_by=request.override_by,
        note=request.note,
        now=datetime.now(timezone.utc).isoformat(),
    )
    return {
        "status": "override_applied",
        "edge": f"({request.src_entity})-[{request.relation}]->({request.tgt_entity})",
        "source_type": "manual",
    }


# ── Conflict resolution ────────────────────────────────────────────────────────

@router.post(
    "/conflict/resolve",
    dependencies=[Depends(require_scope("write"))],
    summary="Resolve a detected semantic contradiction",
)
async def resolve_conflict(request: ConflictResolveRequest):
    valid_resolutions = {"resolved_manual", "resolved_authority", "false_positive"}
    if request.resolution not in valid_resolutions:
        raise HTTPException(
            status_code=400,
            detail=f"resolution must be one of {valid_resolutions}",
        )
    neo4j = get_neo4j()
    detector = ContradictionDetector(neo4j)
    await detector.resolve(
        conflict_id=request.conflict_id,
        resolution=request.resolution,
        winner_doc_id=request.winner_doc_id,
        resolved_by=request.resolved_by,
    )
    return {"status": request.resolution, "conflict_id": request.conflict_id}


# ── Read endpoints ─────────────────────────────────────────────────────────────

@router.get(
    "/conflicts",
    dependencies=[Depends(require_scope("read"))],
    summary="List open semantic contradictions awaiting review",
)
async def list_conflicts(limit: int = 50, tenant: str | None = None):
neo4j = get_neo4j()
    detector = ContradictionDetector(neo4j)
    return await detector.get_open_conflicts(limit=limit, tenant=tenant)


@router.get(
    "/quarantined",
    dependencies=[Depends(require_scope("read"))],
    summary="List currently quarantined entities",
)
async def list_quarantined(limit: int = 100):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    return await svc.list_quarantined(limit=limit)


@router.get(
    "/over-merges",
    dependencies=[Depends(require_scope("read"))],
    summary="List entities that are candidates for splitting (over-merged)",
)
async def list_over_merges(top_n: int = 20, tenant: str = "default"):
    neo4j = get_neo4j()
    splitter = EntitySplitter(neo4j)
    return await splitter.detect_over_merges(top_n=top_n, tenant=tenant)
