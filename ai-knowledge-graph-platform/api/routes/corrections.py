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
GET  /corrections/quarantine/records    Records refused by the publication gate
GET  /corrections/quarantine/summary    Quarantine counts by status, rule, source
POST /corrections/quarantine/records/{id}/retry   Re-validate (corrected) record, publish if valid
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_tenant, require_scope
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.entity_splitter import EntitySplitter
from graphrag.graph.quarantine import QuarantineService
from graphrag.graph.contradiction_detector import ContradictionDetector
from graphrag.graph.corpus_revision import CorpusMutation
from graphrag.graph.validation import PublicationGate
from graphrag.graph.invalidation import EntityRef, EventKind, InvalidationEvent, RelationRef, emit

router = APIRouter()


# ── Request / Response models ──────────────────────────────────────────────────

class EntitySplitRequest(BaseModel):
    entity_name: str
    entity_type: str
    doc_group_a: list[str]   # doc IDs for Entity_A
    doc_group_b: list[str]   # doc IDs for Entity_B
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


class QuarantineRetryRequest(BaseModel):
    # Corrected record payload; omit to retry the stored payload unchanged
    # (e.g. after the ontology or an endpoint entity was fixed).
    payload: dict | None = None


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
async def split_entity(request: EntitySplitRequest, tenant: str = Depends(get_tenant)):
    """
    Splits entity_name into Entity_A (backed by doc_group_a) and
    Entity_B (backed by doc_group_b). Redistributes MENTIONS and
    RELATES_TO edges accordingly. Marks original as status=split.
    """
    if not request.doc_group_a or not request.doc_group_b:
        raise HTTPException(status_code=400, detail="Both doc groups must be non-empty")

    neo4j = get_neo4j()
    splitter = EntitySplitter(neo4j)
    async with CorpusMutation(neo4j, tenant, "manual_entity_split") as mutation:
        result = await splitter.split_entity(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            doc_group_a=request.doc_group_a,
            doc_group_b=request.doc_group_b,
            tenant=tenant,
            split_by=request.reviewed_by,
        )
        if "error" not in result:
            result["invalidation"] = await emit(InvalidationEvent(
                tenant=tenant, kind=EventKind.ER_REVISED, additive=True, reason="entity split",
                actor=request.reviewed_by, cause=f"split:{uuid4()}",
                entities=[EntityRef(name=request.entity_name, type=request.entity_type)],
                document_ids=[*request.doc_group_a, *request.doc_group_b],
            ), neo4j)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    result["corpus_revision"] = mutation.revision
    return result


@router.post(
    "/entity/quarantine",
    dependencies=[Depends(require_scope("write"))],
    summary="Quarantine a suspicious entity (excludes it from retrieval)",
)
async def quarantine_entity(request: QuarantineRequest, tenant: str = Depends(get_tenant)):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    if request.propagate_depth > 0:
        async with CorpusMutation(neo4j, tenant, "manual_quarantine_subgraph") as mutation:
            count = await svc.quarantine_subgraph_from(
                seed_entity_name=request.entity_name,
                seed_entity_type=request.entity_type,
                reason=request.reason,
                flagged_by=request.flagged_by,
                depth=request.propagate_depth,
                tenant=tenant,
            )
            await emit(InvalidationEvent(
                tenant=tenant, kind=EventKind.FACT_CORRECTED, additive=True,
                reason=f"subgraph quarantine: {request.reason}", actor=request.flagged_by,
                cause=f"quarantine-subgraph:{uuid4()}",
                entities=[EntityRef(name=request.entity_name, type=request.entity_type)],
            ), neo4j)
        return {"quarantined_count": count, "mode": "subgraph", "corpus_revision": mutation.revision}
    else:
        # Removing one entity only affects what was built on it: targeted invalidation.
        async with CorpusMutation(neo4j, tenant, "manual_quarantine", advance_revision=False) as mutation:
            await svc.quarantine_entity(
                entity_name=request.entity_name,
                entity_type=request.entity_type,
                reason=request.reason,
                flagged_by=request.flagged_by,
                tenant=tenant,
            )
            invalidation = await emit(InvalidationEvent(
                tenant=tenant, kind=EventKind.FACT_CORRECTED, reason=f"quarantine: {request.reason}",
                actor=request.flagged_by, cause=f"quarantine:{uuid4()}",
                entities=[EntityRef(name=request.entity_name, type=request.entity_type)],
            ), neo4j)
        return {"quarantined_count": 1, "mode": "single", "corpus_revision": mutation.revision,
                "invalidation": invalidation}


@router.post(
    "/entity/release",
    dependencies=[Depends(require_scope("write"))],
    summary="Release a quarantined entity back into active retrieval",
)
async def release_entity(request: ReleaseRequest, tenant: str = Depends(get_tenant)):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    async with CorpusMutation(neo4j, tenant, "manual_quarantine_release") as mutation:
        await svc.release(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            released_by=request.released_by,
            note=request.note,
            tenant=tenant,
        )
        await emit(InvalidationEvent(
            tenant=tenant, kind=EventKind.FACT_CORRECTED, additive=True, reason="quarantine released",
            actor=request.released_by, cause=f"release:{uuid4()}",
            entities=[EntityRef(name=request.entity_name, type=request.entity_type)],
        ), neo4j)
    return {"status": "released", "entity": request.entity_name, "corpus_revision": mutation.revision}


# ── Edge corrections ───────────────────────────────────────────────────────────

@router.post(
    "/edge/reject",
    dependencies=[Depends(require_scope("write"))],
    summary="Delete a specific RELATES_TO edge",
)
async def reject_edge(request: EdgeRejectRequest, tenant: str = Depends(get_tenant)):
    """
    Deletes the specified edge and logs the deletion to AuditTrail.
    """
    neo4j = get_neo4j()
    # Deleting an edge only affects what was built on it: targeted invalidation.
    async with CorpusMutation(neo4j, tenant, "manual_edge_reject", advance_revision=False) as mutation:
        rows = await neo4j.run(
            """
            MATCH (s:Entity {name: $src, tenant: $tenant})-[r:RELATES_TO {relation: $rel, tenant: $tenant}]->(t:Entity {name: $tgt, tenant: $tenant})
            WITH r, s, t,
                 r.confidence AS old_conf, r.source_doc_id AS old_doc,
                 s.type AS src_type, t.type AS tgt_type
            DELETE r
            RETURN count(r) AS deleted,
                   old_conf AS confidence,
                   old_doc  AS source_doc_id,
                   collect(DISTINCT {src_type: src_type, tgt_type: tgt_type}) AS endpoint_types
            """,
            src=request.src_entity,
            tgt=request.tgt_entity,
            rel=request.relation,
            tenant=tenant,
        )
        removed = [ep for row in rows for ep in row.get("endpoint_types") or []]
        if removed:
            await emit(InvalidationEvent(
                tenant=tenant, kind=EventKind.RELATION_CHANGED, reason="edge rejected",
                actor=request.rejected_by, cause=f"edge-reject:{uuid4()}",
                relations=[RelationRef(src_name=request.src_entity, src_type=ep["src_type"],
                                       relation=request.relation, tgt_name=request.tgt_entity,
                                       tgt_type=ep["tgt_type"]) for ep in removed],
            ), neo4j)
    deleted = sum(r["deleted"] for r in rows) if rows else 0
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
        tenant=tenant,
    )
    return {"status": "deleted", "edges_removed": deleted, "corpus_revision": mutation.revision}


@router.post(
    "/edge/override",
    dependencies=[Depends(require_scope("write"))],
    summary="Create a MANUAL source_type override edge",
)
async def override_edge(request: EdgeOverrideRequest, tenant: str = Depends(get_tenant)):
    """
    Creates or updates a RELATES_TO edge with source_type=manual and
    high confidence, overriding any LLM-extracted version.
    """
    from datetime import datetime, timezone
    neo4j = get_neo4j()
    async with CorpusMutation(neo4j, tenant, "manual_edge_override") as mutation:
        overridden = await neo4j.run(
            """
            MATCH (s:Entity {name: $src, tenant: $tenant})
            MATCH (t:Entity {name: $tgt, tenant: $tenant})
            MERGE (s)-[r:RELATES_TO {relation: $rel, tenant: $tenant}]->(t)
            SET r.confidence   = $confidence,
                r.source_type  = 'manual',
                r.override_by  = $override_by,
                r.override_note = $note,
                r.extracted_at = $now,
                r.origin       = 'MANUAL',
                r.verification_status = 'VERIFIED',
                r.verified_by  = $override_by,
                r.verified_at  = datetime()
            RETURN s.type AS src_type, t.type AS tgt_type
            """,
            src=request.src_entity,
            tgt=request.tgt_entity,
            rel=request.relation,
            confidence=request.confidence,
            override_by=request.override_by,
            note=request.note,
            now=datetime.now(timezone.utc).isoformat(),
            tenant=tenant,
        )
        if overridden:
            await emit(InvalidationEvent(
                tenant=tenant, kind=EventKind.RELATION_CHANGED, additive=True, reason="manual override",
                actor=request.override_by, cause=f"edge-override:{uuid4()}",
                relations=[RelationRef(src_name=request.src_entity, src_type=row["src_type"],
                                       relation=request.relation, tgt_name=request.tgt_entity,
                                       tgt_type=row["tgt_type"]) for row in overridden],
            ), neo4j)
    return {
        "status": "override_applied",
        "edge": f"({request.src_entity})-[{request.relation}]->({request.tgt_entity})",
        "source_type": "manual",
        "corpus_revision": mutation.revision,
    }


# ── Conflict resolution ────────────────────────────────────────────────────────

@router.post(
    "/conflict/resolve",
    dependencies=[Depends(require_scope("write"))],
    summary="Resolve a detected semantic contradiction",
)
async def resolve_conflict(request: ConflictResolveRequest, tenant: str = Depends(get_tenant)):
    valid_resolutions = {"resolved_manual", "resolved_authority", "false_positive"}
    if request.resolution not in valid_resolutions:
        raise HTTPException(
            status_code=400,
            detail=f"resolution must be one of {valid_resolutions}",
        )
    neo4j = get_neo4j()
    detector = ContradictionDetector(neo4j)
    async with CorpusMutation(neo4j, tenant, "manual_conflict_resolution") as mutation:
        await detector.resolve(
            conflict_id=request.conflict_id,
            resolution=request.resolution,
            winner_doc_id=request.winner_doc_id,
            resolved_by=request.resolved_by,
            tenant=tenant,
        )
    return {"status": request.resolution, "conflict_id": request.conflict_id, "corpus_revision": mutation.revision}


# ── Read endpoints ─────────────────────────────────────────────────────────────

@router.get(
    "/conflicts",
    dependencies=[Depends(require_scope("read"))],
    summary="List open semantic contradictions awaiting review",
)
async def list_conflicts(limit: int = 50, tenant: str = Depends(get_tenant)):
    neo4j = get_neo4j()
    detector = ContradictionDetector(neo4j)
    return await detector.get_open_conflicts(limit=limit, tenant=tenant)


@router.get(
    "/quarantined",
    dependencies=[Depends(require_scope("read"))],
    summary="List currently quarantined entities",
)
async def list_quarantined(limit: int = 100, tenant: str = Depends(get_tenant)):
    neo4j = get_neo4j()
    svc = QuarantineService(neo4j)
    return await svc.list_quarantined(limit=limit, tenant=tenant)


@router.get(
    "/over-merges",
    dependencies=[Depends(require_scope("read"))],
    summary="List entities that are candidates for splitting (over-merged)",
)
async def list_over_merges(top_n: int = 20, tenant: str = Depends(get_tenant)):
    neo4j = get_neo4j()
    splitter = EntitySplitter(neo4j)
    return await splitter.detect_over_merges(top_n=top_n, tenant=tenant)


# ── Publication-gate quarantine ───────────────────────────────────────────────

@router.get(
    "/quarantine/records",
    dependencies=[Depends(require_scope("read"))],
    summary="List records refused by the publication gate",
)
async def list_quarantine_records(
    status: str | None = "QUARANTINED",
    rule_id: str | None = None,
    limit: int = 100,
    tenant: str = Depends(get_tenant),
):
    gate = PublicationGate(get_neo4j())
    return await gate.store.list(tenant=tenant, status=status, rule_id=rule_id, limit=limit)


@router.get(
    "/quarantine/summary",
    dependencies=[Depends(require_scope("read"))],
    summary="Quarantine counts by status, rule and source",
)
async def quarantine_summary(tenant: str = Depends(get_tenant)):
    return await PublicationGate(get_neo4j()).store.summary(tenant=tenant)


@router.post(
    "/quarantine/records/{record_id}/retry",
    dependencies=[Depends(require_scope("write"))],
    summary="Re-validate a quarantined record and publish it if it now passes",
)
async def retry_quarantine_record(
    record_id: str,
    request: QuarantineRetryRequest,
    tenant: str = Depends(get_tenant),
):
    from graphrag.ingestion.graph_writer import GraphWriter

    neo4j = get_neo4j()
    writer = GraphWriter(changed_by="quarantine_retry", neo4j_client=neo4j)
    await writer.ensure_ontology_schema(tenant)
    gate = PublicationGate(neo4j)
    try:
        return await gate.retry(
            tenant=tenant, record_id=record_id, writer=writer,
            corrected=request.payload, registry=getattr(writer, "_ontology", None),
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="quarantined record not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get(
    "/conflict/{conflict_id}/suggestion",
    dependencies=[Depends(require_scope("read"))],
    summary="Trust-ranked competing sources for a conflict (suggestion only, nothing is applied)",
)
async def suggest_conflict_resolution(conflict_id: str, tenant: str = Depends(get_tenant)):
    out = await ContradictionDetector(get_neo4j()).suggest_resolution(conflict_id, tenant=tenant)
    if out["status"] == "not_found":
        raise HTTPException(status_code=404, detail="conflict not found")
    return out


# ── Targeted invalidation (docs/invalidation.md) ─────────────────────────────

@router.get(
    "/invalidation/artifacts",
    dependencies=[Depends(require_scope("read"))],
    summary="Derived artifacts by invalidation state (NEEDS_REVIEW, RECOMPUTING, ...)",
)
async def list_invalidated_artifacts(
    state: str | None = "NEEDS_REVIEW", kind: str | None = None, limit: int = 100,
    tenant: str = Depends(get_tenant),
):
    from graphrag.graph.invalidation.state_store import StateStore
    return await StateStore(get_neo4j()).list(tenant, state=state, kind=kind, limit=limit)


@router.get(
    "/invalidation/artifacts/{kind}/{artifact_id:path}",
    dependencies=[Depends(require_scope("read"))],
    summary="One artifact's state and the invalidation events that explain it",
)
async def get_invalidated_artifact(kind: str, artifact_id: str, tenant: str = Depends(get_tenant)):
    from graphrag.graph.invalidation.state_store import StateStore
    row = await StateStore(get_neo4j()).get(tenant, kind, artifact_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no invalidation state for this artifact")
    return row


@router.post(
    "/invalidation/recompute",
    dependencies=[Depends(require_scope("write"))],
    summary="Recompute artifacts waiting in NEEDS_REVIEW (bounded batch)",
)
async def recompute_invalidated(limit: int = 50, tenant: str = Depends(get_tenant)):
    from graphrag.graph.invalidation.recompute import RecomputeWorker
    return await RecomputeWorker(get_neo4j()).run_once(tenant, limit=max(1, min(limit, 500)))
