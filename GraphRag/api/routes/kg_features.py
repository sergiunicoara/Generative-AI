"""POST/GET /kg — Knowledge graph architecture feature endpoints.

Covers the 7 production KG gaps added in Phase 7:
  1. Negative knowledge     (NEGATIVE_RELATES_TO assertions)
  2. Entity type taxonomy   (SUBCLASS_OF hierarchy + type expansion)
  3. Bitemporal queries     (valid-time + transaction-time "as of" queries)
  4. Confidence calibration (Brier score, calibration curve, apply correction)
  5. Relation reification   (Statement nodes + meta-statements)
  6. Edge embeddings        (TransE scoring, link prediction)
  7. Graph snapshots        (named checkpoints + diff)

All write endpoints require `write` scope.
All read endpoints require `read` scope.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# 1. Negative Knowledge
# ══════════════════════════════════════════════════════════════════════════════

class NegativeAssertRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
    tenant: str = "default"
    doc_id: str = ""
    confidence: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None


class NegativeRetractRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
    tenant: str = "default"


@router.post(
    "/negative/assert",
    dependencies=[Depends(require_scope("write"))],
    summary="Assert that a relation does NOT hold between two entities",
)
async def assert_negative(request: NegativeAssertRequest):
    """
    Create a NEGATIVE_RELATES_TO edge asserting the relation is absent.
    Triggers a warning if a positive RELATES_TO edge also exists (conflict).
    """
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    neg_id = await svc.assert_negative(
        src_name=request.src_name,
        src_type=request.src_type,
        relation=request.relation,
        tgt_name=request.tgt_name,
        tgt_type=request.tgt_type,
        tenant=request.tenant,
        doc_id=request.doc_id,
        confidence=request.confidence,
        valid_from=request.valid_from,
        valid_to=request.valid_to,
    )
    return {"status": "asserted", "neg_id": neg_id}


@router.post(
    "/negative/retract",
    dependencies=[Depends(require_scope("write"))],
    summary="Retract a negative assertion (remove NEGATIVE_RELATES_TO edge)",
)
async def retract_negative(request: NegativeRetractRequest):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    deleted = await svc.retract_negative(
        src_name=request.src_name,
        src_type=request.src_type,
        relation=request.relation,
        tgt_name=request.tgt_name,
        tgt_type=request.tgt_type,
        tenant=request.tenant,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Negative assertion not found")
    return {"status": "retracted"}


@router.get(
    "/negative/conflicts",
    dependencies=[Depends(require_scope("read"))],
    summary="Find triples with both positive and negative assertions",
)
async def negative_conflicts(tenant: str | None = None, scan_limit: int = 200):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    return await svc.find_positive_negative_conflicts(tenant=tenant, scan_limit=scan_limit)


@router.get(
    "/negative/list",
    dependencies=[Depends(require_scope("read"))],
    summary="List all NEGATIVE_RELATES_TO edges for a tenant",
)
async def list_negatives(tenant: str = "default", limit: int = 100):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    return await svc.list_all_negatives(tenant=tenant, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Entity Type Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

class SubclassRegisterRequest(BaseModel):
    child: str
    parent: str


@router.post(
    "/taxonomy/register",
    dependencies=[Depends(require_scope("write"))],
    summary="Add a SUBCLASS_OF edge to the entity type hierarchy",
)
async def register_subclass(request: SubclassRegisterRequest):
    from graphrag.graph.type_taxonomy import get_type_taxonomy
    tax = get_type_taxonomy(get_neo4j())
    if not tax._loaded:
        await tax.load()
    await tax.register_subclass(child=request.child, parent=request.parent)
    return {"status": "registered", "child": request.child.upper(), "parent": request.parent.upper()}


@router.get(
    "/taxonomy/expand",
    dependencies=[Depends(require_scope("read"))],
    summary="Return a type and all its subtypes (for query expansion)",
)
async def expand_type(type_name: str):
    from graphrag.graph.type_taxonomy import get_type_taxonomy
    tax = get_type_taxonomy(get_neo4j())
    if not tax._loaded:
        await tax.load()
    return {
        "type": type_name.upper(),
        "expanded": tax.expand_type(type_name),
        "ancestors": tax.get_ancestors(type_name),
    }


@router.get(
    "/taxonomy/schema",
    dependencies=[Depends(require_scope("read"))],
    summary="Return the full SUBCLASS_OF graph",
)
async def taxonomy_schema():
    from graphrag.graph.type_taxonomy import get_type_taxonomy
    tax = get_type_taxonomy(get_neo4j())
    if not tax._loaded:
        await tax.load()
    return await tax.get_schema()


@router.get(
    "/taxonomy/entities",
    dependencies=[Depends(require_scope("read"))],
    summary="Fetch entities by type with subtype expansion",
)
async def entities_by_type(type_name: str, tenant: str = "default",
                            include_subtypes: bool = True, limit: int = 100):
    from graphrag.graph.type_taxonomy import get_type_taxonomy
    tax = get_type_taxonomy(get_neo4j())
    if not tax._loaded:
        await tax.load()
    return await tax.query_by_type(type_name=type_name, tenant=tenant,
                                    include_subtypes=include_subtypes, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Bitemporal Queries
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/bitemporal/entities",
    dependencies=[Depends(require_scope("read"))],
    summary="Return entities valid at (valid_time) as recorded by (transaction_time)",
)
async def bitemporal_entities(
    valid_time: str,
    transaction_time: str,
    tenant: str = "default",
    limit: int = 500,
):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.as_of_entities(valid_time, transaction_time, tenant, limit)


@router.get(
    "/bitemporal/edges",
    dependencies=[Depends(require_scope("read"))],
    summary="Return edges valid at (valid_time) as recorded by (transaction_time)",
)
async def bitemporal_edges(
    valid_time: str,
    transaction_time: str,
    tenant: str = "default",
    limit: int = 1000,
):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.as_of_edges(valid_time, transaction_time, tenant, limit)


@router.get(
    "/bitemporal/diff",
    dependencies=[Depends(require_scope("read"))],
    summary="Count entities and edges added between two transaction times",
)
async def bitemporal_diff(
    tt_from: str,
    tt_to: str,
    tenant: str = "default",
):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.transaction_diff(tt_from, tt_to, tenant)


@router.get(
    "/bitemporal/report",
    dependencies=[Depends(require_scope("read"))],
    summary="High-level graph state at (valid_time, transaction_time)",
)
async def bitemporal_report(valid_time: str, transaction_time: str, tenant: str = "default"):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.time_travel_report(valid_time, transaction_time, tenant)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Confidence Calibration
# ══════════════════════════════════════════════════════════════════════════════

class CalibrationSampleRequest(BaseModel):
    predicted_confidence: float
    actual_outcome: float           # 1.0 = correct, 0.0 = incorrect
    relation: str = ""
    source_doc_id: str = ""
    prompt_version: str = ""
    tenant: str = "default"
    verified_by: str = "admin"


class CalibrationBatchRequest(BaseModel):
    samples: list[CalibrationSampleRequest]
    tenant: str = "default"


@router.post(
    "/calibration/sample",
    dependencies=[Depends(require_scope("write"))],
    summary="Record a calibration data point (predicted vs actual outcome)",
)
async def add_calibration_sample(request: CalibrationSampleRequest):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    sid = await svc.add_sample(
        predicted_confidence=request.predicted_confidence,
        actual_outcome=request.actual_outcome,
        relation=request.relation,
        source_doc_id=request.source_doc_id,
        prompt_version=request.prompt_version,
        tenant=request.tenant,
        verified_by=request.verified_by,
    )
    return {"sample_id": sid}


@router.post(
    "/calibration/batch",
    dependencies=[Depends(require_scope("write"))],
    summary="Bulk-record calibration samples from a golden set",
)
async def add_calibration_batch(request: CalibrationBatchRequest):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    samples = [s.model_dump() for s in request.samples]
    ids = await svc.add_batch(samples, tenant=request.tenant)
    return {"added": len(ids), "sample_ids": ids}


@router.get(
    "/calibration/summary",
    dependencies=[Depends(require_scope("read"))],
    summary="Compute Brier score, calibration curve, and verdict",
)
async def calibration_summary(tenant: str = "default"):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    return await svc.calibration_summary(tenant=tenant)


@router.get(
    "/calibration/apply",
    dependencies=[Depends(require_scope("read"))],
    summary="Apply empirical calibration correction to a raw confidence value",
)
async def apply_calibration(confidence: float, tenant: str = "default"):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    calibrated = await svc.apply_calibration(confidence, tenant=tenant)
    return {"raw_confidence": confidence, "calibrated_confidence": calibrated}


@router.post(
    "/calibration/snapshot",
    dependencies=[Depends(require_scope("write"))],
    summary="Persist a calibration snapshot for trend tracking",
)
async def calibration_snapshot(tenant: str = "default", label: str = ""):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    snap_id = await svc.persist_snapshot(tenant=tenant, label=label)
    return {"snap_id": snap_id}


@router.get(
    "/calibration/trend",
    dependencies=[Depends(require_scope("read"))],
    summary="Return recent calibration snapshots for trend analysis",
)
async def calibration_trend(tenant: str = "default", limit: int = 10):
    from graphrag.graph.confidence_calibration import CalibrationService
    svc = CalibrationService(get_neo4j())
    return await svc.get_trend(tenant=tenant, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Relation Reification
# ══════════════════════════════════════════════════════════════════════════════

class ReifyRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
    tenant: str = "default"


class StatementMetaRequest(BaseModel):
    stmt_id: str
    key: str
    value: str


class StatementEndorseRequest(BaseModel):
    stmt_id: str
    endorser_id: str
    endorser_type: str = "Document"
    confidence: float = 1.0
    note: str = ""


class StatementContradictRequest(BaseModel):
    stmt_a_id: str
    stmt_b_id: str
    reason: str = ""


@router.post(
    "/reification/reify",
    dependencies=[Depends(require_scope("write"))],
    summary="Create a Statement node from an existing RELATES_TO edge",
)
async def reify_relation(request: ReifyRequest):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    stmt_id = await svc.reify_relation(
        src_name=request.src_name,
        src_type=request.src_type,
        relation=request.relation,
        tgt_name=request.tgt_name,
        tgt_type=request.tgt_type,
        tenant=request.tenant,
    )
    return {"stmt_id": stmt_id}


@router.post(
    "/reification/meta",
    dependencies=[Depends(require_scope("write"))],
    summary="Attach a key/value annotation to a Statement node",
)
async def add_statement_meta(request: StatementMetaRequest):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    await svc.add_meta(stmt_id=request.stmt_id, key=request.key, value=request.value)
    return {"status": "ok"}


@router.post(
    "/reification/endorse",
    dependencies=[Depends(require_scope("write"))],
    summary="Add an ENDORSED_BY link to a Statement",
)
async def endorse_statement(request: StatementEndorseRequest):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    await svc.endorse(
        stmt_id=request.stmt_id,
        endorser_id=request.endorser_id,
        endorser_type=request.endorser_type,
        confidence=request.confidence,
        note=request.note,
    )
    return {"status": "endorsed"}


@router.post(
    "/reification/contradict",
    dependencies=[Depends(require_scope("write"))],
    summary="Assert that Statement A contradicts Statement B",
)
async def contradict_statements(request: StatementContradictRequest):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    await svc.contradict(
        stmt_a_id=request.stmt_a_id,
        stmt_b_id=request.stmt_b_id,
        reason=request.reason,
    )
    return {"status": "contradiction_asserted"}


@router.get(
    "/reification/statements",
    dependencies=[Depends(require_scope("read"))],
    summary="Return Statement nodes for an entity",
)
async def get_statements(
    entity_name: str,
    entity_type: str,
    tenant: str = "default",
    role: str = "subject",
):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    return await svc.get_statements(entity_name, entity_type, tenant, role)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Edge Embeddings
# ══════════════════════════════════════════════════════════════════════════════

class SeedRelationsRequest(BaseModel):
    relations: list[str]
    overwrite: bool = False


class TripleScoreRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
    tenant: str = "default"


@router.post(
    "/edge-embeddings/seed",
    dependencies=[Depends(require_scope("write"))],
    summary="Pre-compute and store relation embeddings",
)
async def seed_relation_embeddings(request: SeedRelationsRequest):
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    svc = EdgeEmbeddingService(get_neo4j())
    result = await svc.seed_relation_embeddings(request.relations, overwrite=request.overwrite)
    return {"seeded": sum(result.values()), "details": result}


@router.post(
    "/edge-embeddings/embed-all",
    dependencies=[Depends(require_scope("write"))],
    summary="Batch-compute triple embeddings for all edges in a tenant",
)
async def embed_all_edges(tenant: str = "default", limit: int = 5000):
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    svc = EdgeEmbeddingService(get_neo4j())
    count = await svc.embed_all_edges(tenant=tenant, limit=limit)
    return {"embedded": count, "tenant": tenant}


@router.get(
    "/edge-embeddings/predict",
    dependencies=[Depends(require_scope("read"))],
    summary="Predict missing relation targets for an entity using TransE",
)
async def predict_links(
    entity_name: str,
    entity_type: str,
    relation: str,
    tenant: str = "default",
    top_k: int = 10,
):
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    svc = EdgeEmbeddingService(get_neo4j())
    return await svc.predict_missing_links(
        entity_name=entity_name,
        entity_type=entity_type,
        relation=relation,
        tenant=tenant,
        top_k=top_k,
    )


@router.post(
    "/edge-embeddings/score",
    dependencies=[Depends(require_scope("read"))],
    summary="Score a specific (head, relation, tail) triple using TransE",
)
async def score_triple(request: TripleScoreRequest):
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    svc = EdgeEmbeddingService(get_neo4j())
    score = await svc.score_triple(
        src_name=request.src_name,
        src_type=request.src_type,
        relation=request.relation,
        tgt_name=request.tgt_name,
        tgt_type=request.tgt_type,
        tenant=request.tenant,
    )
    if score is None:
        raise HTTPException(status_code=404, detail="Entity embeddings not found")
    return {"transx_score": score, "interpretation": "lower = more plausible triple"}


# ══════════════════════════════════════════════════════════════════════════════
# 7. Graph Snapshots
# ══════════════════════════════════════════════════════════════════════════════

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
    return await svc.list_snapshots(tenant=tenant, limit=limit)


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
