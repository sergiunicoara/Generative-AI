"""POST/GET /kg — Knowledge graph architecture feature endpoints.

Phase 7 — 7 production KG gaps:
  1. Negative knowledge     (NEGATIVE_RELATES_TO assertions)
  2. Entity type taxonomy   (SUBCLASS_OF hierarchy + type expansion)
  3. Bitemporal queries     (valid-time + transaction-time "as of" queries)
  4. Confidence calibration (Brier score, calibration curve, apply correction)
  5. Relation reification   (Statement nodes + meta-statements)
  6. Edge embeddings        (TransE scoring, link prediction, training)
  7. Graph snapshots        (named checkpoints + diff)

Phase 8 — 8 additional features:
  8.  Embedding registry     (version inventory, compatibility check, re-embed queue)
  9.  Property schema        (cardinality validation, conflict detection)
  10. Inference engine       (forward-chaining rules, dry-run)
  11. External entity linking (Wikidata QID lookup + batch)
  12. Counterfactual analysis (simulate document retraction, impact score)
  13. GDPR / erasure         (forget_entity, forget_document, audit log)
  14. PII guard              (scan, redact, tag, inventory)
  15. Query cache            (invalidate by entity, flush tenant, stats)
  16. Incremental community  (change detection, partial rebuild)
  17. Multi-modal entities   (attach images/audio, set embeddings)
  18. Entity type migration  (rename_entity_type cascade)

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


# ══════════════════════════════════════════════════════════════════════════════
# 8. Edge Embeddings — TransE Training
# ══════════════════════════════════════════════════════════════════════════════

class TransETrainRequest(BaseModel):
    tenant: str = "default"
    epochs: int = 100
    lr: float = 0.01
    margin: float = 1.0
    neg_samples: int = 5
    batch_size: int = 256
    seed: int = 42


@router.post(
    "/edge-embeddings/train",
    dependencies=[Depends(require_scope("write"))],
    summary="Train TransE relation embeddings using negative-sampling SGD",
)
async def train_transe(request: TransETrainRequest):
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    svc = EdgeEmbeddingService(get_neo4j())
    result = await svc.train(
        tenant=request.tenant,
        epochs=request.epochs,
        lr=request.lr,
        margin=request.margin,
        neg_samples=request.neg_samples,
        batch_size=request.batch_size,
        seed=request.seed,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9. Embedding Registry
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/embedding-registry/inventory",
    dependencies=[Depends(require_scope("read"))],
    summary="List embedding model versions in use per tenant",
)
async def embedding_inventory(tenant: str = "default"):
    from graphrag.graph.embedding_registry import EmbeddingRegistry
    reg = EmbeddingRegistry(get_neo4j())
    return await reg.inventory(tenant=tenant)


class EmbeddingCompatRequest(BaseModel):
    current_model: str
    current_version: str = "latest"
    expected_dim: int = 768
    tenant: str = "default"


@router.post(
    "/embedding-registry/check-compatibility",
    dependencies=[Depends(require_scope("read"))],
    summary="Check whether the current model/version is compatible with stored embeddings",
)
async def check_embedding_compat(request: EmbeddingCompatRequest):
    from graphrag.graph.embedding_registry import EmbeddingRegistry
    reg = EmbeddingRegistry(get_neo4j())
    return await reg.check_compatibility(
        current_model=request.current_model,
        current_version=request.current_version,
        expected_dim=request.expected_dim,
        tenant=request.tenant,
    )


class QueueReEmbedRequest(BaseModel):
    model: str
    version: str = "latest"
    tenant: str = "default"
    limit: int = 10000
    force: bool = False


@router.post(
    "/embedding-registry/queue-re-embed",
    dependencies=[Depends(require_scope("write"))],
    summary="Flag stale entities for re-embedding",
)
async def queue_re_embed(request: QueueReEmbedRequest):
    from graphrag.graph.embedding_registry import EmbeddingRegistry
    reg = EmbeddingRegistry(get_neo4j())
    count = await reg.queue_re_embed(
        model=request.model,
        version=request.version,
        tenant=request.tenant,
        limit=request.limit,
        force=request.force,
    )
    return {"queued": count, "tenant": request.tenant}


# ══════════════════════════════════════════════════════════════════════════════
# 10. Property Schema Validation
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/property-schema/validate-entity",
    dependencies=[Depends(require_scope("read"))],
    summary="Validate an entity's properties against the registered cardinality rules",
)
async def validate_entity_schema(
    entity_name: str,
    entity_type: str,
    tenant: str = "default",
):
    from graphrag.graph.property_schema import PropertySchemaValidator
    v = PropertySchemaValidator(get_neo4j())
    return await v.validate_entity(entity_name, entity_type, tenant)


@router.get(
    "/property-schema/validate-document",
    dependencies=[Depends(require_scope("read"))],
    summary="Validate all entities in a document against property cardinality rules",
)
async def validate_document_schema(doc_id: str, tenant: str = "default"):
    from graphrag.graph.property_schema import PropertySchemaValidator
    v = PropertySchemaValidator(get_neo4j())
    return await v.validate_document(doc_id, tenant)


@router.get(
    "/property-schema/conflicts",
    dependencies=[Depends(require_scope("read"))],
    summary="Detect property-level conflicts for an entity type",
)
async def detect_property_conflicts(
    entity_type: str,
    tenant: str = "default",
    limit: int = 100,
):
    from graphrag.graph.property_schema import PropertySchemaValidator
    v = PropertySchemaValidator(get_neo4j())
    return await v.detect_property_conflicts(entity_type, tenant, limit)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Forward-Chaining Inference Engine
# ══════════════════════════════════════════════════════════════════════════════

class InferenceRunRequest(BaseModel):
    tenant: str = "default"
    max_iterations: int = 5
    dry_run: bool = False


class InferenceDocRequest(BaseModel):
    doc_id: str
    tenant: str = "default"


@router.post(
    "/inference/run",
    dependencies=[Depends(require_scope("write"))],
    summary="Run forward-chaining inference rules and materialise inferred edges",
)
async def run_inference(request: InferenceRunRequest):
    from graphrag.graph.inference_engine import ForwardChainingEngine
    engine = ForwardChainingEngine(get_neo4j())
    return await engine.run(
        tenant=request.tenant,
        max_iterations=request.max_iterations,
        dry_run=request.dry_run,
    )


@router.post(
    "/inference/run-for-document",
    dependencies=[Depends(require_scope("write"))],
    summary="Run forward-chaining rules scoped to a single document",
)
async def run_inference_for_doc(request: InferenceDocRequest):
    from graphrag.graph.inference_engine import ForwardChainingEngine
    engine = ForwardChainingEngine(get_neo4j())
    return await engine.run_for_document(
        doc_id=request.doc_id,
        tenant=request.tenant,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 12. External Entity Linking (Wikidata)
# ══════════════════════════════════════════════════════════════════════════════

class WikidataLinkRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    force: bool = False


class WikidataBatchRequest(BaseModel):
    tenant: str = "default"
    limit: int = 100
    entity_types: list[str] | None = None


@router.post(
    "/entity-linking/link",
    dependencies=[Depends(require_scope("write"))],
    summary="Look up the Wikidata QID for a single entity and cache the result",
)
async def link_entity_wikidata(request: WikidataLinkRequest):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    result = await linker.link_entity(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        force=request.force,
    )
    if result is None:
        return {"status": "not_found", "entity": request.entity_name}
    return {"status": "linked", **result}


@router.post(
    "/entity-linking/link-all",
    dependencies=[Depends(require_scope("write"))],
    summary="Batch-link all unlinked entities for a tenant",
)
async def link_all_entities(request: WikidataBatchRequest):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    count = await linker.link_all_unlinked(
        tenant=request.tenant,
        limit=request.limit,
        entity_types=request.entity_types,
    )
    return {"linked": count, "tenant": request.tenant}


@router.get(
    "/entity-linking/list",
    dependencies=[Depends(require_scope("read"))],
    summary="List all Wikidata-linked entities for a tenant",
)
async def list_linked_entities(tenant: str = "default", limit: int = 100):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    return await linker.list_linked(tenant=tenant, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 13. Counterfactual Analysis
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/counterfactual/simulate/{doc_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Simulate the impact of removing a document without modifying data",
)
async def simulate_retraction(doc_id: str, tenant: str = "default"):
    from graphrag.graph.counterfactual import CounterfactualAnalyzer
    analyzer = CounterfactualAnalyzer(get_neo4j())
    return await analyzer.simulate_retraction(doc_id=doc_id, tenant=tenant)


# ══════════════════════════════════════════════════════════════════════════════
# 14. GDPR / Right-to-be-Forgotten
# ══════════════════════════════════════════════════════════════════════════════

class ForgetEntityRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    requested_by: str = "dpo"
    request_id: str = ""


class ForgetDocumentRequest(BaseModel):
    doc_id: str
    tenant: str = "default"
    requested_by: str = "dpo"
    request_id: str = ""


@router.post(
    "/gdpr/forget-entity",
    dependencies=[Depends(require_scope("write"))],
    summary="Permanently erase all data for a named entity (GDPR right-to-be-forgotten)",
)
async def gdpr_forget_entity(request: ForgetEntityRequest):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.forget_entity(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        requested_by=request.requested_by,
        request_id=request.request_id,
    )


@router.post(
    "/gdpr/forget-document",
    dependencies=[Depends(require_scope("write"))],
    summary="Erase all data exclusively sourced from a document (GDPR erasure)",
)
async def gdpr_forget_document(request: ForgetDocumentRequest):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.forget_document(
        doc_id=request.doc_id,
        tenant=request.tenant,
        requested_by=request.requested_by,
        request_id=request.request_id,
    )


@router.get(
    "/gdpr/audit-log",
    dependencies=[Depends(require_scope("read"))],
    summary="Return the GDPR deletion audit log for a tenant",
)
async def gdpr_audit_log(tenant: str = "default", limit: int = 100):
    from graphrag.graph.gdpr import GDPRService
    svc = GDPRService(get_neo4j())
    return await svc.deletion_audit_log(tenant=tenant, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 15. PII Guard
# ══════════════════════════════════════════════════════════════════════════════

class PIIScanRequest(BaseModel):
    text: str
    min_confidence: float = 0.80


class PIITagRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    reason: str = ""


@router.post(
    "/pii/scan",
    dependencies=[Depends(require_scope("read"))],
    summary="Scan text for PII patterns (SSN, email, phone, credit card, etc.)",
)
async def pii_scan(request: PIIScanRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j(), min_confidence=request.min_confidence)
    findings = guard.scan_text(request.text)
    return {
        "finding_count": len(findings),
        "findings": [
            {
                "pii_class":  f.pii_class,
                "confidence": f.confidence,
                "offset":     f.start,
                "length":     f.end - f.start,
            }
            for f in findings
        ],
    }


@router.post(
    "/pii/redact",
    dependencies=[Depends(require_scope("read"))],
    summary="Return text with PII replaced by [CLASS_REDACTED] placeholders",
)
async def pii_redact(request: PIIScanRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j(), min_confidence=request.min_confidence)
    return {"redacted_text": guard.redact(request.text)}


@router.post(
    "/pii/tag-entity",
    dependencies=[Depends(require_scope("write"))],
    summary="Mark an entity as PII-sensitive in Neo4j",
)
async def pii_tag_entity(request: PIITagRequest):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    await guard.tag_entity_pii(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        reason=request.reason,
    )
    return {"status": "tagged"}


@router.post(
    "/pii/auto-tag-persons",
    dependencies=[Depends(require_scope("write"))],
    summary="Tag all PERSON entities in a tenant as PII-sensitive",
)
async def pii_auto_tag_persons(tenant: str = "default"):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    count = await guard.auto_tag_persons(tenant=tenant)
    return {"tagged": count, "tenant": tenant}


@router.get(
    "/pii/scan-document/{doc_id}",
    dependencies=[Depends(require_scope("read"))],
    summary="Scan all chunks of a document for PII (diagnostic only — no mutations)",
)
async def pii_scan_document(doc_id: str, tenant: str = "default"):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    return await guard.scan_document(doc_id=doc_id, tenant=tenant)


@router.get(
    "/pii/inventory",
    dependencies=[Depends(require_scope("read"))],
    summary="List all entities tagged as PII-sensitive",
)
async def pii_inventory(tenant: str = "default", limit: int = 100):
    from graphrag.graph.pii_guard import PIIGuard
    guard = PIIGuard(get_neo4j())
    return await guard.list_pii_entities(tenant=tenant, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 16. Query Cache
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 17. Incremental Community Detection
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/incremental-community/summary",
    dependencies=[Depends(require_scope("read"))],
    summary="Show how many entities changed since the last community build",
)
async def incremental_community_summary(tenant: str = "default"):
    from graphrag.graph.incremental_community import IncrementalCommunityDetector
    detector = IncrementalCommunityDetector(get_neo4j())
    return await detector.community_change_summary(tenant=tenant)


class IncrementalRebuildRequest(BaseModel):
    tenant: str = "default"
    dry_run: bool = False


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


# ══════════════════════════════════════════════════════════════════════════════
# 18. Multi-Modal Entities
# ══════════════════════════════════════════════════════════════════════════════

class AttachMediaRequest(BaseModel):
    entity_name: str
    entity_type: str
    tenant: str = "default"
    modality: str = "image"
    media_url: str = ""
    caption: str = ""
    mime_type: str = ""


class SetEmbeddingRequest(BaseModel):
    attachment_id: str
    embedding: list[float]


@router.post(
    "/multimodal/attach",
    dependencies=[Depends(require_scope("write"))],
    summary="Attach a media reference (image, audio, video) to an entity",
)
async def attach_media(request: AttachMediaRequest):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    attachment_id = await svc.attach_media(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=request.tenant,
        modality=request.modality,
        media_url=request.media_url,
        caption=request.caption,
        mime_type=request.mime_type,
    )
    return {"attachment_id": attachment_id}


@router.get(
    "/multimodal/entity",
    dependencies=[Depends(require_scope("read"))],
    summary="Return all media attachments for an entity",
)
async def get_entity_media(
    entity_name: str,
    entity_type: str,
    tenant: str = "default",
):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    return await svc.get_modalities(entity_name, entity_type, tenant)


@router.post(
    "/multimodal/set-embedding",
    dependencies=[Depends(require_scope("write"))],
    summary="Store a cross-modal embedding on a MediaAttachment",
)
async def set_media_embedding(request: SetEmbeddingRequest):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    await svc.set_embedding(
        attachment_id=request.attachment_id,
        embedding=request.embedding,
    )
    return {"status": "stored", "dim": len(request.embedding)}


@router.get(
    "/multimodal/unembedded",
    dependencies=[Depends(require_scope("read"))],
    summary="List MediaAttachments that have no embedding yet",
)
async def list_unembedded_media(
    tenant: str = "default",
    modality: str | None = None,
    limit: int = 100,
):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    return await svc.get_unembedded(tenant=tenant, modality=modality, limit=limit)


# ══════════════════════════════════════════════════════════════════════════════
# 19. Entity Type Migration
# ══════════════════════════════════════════════════════════════════════════════

class EntityTypeRenameRequest(BaseModel):
    old_type: str
    new_type: str
    tenant: str = "default"
    dry_run: bool = False


@router.post(
    "/ontology/rename-entity-type",
    dependencies=[Depends(require_scope("write"))],
    summary="Cascade-rename an entity type (entities, edges, WikidataLinks, Statements)",
)
async def rename_entity_type(request: EntityTypeRenameRequest):
    from graphrag.graph.ontology_registry import get_ontology_registry
    registry = get_ontology_registry(neo4j_client=get_neo4j())
    return await registry.rename_entity_type(
        old_type=request.old_type,
        new_type=request.new_type,
        tenant=request.tenant,
        dry_run=request.dry_run,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 20. HDBSCAN Semantic Community Detection
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/semantic-communities/build",
    dependencies=[Depends(require_scope("write"))],
    summary="Build HDBSCAN semantic communities as a parallel signal to Leiden",
)
async def build_semantic_communities(tenant: str = "default"):
    from graphrag.graph.community_builder import CommunityBuilder
    builder = CommunityBuilder(tenant=tenant)
    return await builder.build_semantic_communities()


# ══════════════════════════════════════════════════════════════════════════════
# 21. Alert history
# ══════════════════════════════════════════════════════════════════════════════

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
