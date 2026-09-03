"""Knowledge graph core endpoints: negative knowledge, type taxonomy, bitemporal queries,
reification, property schema, external entity linking, multi-modal entities, and type migration."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_tenant, require_scope
from graphrag.graph.corpus_revision import CorpusMutation
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── Negative Knowledge ────────────────────────────────────────────────────────

class NegativeAssertRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str
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


@router.post(
    "/negative/assert",
    dependencies=[Depends(require_scope("write"))],
    summary="Assert that a relation does NOT hold between two entities",
)
async def assert_negative(request: NegativeAssertRequest, tenant: str = Depends(get_tenant)):
    """
    Create a NEGATIVE_RELATES_TO edge asserting the relation is absent.
    Triggers a warning if a positive RELATES_TO edge also exists (conflict).
    """
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    neo4j = get_neo4j()
    svc = NegativeKnowledgeService(neo4j)
    async with CorpusMutation(neo4j, tenant, "negative_assertion") as mutation:
        neg_id = await svc.assert_negative(
            src_name=request.src_name,
            src_type=request.src_type,
            relation=request.relation,
            tgt_name=request.tgt_name,
            tgt_type=request.tgt_type,
            tenant=tenant,
            doc_id=request.doc_id,
            confidence=request.confidence,
            valid_from=request.valid_from,
            valid_to=request.valid_to,
        )
    return {"status": "asserted", "neg_id": neg_id, "corpus_revision": mutation.revision}


@router.post(
    "/negative/retract",
    dependencies=[Depends(require_scope("write"))],
    summary="Retract a negative assertion (remove NEGATIVE_RELATES_TO edge)",
)
async def retract_negative(request: NegativeRetractRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    neo4j = get_neo4j()
    svc = NegativeKnowledgeService(neo4j)
    async with CorpusMutation(neo4j, tenant, "negative_retraction") as mutation:
        deleted = await svc.retract_negative(
            src_name=request.src_name,
            src_type=request.src_type,
            relation=request.relation,
            tgt_name=request.tgt_name,
            tgt_type=request.tgt_type,
            tenant=tenant,
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="Negative assertion not found")
    return {"status": "retracted", "corpus_revision": mutation.revision}


@router.get(
    "/negative/conflicts",
    dependencies=[Depends(require_scope("read"))],
    summary="Find triples with both positive and negative assertions",
)
async def negative_conflicts(tenant: str = Depends(get_tenant), scan_limit: int = 200):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    return await svc.find_positive_negative_conflicts(tenant=tenant, scan_limit=scan_limit)


@router.get(
    "/negative/list",
    dependencies=[Depends(require_scope("read"))],
    summary="List all NEGATIVE_RELATES_TO edges for a tenant",
)
async def list_negatives(tenant: str = Depends(get_tenant), limit: int = 100):
    from graphrag.graph.negative_knowledge import NegativeKnowledgeService
    svc = NegativeKnowledgeService(get_neo4j())
    return await svc.list_all_negatives(tenant=tenant, limit=limit)


# ── Entity Type Taxonomy ──────────────────────────────────────────────────────

class SubclassRegisterRequest(BaseModel):
    child: str
    parent: str


@router.post(
    "/taxonomy/register",
    dependencies=[Depends(require_scope("admin"))],
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
async def entities_by_type(type_name: str, tenant: str = Depends(get_tenant),
                            include_subtypes: bool = True, limit: int = 100):
    from graphrag.graph.type_taxonomy import get_type_taxonomy
    tax = get_type_taxonomy(get_neo4j())
    if not tax._loaded:
        await tax.load()
    return await tax.query_by_type(type_name=type_name, tenant=tenant,
                                    include_subtypes=include_subtypes, limit=limit)


# ── Bitemporal Queries ────────────────────────────────────────────────────────

@router.get(
    "/bitemporal/entities",
    dependencies=[Depends(require_scope("read"))],
    summary="Return entities valid at (valid_time) as recorded by (transaction_time)",
)
async def bitemporal_entities(
    valid_time: str,
    transaction_time: str,
    tenant: str = Depends(get_tenant),
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
    tenant: str = Depends(get_tenant),
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
    tenant: str = Depends(get_tenant),
):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.transaction_diff(tt_from, tt_to, tenant)


@router.get(
    "/bitemporal/report",
    dependencies=[Depends(require_scope("read"))],
    summary="High-level graph state at (valid_time, transaction_time)",
)
async def bitemporal_report(valid_time: str, transaction_time: str, tenant: str = Depends(get_tenant)):
    from graphrag.graph.bitemporal import BitemporalStore
    store = BitemporalStore(get_neo4j())
    return await store.time_travel_report(valid_time, transaction_time, tenant)


# ── Relation Reification ──────────────────────────────────────────────────────

class ReifyRequest(BaseModel):
    src_name: str
    src_type: str
    relation: str
    tgt_name: str
    tgt_type: str


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
async def reify_relation(request: ReifyRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.reification import ReificationService
    neo4j = get_neo4j()
    svc = ReificationService(neo4j)
    async with CorpusMutation(neo4j, tenant, "statement_reification") as mutation:
        stmt_id = await svc.reify_relation(
            src_name=request.src_name,
            src_type=request.src_type,
            relation=request.relation,
            tgt_name=request.tgt_name,
            tgt_type=request.tgt_type,
            tenant=tenant,
        )
    return {"stmt_id": stmt_id, "corpus_revision": mutation.revision}


@router.post(
    "/reification/meta",
    dependencies=[Depends(require_scope("write"))],
    summary="Attach a key/value annotation to a Statement node",
)
async def add_statement_meta(request: StatementMetaRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.reification import ReificationService
    neo4j = get_neo4j()
    svc = ReificationService(neo4j)
    async with CorpusMutation(neo4j, tenant, "statement_metadata") as mutation:
        await svc.add_meta(
            stmt_id=request.stmt_id, key=request.key, value=request.value,
            tenant=tenant,
        )
    return {"status": "ok", "corpus_revision": mutation.revision}


@router.post(
    "/reification/endorse",
    dependencies=[Depends(require_scope("write"))],
    summary="Add an ENDORSED_BY link to a Statement",
)
async def endorse_statement(request: StatementEndorseRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.reification import ReificationService
    neo4j = get_neo4j()
    svc = ReificationService(neo4j)
    async with CorpusMutation(neo4j, tenant, "statement_endorsement") as mutation:
        await svc.endorse(
            stmt_id=request.stmt_id,
            endorser_id=request.endorser_id,
            endorser_type=request.endorser_type,
            confidence=request.confidence,
            note=request.note,
            tenant=tenant,
        )
    return {"status": "endorsed", "corpus_revision": mutation.revision}


@router.post(
    "/reification/contradict",
    dependencies=[Depends(require_scope("write"))],
    summary="Assert that Statement A contradicts Statement B",
)
async def contradict_statements(request: StatementContradictRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.reification import ReificationService
    neo4j = get_neo4j()
    svc = ReificationService(neo4j)
    async with CorpusMutation(neo4j, tenant, "statement_contradiction") as mutation:
        await svc.contradict(
            stmt_a_id=request.stmt_a_id,
            stmt_b_id=request.stmt_b_id,
            reason=request.reason,
            tenant=tenant,
        )
    return {"status": "contradiction_asserted", "corpus_revision": mutation.revision}


@router.get(
    "/reification/statements",
    dependencies=[Depends(require_scope("read"))],
    summary="Return Statement nodes for an entity",
)
async def get_statements(
    entity_name: str,
    entity_type: str,
    tenant: str = Depends(get_tenant),
    role: str = "subject",
):
    from graphrag.graph.reification import ReificationService
    svc = ReificationService(get_neo4j())
    return await svc.get_statements(entity_name, entity_type, tenant, role)


# ── Property Schema Validation ────────────────────────────────────────────────

@router.get(
    "/property-schema/validate-entity",
    dependencies=[Depends(require_scope("read"))],
    summary="Validate an entity's properties against the registered cardinality rules",
)
async def validate_entity_schema(
    entity_name: str,
    entity_type: str,
    tenant: str = Depends(get_tenant),
):
    from graphrag.graph.property_schema import PropertySchemaValidator
    v = PropertySchemaValidator(get_neo4j())
    return await v.validate_entity(entity_name, entity_type, tenant)


@router.get(
    "/property-schema/validate-document",
    dependencies=[Depends(require_scope("read"))],
    summary="Validate all entities in a document against property cardinality rules",
)
async def validate_document_schema(doc_id: str, tenant: str = Depends(get_tenant)):
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
    tenant: str = Depends(get_tenant),
    limit: int = 100,
):
    from graphrag.graph.property_schema import PropertySchemaValidator
    v = PropertySchemaValidator(get_neo4j())
    return await v.detect_property_conflicts(entity_type, tenant, limit)


# ── Property violations health summary ────────────────────────────────────────

@router.get(
    "/health/property-violations",
    dependencies=[Depends(require_scope("read"))],
    summary="Summarise property cardinality violations across all monitored entity types",
)
async def property_violations_summary(
    tenant: str = Depends(get_tenant),
    limit: int = 50,
):
    """
    Scan all monitored entity types (PERSON, ORG, PRODUCT, LOCATION, EVENT) and
    return a count of property violations per type, plus up to `limit` individual
    issues.  Violations are advisory — they do not block ingestion — but they
    signal extraction noise or conflicting source documents.
    """
    from graphrag.graph.property_schema import PropertySchemaValidator, PROPERTY_RULES
    v = PropertySchemaValidator(get_neo4j())

    all_issues: list[dict] = []
    counts_by_type: dict[str, int] = {}

    for entity_type in PROPERTY_RULES:
        conflicts = await v.detect_property_conflicts(entity_type, tenant, limit=limit)
        n = len(conflicts) if isinstance(conflicts, list) else 0
        counts_by_type[entity_type] = n
        if isinstance(conflicts, list):
            all_issues.extend(conflicts)

    return {
        "tenant":            tenant,
        "total_violations":  sum(counts_by_type.values()),
        "violations_by_type": counts_by_type,
        "sample_issues":     all_issues[:limit],
    }


# ── External Entity Linking (Wikidata) ────────────────────────────────────────

class WikidataLinkRequest(BaseModel):
    entity_name: str
    entity_type: str
    force: bool = False


class WikidataBatchRequest(BaseModel):
    limit: int = 100
    entity_types: list[str] | None = None


@router.post(
    "/entity-linking/link",
    dependencies=[Depends(require_scope("write"))],
    summary="Look up the Wikidata QID for a single entity and cache the result",
)
async def link_entity_wikidata(request: WikidataLinkRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    result = await linker.link_entity(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=tenant,
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
async def link_all_entities(request: WikidataBatchRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    count = await linker.link_all_unlinked(
        tenant=tenant,
        limit=request.limit,
        entity_types=request.entity_types,
    )
    return {"linked": count, "tenant": tenant}


@router.get(
    "/entity-linking/list",
    dependencies=[Depends(require_scope("read"))],
    summary="List all Wikidata-linked entities for a tenant",
)
async def list_linked_entities(tenant: str = Depends(get_tenant), limit: int = 100):
    from graphrag.graph.entity_linker import WikidataEntityLinker
    linker = WikidataEntityLinker(get_neo4j())
    return await linker.list_linked(tenant=tenant, limit=limit)


# ── Multi-Modal Entities ──────────────────────────────────────────────────────

class AttachMediaRequest(BaseModel):
    entity_name: str
    entity_type: str
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
async def attach_media(request: AttachMediaRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    attachment_id = await svc.attach_media(
        entity_name=request.entity_name,
        entity_type=request.entity_type,
        tenant=tenant,
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
    tenant: str = Depends(get_tenant),
):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    return await svc.get_modalities(entity_name, entity_type, tenant)


@router.post(
    "/multimodal/set-embedding",
    dependencies=[Depends(require_scope("write"))],
    summary="Store a cross-modal embedding on a MediaAttachment",
)
async def set_media_embedding(request: SetEmbeddingRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    await svc.set_embedding(
        tenant=tenant,
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
    tenant: str = Depends(get_tenant),
    modality: str | None = None,
    limit: int = 100,
):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    return await svc.get_unembedded(tenant=tenant, modality=modality, limit=limit)


class AttachImageRequest(BaseModel):
    entity_name: str
    entity_type: str
    image_url: str = ""
    caption: str = ""
    mime_type: str = "image/jpeg"
    image_base64: str | None = None


class MediaIdRequest(BaseModel):
    attachment_id: str
    image_base64: str


@router.post(
    "/multimodal/attach-image",
    dependencies=[Depends(require_scope("write"))],
    summary="Attach an image to an entity, computing a perceptual hash inline",
)
async def attach_image(request: AttachImageRequest, tenant: str = Depends(get_tenant)):
    """Attach an image, optionally computing its perceptual hash inline.

    When ``image_base64`` is provided, the decoded bytes are hashed via
    ``compute_phash`` and stored on the attachment, so it's immediately
    eligible for ``GET /multimodal/similar``.
    """
    import base64

    from graphrag.graph.multimodal import MultiModalEntityService

    image_bytes = base64.b64decode(request.image_base64) if request.image_base64 else None
    svc = MultiModalEntityService(get_neo4j())
    try:
        attachment_id = await svc.attach_image(
            entity_name=request.entity_name,
            entity_type=request.entity_type,
            tenant=tenant,
            image_url=request.image_url,
            caption=request.caption,
            mime_type=request.mime_type,
            image_bytes=image_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"attachment_id": attachment_id, "phash_computed": image_bytes is not None}


@router.post(
    "/multimodal/ocr",
    dependencies=[Depends(require_scope("write"))],
    summary="Run OCR on an image attachment and record the extracted text",
)
async def run_media_ocr(request: MediaIdRequest, tenant: str = Depends(get_tenant)):
    """Extract text from an already-attached image via EasyOCR.

    Backfills the attachment's caption with the extracted text if it was
    previously empty, so the text becomes searchable through the existing
    caption-based retrieval path.
    """
    import base64

    from graphrag.graph.multimodal import MultiModalEntityService

    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc

    svc = MultiModalEntityService(get_neo4j())
    try:
        transformation = await svc.run_ocr(tenant, request.attachment_id, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "text": transformation.output_digest,
        "confidence": transformation.metadata.get("confidence", 0.0),
    }


@router.get(
    "/multimodal/similar",
    dependencies=[Depends(require_scope("read"))],
    summary="Find images perceptually similar to a given MediaAttachment",
)
async def find_similar_media(
    attachment_id: str,
    tenant: str = Depends(get_tenant),
    max_distance: int = 8,
    limit: int = 20,
):
    from graphrag.graph.multimodal import MultiModalEntityService
    svc = MultiModalEntityService(get_neo4j())
    return await svc.find_similar_images(
        tenant, attachment_id, max_distance=max_distance, limit=limit
    )


# ── SPARQL Bridge ────────────────────────────────────────────────────────────

class SPARQLRequest(BaseModel):
    query: str
    namespaces: dict[str, str] = {}
    # export_path was client-supplied and flowed straight into Path(...) and
    # rdflib's parser, so any caller could name an arbitrary filesystem path.
    # The export location is server configuration, not request data.


_TENANT_PATH_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@router.post(
    "/sparql",
    dependencies=[Depends(require_scope("read"))],
    summary="Execute a SPARQL 1.1 SELECT query against the exported Turtle graph",
)
async def sparql_query(request: SPARQLRequest, tenant: str = Depends(get_tenant)):
    """Run a read-only SPARQL query against the caller's tenant's RDF export.

    The export is produced by ``scripts/export_rdf.py --tenant <t>``. Its
    location is server configuration derived from the caller's tenant
    (``GRAPHRAG_RDF_EXPORT_DIR``, default ``exports``) plus a fixed
    filename, never a request field. If no export exists there, returns a
    404 with a hint to run the script.

    Previously this route took no tenant dependency at all and read one
    shared export file for every caller -- whichever tenant last ran the
    export script silently became the data every tenant's SPARQL queries
    saw. Tenant is validated against a strict filename-safe pattern before
    it ever touches Path(...), so it can't be used for traversal even
    though it's now part of the constructed path.
    """
    import os
    from pathlib import Path

    from graphrag.graph.sparql_bridge import SPARQLBridge

    if not _TENANT_PATH_RE.fullmatch(tenant):
        raise HTTPException(status_code=403, detail="Invalid tenant")

    export_dir = Path(os.getenv("GRAPHRAG_RDF_EXPORT_DIR", "exports"))
    path = export_dir / tenant / "graph_export.ttl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"RDF export not found at '{path}'. "
                   f"Run: python scripts/export_rdf.py --tenant {tenant}",
        )
    try:
        bridge = SPARQLBridge.from_turtle(path)
        rows   = bridge.query(request.query, init_ns=request.namespaces)
        return {"rows": rows, "count": len(rows)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class SPARQLUpdateRequest(BaseModel):
    query: str
    namespaces: dict[str, str] = {}
    persist: bool = False


@router.post(
    "/sparql/update",
    dependencies=[Depends(require_scope("write"))],
    summary="Execute a SPARQL 1.1 Update against the caller's tenant RDF export",
)
async def sparql_update(request: SPARQLUpdateRequest, tenant: str = Depends(get_tenant)):
    """Run a SPARQL 1.1 Update (INSERT/DELETE/CLEAR/...) against the caller's
    tenant export.

    By default (``persist=False``) the mutation is applied to an in-memory
    copy of the export and discarded -- nothing on disk changes, and Neo4j
    is never touched by this endpoint (RDF here is an export/interop layer,
    not the source of truth). Pass ``persist=True`` to overwrite the
    tenant's ``graph_export.ttl`` with the mutated graph, so a subsequent
    ``POST /kg/sparql`` SELECT sees the change.
    """
    import os
    from pathlib import Path

    from graphrag.graph.sparql_bridge import SPARQLBridge

    if not _TENANT_PATH_RE.fullmatch(tenant):
        raise HTTPException(status_code=403, detail="Invalid tenant")

    export_dir = Path(os.getenv("GRAPHRAG_RDF_EXPORT_DIR", "exports"))
    path = export_dir / tenant / "graph_export.ttl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"RDF export not found at '{path}'. "
                   f"Run: python scripts/export_rdf.py --tenant {tenant}",
        )
    try:
        bridge = SPARQLBridge.from_turtle(path)
        triples_before = len(bridge._g)
        triples_after  = bridge.update(request.query, init_ns=request.namespaces)
        if request.persist:
            bridge.save(path)
        return {
            "triples_before": triples_before,
            "triples_after": triples_after,
            "persisted": request.persist,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Link Prediction ───────────────────────────────────────────────────────────

class LinkPredictionRequest(BaseModel):
    head_id: str
    relation: str
    top_k: int = 10


@router.post(
    "/predict-links",
    dependencies=[Depends(require_scope("read"))],
    summary="Predict candidate tail entities using trained TransE embeddings",
)
async def predict_links(request: LinkPredictionRequest, tenant: str = Depends(get_tenant)):
    """Rank candidate tail entities for (head, relation, ?).

    Requires a trained ``TransXTrainer`` — call ``POST /kg/train-embeddings``
    first to learn relation vectors, then this endpoint scores candidates via
    TransE: h + r ≈ t.
    """
    from graphrag.graph.edge_embeddings import EdgeEmbeddingService
    from graphrag.graph.link_predictor import LinkPredictor
    from graphrag.graph.transx_trainer import TransXTrainer

    neo4j = get_neo4j()
    svc   = EdgeEmbeddingService(neo4j)

    # Load persisted relation embeddings into the shared dict
    await svc.load_relation_embeddings(tenant=tenant)

    trainer   = TransXTrainer(neo4j, rel_emb=svc._rel_emb, embed_dim=svc._embed_dim)
    predictor = LinkPredictor(neo4j, trainer)

    try:
        candidates = await predictor.predict_tail(
            head_id=request.head_id,
            relation=request.relation,
            top_k=request.top_k,
            tenant=tenant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {"candidates": candidates, "head_id": request.head_id,
            "relation": request.relation}


# ── Entity Type Migration ─────────────────────────────────────────────────────

class EntityTypeRenameRequest(BaseModel):
    old_type: str
    new_type: str
    dry_run: bool = False


class OntologyMigrationRequest(BaseModel):
    current: dict
    target: dict
    apply: bool = False


@router.post(
    "/ontology/migration",
    dependencies=[Depends(require_scope("write"))],
    summary="Plan or apply a versioned ontology migration",
)
async def ontology_migration(request: OntologyMigrationRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.ontology_migration import plan_migration
    from graphrag.graph.ontology_registry import get_ontology_registry
    from graphrag.graph.corpus_revision import CorpusMutation
    report = plan_migration(request.current, request.target)
    result = {"compatible": report.compatible, "added": report.added,
              "removed": report.removed, "renamed": report.renamed,
              "warnings": report.warnings, "applied": False}
    if request.apply:
        if not report.compatible:
            raise HTTPException(status_code=409, detail="ontology migration has unmapped removals")
        neo4j = get_neo4j()
        registry = get_ontology_registry(neo4j_client=neo4j, tenant=tenant)
        async with CorpusMutation(neo4j, tenant, "ontology_relation_migration") as mutation:
            result.update(await registry.apply_ontology_migration(request.current, request.target))
        result["corpus_revision"] = mutation.revision
        result["applied"] = True
    return result


@router.post(
    "/ontology/rename-entity-type",
    dependencies=[Depends(require_scope("write"))],
    summary="Cascade-rename an entity type (entities, edges, WikidataLinks, Statements)",
)
async def rename_entity_type(request: EntityTypeRenameRequest, tenant: str = Depends(get_tenant)):
    from graphrag.graph.ontology_registry import get_ontology_registry
    from graphrag.graph.corpus_revision import CorpusMutation
    neo4j = get_neo4j()
    registry = get_ontology_registry(neo4j_client=neo4j, tenant=tenant)
    if request.dry_run:
        return await registry.rename_entity_type(
            old_type=request.old_type, new_type=request.new_type,
            tenant=tenant, dry_run=True,
        )
    async with CorpusMutation(neo4j, tenant, "ontology_entity_type_rename") as mutation:
        result = await registry.rename_entity_type(
            old_type=request.old_type, new_type=request.new_type,
            tenant=tenant, dry_run=False,
        )
    result["corpus_revision"] = mutation.revision
    return result
