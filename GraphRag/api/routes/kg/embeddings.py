"""Edge embeddings (TransE scoring, link prediction, training) and embedding registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import require_scope
from graphrag.graph.neo4j_client import get_neo4j

router = APIRouter()


# ── Edge Embeddings ───────────────────────────────────────────────────────────

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


class TransETrainRequest(BaseModel):
    tenant: str = "default"
    epochs: int = 100
    lr: float = 0.01
    margin: float = 1.0
    neg_samples: int = 5
    batch_size: int = 256
    seed: int = 42


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


# ── Embedding Registry ────────────────────────────────────────────────────────

class EmbeddingCompatRequest(BaseModel):
    current_model: str
    current_version: str = "latest"
    expected_dim: int = 768
    tenant: str = "default"


class QueueReEmbedRequest(BaseModel):
    model: str
    version: str = "latest"
    tenant: str = "default"
    limit: int = 10000
    force: bool = False


@router.get(
    "/embedding-registry/inventory",
    dependencies=[Depends(require_scope("read"))],
    summary="List embedding model versions in use per tenant",
)
async def embedding_inventory(tenant: str = "default"):
    from graphrag.graph.embedding_registry import EmbeddingRegistry
    reg = EmbeddingRegistry(get_neo4j())
    return await reg.inventory(tenant=tenant)


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
