"""POST /search — synchronous semantic search: ranked chunks, no LLM synthesis.

The platform's retrieval stack (query embedding, Neo4j vector ANN, BM25, RRF
fusion, cross-encoder reranking) was reachable only via POST /query, which
always runs the full pipeline through to LLM answer synthesis. A caller that
only wants ranked passages had to pay for a generation to get them. This route
calls the same retrieval primitive, LocalSearch.search(), directly and returns
before synthesis — because LocalSearch itself never invokes an LLM (see its
module docstring), there is nothing to skip; this route simply never reaches
the layer (HybridRetriever.retrieve_and_answer) that would.

Fixed retrieval profile — not a client choice
----------------------------------------------
The profile is hardcoded to "text_hybrid" (BM25 + vector + cross-encoder
rerank; no multi-hop, no GNN, no entity context, no PageRank tiebreak, no
authority weighting). This is not a performance default — it is the ACL
boundary. Five Neo4j queries used by the disabled stages (multi-hop traversal,
entity neighbors, entity-relation subgraphs, chunk-entity embeddings, PageRank
by entity name) take no `access_context` parameter and cannot be filtered by
it, which is why `local_search.py`'s own ACL-safe-mode already disables them
when access control is enforced. Accepting a client-supplied profile would let
a caller select into those unfiltered paths. See
graphrag/retrieval/hybrid_retriever.py's _RETRIEVAL_PROFILE_OVERRIDES.

No entities in the response for the same reason: entity context is produced
by get_entity_neighbors, one of the five un-ACL'd queries.
"""

from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth.dependencies import get_current_user, get_tenant, require_scope
from api.limiter import SEARCH_LIMIT, rate_limit
from api.quota import enforce_tenant_quota
from graphrag.enterprise.models import AccessContext
from graphrag.retrieval.hybrid_retriever import retrieval_profile_overrides
from graphrag.retrieval.local_search import LocalSearch

router = APIRouter()
log = structlog.get_logger(__name__)

_SEARCH_PROFILE = "text_hybrid"
_SEARCH_TIMEOUT_SECONDS = 20.0


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=10, ge=1, le=50)
    valid_at: str | None = Field(default=None, max_length=64)
    transaction_at: str | None = Field(default=None, max_length=64)


class SearchHit(BaseModel):
    chunk_id: str
    text: str
    score: float
    document: str | None = None
    retrieval: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchHit]


# Constructed once: LocalSearch.__init__ builds a cross-encoder reranker and a
# GNN scorer, both model holders, so building one per request would reload
# model weights on every call. Matches HybridRetriever.__init__'s own
# self._local = LocalSearch() (hybrid_retriever.py:130). Imported and called
# as a module-level function (not a class attribute) so tests can patch
# api.routes.search.LocalSearch directly.
_searcher: LocalSearch | None = None


def _get_searcher() -> LocalSearch:
    global _searcher
    if _searcher is None:
        _searcher = LocalSearch()
    return _searcher


def _rank_key(chunk: dict) -> float:
    """Score to sort by, defensively.

    LocalSearch only re-sorts its chunk list by `final_score` inside the GNN
    branch (see gnn_scorer.py), which the fixed "text_hybrid" profile disables
    — so chunks here arrive in RRF/fusion order and carry no `final_score` at
    all. Fall through to `rerank_score` (set by the cross-encoder, which
    text_hybrid does run) and finally the raw fusion `score`.
    """
    return chunk.get("final_score", chunk.get("rerank_score", chunk.get("score", 0.0)))


@router.post(
    "",
    response_model=SearchResponse,
    # Same ordering rationale as POST /query: scope, then burst protection,
    # then budget — the quota check is the most expensive, so it only runs for
    # requests already known to be authorized and within their rate.
    dependencies=[
        Depends(require_scope("read")),
        Depends(rate_limit(SEARCH_LIMIT)),
        Depends(enforce_tenant_quota),
    ],
)
async def search(
    request: Request,
    body: SearchRequest,
    tenant: str = Depends(get_tenant),
    user: dict = Depends(get_current_user),
) -> SearchResponse:
    """Ranked semantic search over this tenant's corpus. No LLM call is made.

    Rate-limited (GRAPHRAG_RATE_LIMIT_SEARCH, default 60/minute) and subject
    to the tenant's quota: this still costs one embedding call and one
    cross-encoder pass per request, so it is not free.
    """
    searcher = _get_searcher()
    # local_top_k/rerank_top_k (not a search() parameter — LocalSearch reads
    # both from config) govern how many candidates are actually fetched and
    # reranked. Without overriding them here, a caller's top_k only slices
    # client-side after the fact and a request for more than the tenant's
    # configured default (10/5) would silently come back short.
    overrides = {
        **retrieval_profile_overrides(_SEARCH_PROFILE),
        "local_top_k": body.top_k,
        "rerank_top_k": body.top_k,
    }
    try:
        result = await asyncio.wait_for(
            searcher.search(
                body.query,
                tenant=tenant,
                valid_at=body.valid_at,
                transaction_at=body.transaction_at,
                config_overrides=overrides,
                access_context=AccessContext.from_claims(user),
            ),
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        log.error(
            "search.retrieval_timeout",
            correlation_id=getattr(request.state, "correlation_id", ""),
            timeout_s=_SEARCH_TIMEOUT_SECONDS,
        )
        raise HTTPException(status_code=504, detail="Search timed out") from exc
    except Exception as exc:
        log.error(
            "search.retrieval_failed",
            correlation_id=getattr(request.state, "correlation_id", ""),
            exception_type=type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="Search unavailable") from exc

    chunks = sorted(result.get("chunks", []), key=_rank_key, reverse=True)[: body.top_k]
    return SearchResponse(
        results=[
            SearchHit(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=_rank_key(chunk),
                document=chunk.get("_doc_name") or chunk.get("source"),
                retrieval=chunk.get("retrieval"),
            )
            for chunk in chunks
        ]
    )
