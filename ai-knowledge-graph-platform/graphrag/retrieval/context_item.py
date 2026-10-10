"""Shared internal context contract (plan Phase 8, optional).

One shape for evidence from different surfaces -- document chunks, graph facts,
community summaries, structured rows, session memory -- WITHOUT flattening them
into one store. Each existing result is adapted on the way out of retrieval; no
data is migrated. ``fuse`` merges items from several surfaces while keeping
every item's identity, provenance, temporal validity and authority: two
surfaces returning the same identity are merged (provenance concatenated), never
silently replaced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ContextType = Literal["document_chunk", "graph_fact", "community_summary", "structured_record", "memory"]


class ContextItem(BaseModel):
    identity: str                      # stable, surface-qualified id ("chunk:<id>", "fact:<key>", ...)
    content: str | dict
    context_type: ContextType
    source: str                        # surface that produced it
    provenance: list[dict] = Field(default_factory=list)
    tenant: str
    scope: list[str] = Field(default_factory=list)   # access scopes it was retrieved under
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    trust: dict[str, Any] | None = None


def _dt(value) -> datetime | None:
    from graphrag.graph.trust import _parse
    return _parse(value)


def _unit(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def from_chunk(chunk: dict, *, tenant: str, scope: list[str] | None = None) -> ContextItem:
    trust = chunk.get("trust") or {}
    return ContextItem(
        identity=f"chunk:{chunk.get('chunk_id')}",
        content=chunk.get("text") or "",
        context_type="document_chunk",
        source=str(chunk.get("retrieval") or "local_search"),
        provenance=[{"surface": chunk.get("retrieval") or "local_search",
                     "document_id": chunk.get("document_id"), "document": chunk.get("_doc_name"),
                     "origin": trust.get("origin", "EXTRACTED"),
                     "authority_level": chunk.get("authority_level"),
                     "score_components": chunk.get("score_components")}],
        tenant=tenant, scope=list(scope or []),
        valid_from=_dt(chunk.get("doc_valid_from")), valid_to=_dt(chunk.get("doc_valid_to")),
        confidence=_unit(trust.get("trust_factor", 1.0)), trust=trust or None,
    )


def from_graph_edge(edge: dict, *, tenant: str, scope: list[str] | None = None) -> ContextItem:
    from graphrag.graph.inference_engine import relation_key

    trust = edge.get("trust") or {}
    key = relation_key(edge.get("src_type") or "", edge.get("src") or "", edge.get("relation") or "",
                       edge.get("tgt_type") or "", edge.get("tgt") or "")
    return ContextItem(
        identity=f"fact:{key}",
        content={"src": edge.get("src"), "relation": edge.get("relation"), "tgt": edge.get("tgt")},
        context_type="graph_fact",
        source="graph_subgraph",
        provenance=[{"source_document_id": edge.get("source_doc_id"),
                     "origin": trust.get("origin") or edge.get("origin"),
                     "inferred_by": edge.get("inferred_by"), "premises": edge.get("premise_keys")}],
        tenant=tenant, scope=list(scope or []),
        valid_from=_dt(edge.get("valid_from")), valid_to=_dt(edge.get("valid_to")),
        confidence=_unit(edge.get("confidence", 1.0)), trust=trust or None,
    )


def from_community(community: dict, *, tenant: str, scope: list[str] | None = None) -> ContextItem:
    return ContextItem(
        identity=f"community:{community.get('id') or community.get('community_id')}",
        content=community.get("summary") or "",
        context_type="community_summary",
        source="global_search",
        provenance=[{"documents": community.get("document_ids") or community.get("source_documents") or []}],
        tenant=tenant, scope=list(scope or []),
        valid_from=_dt(community.get("valid_from")), valid_to=_dt(community.get("valid_to")),
        confidence=_unit(community.get("score", 1.0)),
    )


def from_structured_row(row: dict, *, intent: str, tenant: str, index: int) -> ContextItem:
    return ContextItem(
        identity=f"row:{intent}:{index}",
        content=dict(row),
        context_type="structured_record",
        source=f"controlled_query:{intent}",
        provenance=[{"source_document_id": row.get("source_doc_id"), "origin": "EXTRACTED"}],
        tenant=tenant,
    )


def from_episode(episode: dict, *, tenant: str) -> ContextItem:
    return ContextItem(
        identity=f"memory:{episode.get('id')}",
        content=episode.get("summary") or episode.get("content") or "",
        context_type="memory",
        source="session_memory",
        provenance=[{"session_id": episode.get("session_id"), "origin": "GENERATED"}],
        tenant=tenant,
        observed_at=_dt(episode.get("recorded_at")) or datetime.now(timezone.utc),
    )


def fuse(*surfaces: list[ContextItem]) -> list[ContextItem]:
    """Merge items from several surfaces, keyed by identity.

    Items must share one tenant (mixing tenants raises). Duplicates keep the
    first item's content and the union of provenance and scopes; the highest
    confidence wins and the narrowest validity window is kept, so merging can
    never make evidence look more current than any source says.
    """
    merged: dict[str, ContextItem] = {}
    tenants = set()
    for items in surfaces:
        for item in items:
            tenants.add(item.tenant)
            if len(tenants) > 1:
                raise ValueError("cannot fuse context items from different tenants")
            prior = merged.get(item.identity)
            if prior is None:
                merged[item.identity] = item.model_copy(deep=True)
                continue
            prior.provenance.extend(p for p in item.provenance if p not in prior.provenance)
            prior.scope = sorted(set(prior.scope) | set(item.scope))
            prior.confidence = max(prior.confidence, item.confidence)
            starts = [d for d in (prior.valid_from, item.valid_from) if d]
            ends = [d for d in (prior.valid_to, item.valid_to) if d]
            prior.valid_from = max(starts) if starts else None
            prior.valid_to = min(ends) if ends else None
    return list(merged.values())


def context_items_from_results(local_results: dict, global_results: dict | None, *, tenant: str,
                               acl_enforced: bool) -> list[ContextItem]:
    """Adapt one query's retrieval output. Graph facts are omitted under ACL,
    matching the retrieval safe mode."""
    chunks = [from_chunk(c, tenant=tenant) for c in local_results.get("chunks") or []]
    facts = [] if acl_enforced else [from_graph_edge(e, tenant=tenant)
                                     for e in local_results.get("entity_edges") or []]
    communities = [from_community(c, tenant=tenant) for c in (global_results or {}).get("communities") or []]
    return fuse(chunks, facts, communities)
