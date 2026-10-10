"""Expiry sweep: turn passed ``valid_to`` dates into EVIDENCE_EXPIRED events.

Retrieval already filters expired evidence at query time; what nothing did was
notify the artifacts derived from it (cached answers, snapshots, inferred
edges, recorded decisions). Each expired document/edge is processed once per
``valid_to`` value (``expiry_processed_for`` marker), so re-running the sweep is
idempotent and a later change of ``valid_to`` is seen again.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from graphrag.graph.invalidation.models import EventKind, InvalidationEvent, RelationRef
from graphrag.graph.invalidation.service import emit
from graphrag.graph.validation.graph_checks import run_read_only

_EXPIRED_DOCS = """
MATCH (d:Document {tenant: $tenant})
WHERE d.valid_to IS NOT NULL AND d.valid_to <= datetime($now)
  AND coalesce(d.expiry_processed_for, '') <> toString(d.valid_to)
RETURN d.id AS id, toString(d.valid_to) AS valid_to LIMIT $limit
"""

_EXPIRED_EDGES = """
MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(o:Entity {tenant: $tenant})
WHERE r.tenant = $tenant AND r.valid_to IS NOT NULL AND r.valid_to <= datetime($now)
  AND coalesce(r.expiry_processed_for, '') <> toString(r.valid_to)
RETURN s.name AS src_name, s.type AS src_type, r.relation AS relation,
       o.name AS tgt_name, o.type AS tgt_type, toString(r.valid_to) AS valid_to
LIMIT $limit
"""


async def sweep_expired(neo4j, tenant: str, *, now: datetime | None = None, limit: int = 500) -> dict:
    now_iso = (now or datetime.now(timezone.utc)).isoformat()
    docs = await run_read_only(neo4j, _EXPIRED_DOCS, tenant=tenant, now=now_iso, limit=limit)
    edges = await run_read_only(neo4j, _EXPIRED_EDGES, tenant=tenant, now=now_iso, limit=limit)
    if not docs and not edges:
        return {"documents": 0, "relations": 0, "event": None}
    cause = hashlib.sha256("|".join(
        sorted(f"d:{d['id']}@{d['valid_to']}" for d in docs)
        + sorted(f"r:{e['src_type']}:{e['src_name']}|{e['relation']}|{e['tgt_type']}:{e['tgt_name']}"
                 f"@{e['valid_to']}" for e in edges)).encode("utf-8")).hexdigest()[:24]
    event = InvalidationEvent(
        tenant=tenant, kind=EventKind.EVIDENCE_EXPIRED, reason="valid_to passed", actor="expiry_sweep",
        cause=cause, document_ids=[d["id"] for d in docs],
        relations=[RelationRef(**{k: e[k] for k in ("src_name", "src_type", "relation", "tgt_name", "tgt_type")})
                   for e in edges],
    )
    summary = await emit(event, neo4j, inside_mutation=False)
    if summary.get("fallback") != "error":
        await neo4j.run(
            "UNWIND $ids AS id MATCH (d:Document {tenant: $tenant, id: id}) "
            "SET d.expiry_processed_for = toString(d.valid_to)",
            tenant=tenant, ids=[d["id"] for d in docs])
        await neo4j.run(
            """
            UNWIND $edges AS k
            MATCH (s:Entity {tenant: $tenant, name: k.src_name, type: k.src_type})
                  -[r:RELATES_TO {relation: k.relation}]->
                  (o:Entity {tenant: $tenant, name: k.tgt_name, type: k.tgt_type})
            WHERE r.tenant = $tenant
            SET r.expiry_processed_for = toString(r.valid_to)
            """,
            tenant=tenant, edges=[{k: e[k] for k in ("src_name", "src_type", "relation", "tgt_name", "tgt_type")}
                                  for e in edges])
    return {"documents": len(docs), "relations": len(edges), "event": summary}
