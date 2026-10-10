"""Reverse dependency walk: from changed facts to the derived artifacts built on them.

Dependency edges followed (all already in the graph; read-only, tenant-scoped):

    Document --(Chunk.document_id)--> Chunk
    Document --(RELATES_TO.source_doc_ids)--> relation
    relation --(endpoints)--> Entity                      (relation facts reach answers via entities)
    Entity <-[:MENTIONS]- Chunk
    relation / Entity --(premise_keys)--> inferred edge   (transitively: multi-hop)
    Chunk / Document / Entity --(snapshot.chunk_ids/document_ids/entity_ids)--> CommunitySummarySnapshot
    Chunk / Document --(manifest chunk_ids/document_ids, INCLUDED_*)--> CGContextManifest -> CGDecision

The walk over-approximates (an answer that cited an entity whose edge changed is
included even if that edge was not in its context): a false positive costs one
recomputation, a false negative serves a stale answer.
"""
from __future__ import annotations

from graphrag.graph.invalidation.models import Closure, EntityRef, EventKind, InvalidationEvent
from graphrag.graph.validation.graph_checks import run_read_only

_DOC_CHUNKS = """
MATCH (c:Chunk {tenant: $tenant}) WHERE c.document_id IN $ids
RETURN c.id AS id LIMIT $cap
"""

_DOC_RELATIONS = """
MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(o:Entity {tenant: $tenant})
WHERE r.tenant = $tenant AND any(d IN coalesce(r.source_doc_ids, []) WHERE d IN $ids)
RETURN s.name AS src_name, s.type AS src_type, r.relation AS relation,
       o.name AS tgt_name, o.type AS tgt_type
LIMIT $cap
"""

_ENTITY_CHUNKS_AND_IDS = """
UNWIND $entities AS e
MATCH (x:Entity {tenant: $tenant, name: e.name, type: e.type})
OPTIONAL MATCH (c:Chunk {tenant: $tenant})-[:MENTIONS]->(x)
RETURN x.id AS entity_id, collect(DISTINCT c.id)[..$cap] AS chunk_ids
"""

_DEPENDENT_INFERRED = """
MATCH (s:Entity {tenant: $tenant})-[r:RELATES_TO]->(o:Entity {tenant: $tenant})
WHERE r.tenant = $tenant AND r.source_type = 'inferred'
  AND (
    any(p IN coalesce(r.premise_keys, []) WHERE p IN $keys
        OR any(tok IN $tokens WHERE p STARTS WITH tok + '|' OR p ENDS WITH '|' + tok))
    // Inferred edges written before premises were recorded: depend on their endpoints.
    OR (size(coalesce(r.premise_keys, [])) = 0
        AND (s.type + ':' + s.name IN $tokens OR o.type + ':' + o.name IN $tokens))
  )
RETURN s.name AS src_name, s.type AS src_type, r.relation AS relation,
       o.name AS tgt_name, o.type AS tgt_type, r.inferred_by AS rule
LIMIT $cap
"""

_SNAPSHOTS = """
MATCH (s:CommunitySummarySnapshot {tenant: $tenant})
WHERE s.transaction_to IS NULL
  AND (any(c IN coalesce(s.chunk_ids, []) WHERE c IN $chunks)
       OR any(d IN coalesce(s.document_ids, []) WHERE d IN $docs)
       OR any(e IN coalesce(s.entity_ids, []) WHERE e IN $entity_ids))
RETURN s.id AS id LIMIT $cap
"""

_DECISIONS = """
MATCH (m:CGContextManifest {tenant: $tenant})
WHERE any(c IN coalesce(m.chunk_ids, []) WHERE c IN $chunks)
   OR any(d IN coalesce(m.document_ids, []) WHERE d IN $docs)
   OR EXISTS { MATCH (m)-[:INCLUDED_CHUNK]->(c:Chunk) WHERE c.id IN $chunks }
   OR EXISTS { MATCH (m)-[:INCLUDED_DOCUMENT]->(d:Document) WHERE d.id IN $docs }
MATCH (run:CGAgentRun {tenant: $tenant})-[:USED_CONTEXT]->(m)
MATCH (run)-[:PRODUCED_DECISION]->(dec:CGDecision {tenant: $tenant})
RETURN DISTINCT dec.id AS id LIMIT $cap
"""

_DECISIONS_UNDER_OTHER_SCHEMA = """
MATCH (m:CGContextManifest {tenant: $tenant})
WHERE coalesce(m.ontology_version, '') <> $schema_version
MATCH (run:CGAgentRun {tenant: $tenant})-[:USED_CONTEXT]->(m)
MATCH (run)-[:PRODUCED_DECISION]->(dec:CGDecision {tenant: $tenant})
RETURN DISTINCT dec.id AS id LIMIT $cap
"""


class DependencyIndex:
    def __init__(self, neo4j_client, *, max_depth: int = 5, cap: int = 5000):
        self._neo4j = neo4j_client
        self.max_depth = max_depth
        self.cap = cap

    async def _read(self, closure: Closure, query: str, **params) -> list[dict]:
        rows = await run_read_only(self._neo4j, query, cap=self.cap, **params)
        if len(rows) >= self.cap:
            closure.truncated = True
        return rows

    async def resolve(self, event: InvalidationEvent) -> Closure:
        t = event.tenant
        c = Closure()
        for e in event.entities:
            c.entities[e.token] = e
        for r in event.relations:
            c.relation_keys.add(r.key)
            for ep in r.endpoints:
                c.entities[ep.token] = ep
        c.document_ids.update(event.document_ids)
        c.chunk_ids.update(event.chunk_ids)

        if c.document_ids:
            ids = sorted(c.document_ids)
            c.chunk_ids.update(r["id"] for r in await self._read(c, _DOC_CHUNKS, tenant=t, ids=ids))
            for row in await self._read(c, _DOC_RELATIONS, tenant=t, ids=ids):
                self._add_relation(c, row)

        # Fixpoint over inferred edges: an invalidated inferred edge is itself a
        # premise for further inferred edges (multi-hop downstream dependencies).
        frontier_keys, frontier_tokens = set(c.relation_keys), set(c.entities)
        while (frontier_keys or frontier_tokens) and c.depth_reached < self.max_depth:
            c.depth_reached += 1
            rows = await self._read(c, _DEPENDENT_INFERRED, tenant=t,
                                    keys=sorted(frontier_keys), tokens=sorted(frontier_tokens))
            frontier_keys, frontier_tokens = set(), set()
            for row in rows:
                key = self._add_relation(c, row)
                if key not in c.inferred_edges:
                    c.inferred_edges[key] = dict(row)
                    frontier_keys.add(key)
        if frontier_keys or frontier_tokens:
            c.truncated = True  # dependency chain longer than max_depth

        if c.entities:
            rows = await self._read(c, _ENTITY_CHUNKS_AND_IDS, tenant=t,
                                    entities=[{"name": e.name, "type": e.type} for e in c.entities.values()])
            for row in rows:
                if row.get("entity_id"):
                    c.entity_ids.add(row["entity_id"])
                c.chunk_ids.update(x for x in row.get("chunk_ids") or [] if x)

        chunks, docs = sorted(c.chunk_ids), sorted(c.document_ids)
        if chunks or docs or c.entity_ids:
            c.snapshot_ids.update(r["id"] for r in await self._read(
                c, _SNAPSHOTS, tenant=t, chunks=chunks, docs=docs, entity_ids=sorted(c.entity_ids)))
        if chunks or docs:
            c.decision_ids.update(r["id"] for r in await self._read(
                c, _DECISIONS, tenant=t, chunks=chunks, docs=docs))
        if event.kind is EventKind.SCHEMA_CHANGED and event.schema_version:
            c.decision_ids.update(r["id"] for r in await self._read(
                c, _DECISIONS_UNDER_OTHER_SCHEMA, tenant=t, schema_version=event.schema_version))
        return c

    @staticmethod
    def _add_relation(c: Closure, row: dict) -> str:
        from graphrag.graph.inference_engine import relation_key

        key = relation_key(row["src_type"], row["src_name"], row["relation"],
                           row["tgt_type"], row["tgt_name"])
        c.relation_keys.add(key)
        for name, typ in ((row["src_name"], row["src_type"]), (row["tgt_name"], row["tgt_type"])):
            ref = EntityRef(name=name, type=typ)
            c.entities.setdefault(ref.token, ref)
        return key
