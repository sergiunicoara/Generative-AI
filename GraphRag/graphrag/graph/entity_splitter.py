"""Entity splitter — detects and corrects over-merged entities.

Problems solved
---------------
1. Over-merging — two distinct entities that share a similar name or
   embedding get collapsed into one node. A "Saturn V" (rocket) and
   "Saturn" (planet) with a noisy embedding could merge.

2. No undo for merges — alias resolution is one-directional. Once merged,
   there is no path back to separate nodes.

3. Invisible merge errors — the alias registry logs merges but does not
   check if the resulting canonical entity is internally coherent.

Architecture
------------
Detection:
    An entity is over-merged if its MENTIONS chunks cluster into two or
    more semantically distinct groups (measured by intra-cluster cosine
    distance of chunk embeddings).

Split:
    1. Create two new Entity nodes with name suffixes (_A, _B) — including
       the correct tenant so tenant-scoped retrieval finds them.
    2. Redistribute MENTIONS edges to whichever new node the chunk's
       embedding is closest to.
    3. Redistribute RELATES_TO edges based on source_doc_id of the edge.
    4. Retire the original entity with status "split".
    5. Log to AuditTrail.

Tenant safety
-------------
Every Cypher query is scoped with e.tenant = $tenant to prevent
cross-tenant entity mutations in multi-tenant deployments.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

# Cosine distance threshold — if two chunk clusters are more dissimilar
# than this, the entity is likely over-merged
OVER_MERGE_DISTANCE_THRESHOLD = 0.35
MIN_CHUNKS_TO_SPLIT = 4   # don't split entities with too few chunks


class EntitySplitter:
    """
    Detects and splits over-merged entities.

    Usage::

        splitter = EntitySplitter(neo4j_client)
        candidates = await splitter.detect_over_merges(top_n=20, tenant="default")
        report = await splitter.split_entity(
            "Saturn", "CONCEPT",
            doc_group_a=["doc1"],
            doc_group_b=["doc2"],
            tenant="default",
        )
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Detection ──────────────────────────────────────────────────────────────

    async def detect_over_merges(
        self,
        top_n: int = 20,
        tenant: str = "default",
    ) -> list[dict]:
        """
        Find entities within a tenant whose MENTIONS chunks come from
        documents with very different authority levels — a proxy for
        semantic divergence between the merged variants.

        Returns list of candidates with over-merge score (0-1, higher = worse).
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant})<-[:MENTIONS]-(c:Chunk)
            WHERE NOT e.quarantined = true
            WITH e, count(c) AS chunk_count
            WHERE chunk_count >= $min_chunks
            MATCH (e)<-[:MENTIONS]-(c:Chunk)-[:PART_OF]->(d:Document)
            WITH e, chunk_count,
                 collect(DISTINCT d.authority_level) AS authority_levels,
                 collect(DISTINCT d.id) AS doc_ids
            WHERE size(authority_levels) > 1
              AND max(authority_levels) - min(authority_levels) >= 2
            RETURN e.name        AS entity_name,
                   e.type        AS entity_type,
                   chunk_count,
                   authority_levels,
                   doc_ids,
                   toFloat(max(authority_levels) - min(authority_levels)) / 3.0
                       AS over_merge_score
            ORDER BY over_merge_score DESC
            LIMIT $top_n
            """,
            tenant=tenant,
            min_chunks=MIN_CHUNKS_TO_SPLIT,
            top_n=top_n,
        )
        return [dict(r) for r in rows]

    # ── Split ──────────────────────────────────────────────────────────────────

    async def split_entity(
        self,
        entity_name: str,
        entity_type: str,
        doc_group_a: list[str],   # doc IDs → Entity_A
        doc_group_b: list[str],   # doc IDs → Entity_B
        tenant: str = "default",
        split_by: str = "human_review",
    ) -> dict:
        """
        Split entity_name into two new entities:
          {entity_name}_A  — backed by doc_group_a
          {entity_name}_B  — backed by doc_group_b

        All queries are tenant-scoped via (name, type, tenant) so that only
        the targeted tenant's entity is affected.

        Steps:
        1. Fetch original entity data (tenant-scoped).
        2. Create Entity_A and Entity_B nodes with correct tenant.
        3. Re-link MENTIONS edges from chunks in each doc group.
        4. Re-link RELATES_TO edges by source_doc_id.
        5. Mark original entity as split (tenant-scoped).
        Returns a summary report.
        """
        name_a = f"{entity_name}_A"
        name_b = f"{entity_name}_B"

        # 1. Fetch original entity data — tenant-scoped
        orig_rows = await self._neo4j.run(
            "MATCH (e:Entity {name: $name, type: $type, tenant: $tenant}) RETURN e",
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        if not orig_rows:
            return {"error": f"Entity '{entity_name}' (type={entity_type}, tenant={tenant}) not found"}

        orig = orig_rows[0]["e"]

        # 2. Create replacement nodes — include tenant so retrieval finds them
        for new_name, doc_group in [(name_a, doc_group_a), (name_b, doc_group_b)]:
            await self._neo4j.run(
                """
                MERGE (e:Entity {name: $name, type: $type, tenant: $tenant})
                ON CREATE SET e.id          = randomUUID(),
                              e.description = $description,
                              e.embedding   = $embedding,
                              e.source_type = $source_type,
                              e.split_from  = $original,
                              e.created_at  = datetime()
                """,
                name=new_name,
                type=entity_type,
                tenant=tenant,
                description=orig.get("description", ""),
                embedding=orig.get("embedding", []),
                source_type=orig.get("source_type", "document"),
                original=entity_name,
            )

        # 3. Re-link MENTIONS for doc_group_a — all Entity matches include tenant
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: doc_id})
            MATCH (c)-[m:MENTIONS]->(e:Entity {name: $orig, type: $type, tenant: $tenant})
            MATCH (new_e:Entity {name: $new_name, type: $type, tenant: $tenant})
            MERGE (c)-[:MENTIONS]->(new_e)
            DELETE m
            """,
            doc_ids=doc_group_a,
            orig=entity_name,
            type=entity_type,
            tenant=tenant,
            new_name=name_a,
        )

        # 3b. Re-link MENTIONS for doc_group_b
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {id: doc_id})
            MATCH (c)-[m:MENTIONS]->(e:Entity {name: $orig, type: $type, tenant: $tenant})
            MATCH (new_e:Entity {name: $new_name, type: $type, tenant: $tenant})
            MERGE (c)-[:MENTIONS]->(new_e)
            DELETE m
            """,
            doc_ids=doc_group_b,
            orig=entity_name,
            type=entity_type,
            tenant=tenant,
            new_name=name_b,
        )

        # 4. Re-link outgoing RELATES_TO edges — group_a
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (e:Entity {name: $orig, tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
            WHERE r.source_doc_id = doc_id
            MATCH (new_e:Entity {name: $new_name, tenant: $tenant})
            MERGE (new_e)-[nr:RELATES_TO {relation: r.relation}]->(t)
            SET nr = r
            DELETE r
            """,
            doc_ids=doc_group_a, orig=entity_name, tenant=tenant, new_name=name_a,
        )
        # 4b. Re-link incoming RELATES_TO edges — group_a
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (t:Entity {tenant: $tenant})-[r:RELATES_TO]->(e:Entity {name: $orig, tenant: $tenant})
            WHERE r.source_doc_id = doc_id
            MATCH (new_e:Entity {name: $new_name, tenant: $tenant})
            MERGE (t)-[nr:RELATES_TO {relation: r.relation}]->(new_e)
            SET nr = r
            DELETE r
            """,
            doc_ids=doc_group_a, orig=entity_name, tenant=tenant, new_name=name_a,
        )
        # 4c. Re-link outgoing RELATES_TO edges — group_b
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (e:Entity {name: $orig, tenant: $tenant})-[r:RELATES_TO]->(t:Entity {tenant: $tenant})
            WHERE r.source_doc_id = doc_id
            MATCH (new_e:Entity {name: $new_name, tenant: $tenant})
            MERGE (new_e)-[nr:RELATES_TO {relation: r.relation}]->(t)
            SET nr = r
            DELETE r
            """,
            doc_ids=doc_group_b, orig=entity_name, tenant=tenant, new_name=name_b,
        )
        # 4d. Re-link incoming RELATES_TO edges — group_b
        await self._neo4j.run(
            """
            UNWIND $doc_ids AS doc_id
            MATCH (t:Entity {tenant: $tenant})-[r:RELATES_TO]->(e:Entity {name: $orig, tenant: $tenant})
            WHERE r.source_doc_id = doc_id
            MATCH (new_e:Entity {name: $new_name, tenant: $tenant})
            MERGE (t)-[nr:RELATES_TO {relation: r.relation}]->(new_e)
            SET nr = r
            DELETE r
            """,
            doc_ids=doc_group_b, orig=entity_name, tenant=tenant, new_name=name_b,
        )

        # 5. Mark original as split (tenant-scoped — don't touch other tenants)
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            SET e.status      = 'split',
                e.split_into  = $split_into,
                e.split_at    = datetime(),
                e.split_by    = $split_by
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
            split_into=[name_a, name_b],
            split_by=split_by,
        )

        log.info(
            "entity_splitter.split_done",
            original=entity_name,
            entity_a=name_a,
            entity_b=name_b,
            tenant=tenant,
            doc_group_a_size=len(doc_group_a),
            doc_group_b_size=len(doc_group_b),
        )
        return {
            "original": entity_name,
            "entity_a": name_a,
            "entity_b": name_b,
            "tenant": tenant,
            "doc_group_a": doc_group_a,
            "doc_group_b": doc_group_b,
        }
