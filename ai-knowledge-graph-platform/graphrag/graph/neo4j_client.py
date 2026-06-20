"""Neo4j connection pool, query runner, and graph MERGE helpers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, TransientError

from graphrag.core.config import get_settings
from graphrag.core.exceptions import GraphRAGError
from graphrag.core.models import Chunk, Community, Entity, Relation
from graphrag.core.retry import with_retry

log = structlog.get_logger(__name__)


class Neo4jClient:
    """Thin wrapper around the Neo4j async driver with retry logic."""

    def __init__(self):
        cfg = get_settings()
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=50,
        )

    async def close(self):
        await self._driver.close()

    @with_retry(exceptions=(TransientError, ServiceUnavailable), max_attempts=3)
    async def run(self, cypher: str, **params) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            return [record.data() async for record in result]

    # ── Schema initialization ────────────────────────────────────────────────────

    async def init_schema(self):
        schema_cypher = Path(__file__).parent / "schema.cypher"
        raw = schema_cypher.read_text()
        for fragment in raw.split(";"):
            # Strip comment lines per-fragment (A59: never check the whole fragment
            # for "--" — that skips CREATE statements that follow a comment line)
            lines = [l for l in fragment.splitlines()
                     if not l.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if stmt:
                result = await self.run(stmt)
                # Consume result so DDL actually executes (A58)
                _ = result
        log.info("neo4j.schema_initialized")

    # ── Ingestion helpers ────────────────────────────────────────────────────────

    async def entity_exists(self, name: str, entity_type: str, tenant: str = "default") -> bool:
        """Check if an entity node already exists within the given tenant."""
        rows = await self.run(
            "MATCH (e:Entity {name: $name, type: $type, tenant: $tenant}) RETURN count(e) AS n",
            name=name,
            type=entity_type,
            tenant=tenant,
        )
        return bool(rows and rows[0]["n"] > 0)

    async def merge_document(
        self,
        doc_id: str,
        filename: str,
        ingested_at: str,
        authority_level: int = 4,
        valid_from: str | None = None,
        valid_to: str | None = None,
        tenant: str = "default",
    ):
        await self.run(
            """
            MERGE (d:Document {id: $id})
            SET d.filename        = $filename,
                d.ingested_at     = $ingested_at,
                d.status          = 'done',
                d.authority_level = $authority_level,
                d.valid_from      = $valid_from,
                d.valid_to        = $valid_to,
                d.tenant          = $tenant
            """,
            id=doc_id,
            filename=filename,
            ingested_at=ingested_at,
            authority_level=authority_level,
            valid_from=valid_from,
            valid_to=valid_to,
            tenant=tenant,
        )

    async def merge_chunk(self, chunk: Chunk, tenant: str = "default"):
        await self.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text        = $text,
                c.chunk_index = $chunk_index,
                c.embedding   = $embedding,
                c.tenant      = $tenant
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            id=chunk.id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding,
            doc_id=chunk.document_id,
            tenant=tenant,
        )

    async def merge_chunks_batch(self, chunks: list[Chunk], tenant: str = "default") -> None:
        """Same MERGE semantics as merge_chunk(), one round-trip for the batch.

        Caller (GraphWriter.write_chunks) sub-batches this — a single UNWIND
        carrying hundreds of 3072-dim chunk embeddings in one payload is the
        wrong tradeoff; keep payload size bounded like embedding_batch_size."""
        if not chunks:
            return
        rows = [
            {
                "id": c.id,
                "text": c.text,
                "chunk_index": c.chunk_index,
                "embedding": c.embedding,
                "doc_id": c.document_id,
            }
            for c in chunks
        ]
        await self.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {id: row.id})
            SET c.text        = row.text,
                c.chunk_index = row.chunk_index,
                c.embedding   = row.embedding,
                c.tenant      = $tenant
            WITH c, row
            MATCH (d:Document {id: row.doc_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_entity(self, entity: Entity, tenant: str = "default"):
        """Merge entity scoped to tenant — same (name, type) in different tenants are distinct nodes."""
        await self.run(
            """
            MERGE (e:Entity {name: $name, type: $type, tenant: $tenant})
            ON CREATE SET e.id               = $id,
                          e.description      = $description,
                          e.embedding        = $embedding,
                          e.source_type      = $source_type,
                          e.source_doc_id    = $source_doc_id,
                          e.extraction_model = $extraction_model,
                          e.prompt_version   = $prompt_version,
                          e.created_at       = datetime(),
                          e.recorded_at      = datetime()   // transaction time — never updated
            ON MATCH SET  e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END,
                          e.embedding   = CASE WHEN $embedding IS NOT NULL AND size($embedding) > 0 THEN $embedding ELSE e.embedding END,
                          e.updated_at  = datetime()
            """,
            id=entity.id,
            name=entity.name,
            type=entity.type,
            tenant=tenant,
            description=entity.description,
            embedding=entity.embedding,
            source_type=entity.source_type if isinstance(entity.source_type, str) else entity.source_type.value,
            source_doc_id=entity.source_doc_id,
            extraction_model=entity.extraction_model,
            prompt_version=entity.prompt_version,
        )

    async def merge_mentions(self, chunk_id: str, entity_name: str, entity_type: str, tenant: str = "default"):
        await self.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {name: $entity_name, type: $entity_type, tenant: $tenant})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
        )

    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> None:
        """Same MERGE semantics as merge_entity(), one round-trip for the batch."""
        if not entities:
            return
        rows = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "embedding": e.embedding,
                "source_type": e.source_type if isinstance(e.source_type, str) else e.source_type.value,
                "source_doc_id": e.source_doc_id,
                "extraction_model": e.extraction_model,
                "prompt_version": e.prompt_version,
            }
            for e in entities
        ]
        await self.run(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.name, type: row.type, tenant: $tenant})
            ON CREATE SET e.id               = row.id,
                          e.description      = row.description,
                          e.embedding        = row.embedding,
                          e.source_type      = row.source_type,
                          e.source_doc_id    = row.source_doc_id,
                          e.extraction_model = row.extraction_model,
                          e.prompt_version   = row.prompt_version,
                          e.created_at       = datetime(),
                          e.recorded_at      = datetime()
            ON MATCH SET  e.description = CASE WHEN e.description = '' THEN row.description ELSE e.description END,
                          e.embedding   = CASE WHEN row.embedding IS NOT NULL AND size(row.embedding) > 0 THEN row.embedding ELSE e.embedding END,
                          e.updated_at  = datetime()
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_mentions_batch(
        self, chunk_id: str, entity_refs: list[tuple[str, str]], tenant: str = "default"
    ) -> None:
        """Same MERGE semantics as merge_mentions(), one round-trip for the batch."""
        if not entity_refs:
            return
        rows = [{"name": name, "type": etype} for name, etype in entity_refs]
        await self.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            UNWIND $rows AS row
            MATCH (e:Entity {name: row.name, type: row.type, tenant: $tenant})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            rows=rows,
            tenant=tenant,
        )

    async def merge_relation(
        self,
        rel: Relation,
        src_name: str,
        src_type: str,
        tgt_name: str,
        tgt_type: str,
        tenant: str = "default",
    ):
        """Write a RELATES_TO edge, matching endpoints by (name, type, tenant).

        Including ``type`` in the MATCH prevents ambiguous matches when a
        tenant has two entities with the same name but different types (e.g.
        "Apple" as ORG vs. PRODUCT).
        """
        await self.run(
            """
            MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
            MATCH (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
            MERGE (s)-[r:RELATES_TO {relation: $relation}]->(t)
            ON CREATE SET r.recorded_at = datetime()   // transaction time — set once, never updated
            SET r.weight           = $weight,
                r.extracted_at     = $extracted_at,
                r.source_doc_id    = $source_doc_id,
                r.source_type      = $source_type,
                r.constraint_type  = $constraint_type,
                r.valid_from       = $valid_from,
                r.valid_to         = $valid_to,
                r.tenant           = $tenant,
                // Accumulate all contributing document IDs as a list so that
                // contradiction detection can see every source even after
                // multiple merges collapse to a single edge.
                r.source_doc_ids   = CASE
                    WHEN r.source_doc_ids IS NULL         THEN [$source_doc_id]
                    WHEN $source_doc_id IN r.source_doc_ids THEN r.source_doc_ids
                    ELSE r.source_doc_ids + [$source_doc_id]
                END,
                r.confidence       = CASE
                    WHEN r.confidence IS NULL THEN $confidence
                    ELSE 1.0 - (1.0 - r.confidence) * (1.0 - $confidence)
                END
            """,
            src_name=src_name,
            src_type=src_type,
            tgt_name=tgt_name,
            tgt_type=tgt_type,
            tenant=tenant,
            relation=rel.relation,
            weight=rel.weight,
            confidence=rel.confidence,
            extracted_at=rel.extracted_at.isoformat(),
            source_doc_id=rel.source_doc_id,
            source_type=rel.source_type if isinstance(rel.source_type, str) else rel.source_type.value,
            constraint_type=rel.constraint_type if isinstance(rel.constraint_type, str) else rel.constraint_type.value,
            valid_from=rel.valid_from.isoformat() if rel.valid_from else None,
            valid_to=rel.valid_to.isoformat() if rel.valid_to else None,
        )
        # Store deep provenance if present — scoped to the exact (name, type, tenant) edge
        if rel.chunk_span_start is not None or rel.extraction_model:
            await self.run(
                """
                MATCH (s:Entity {name: $src_name, type: $src_type, tenant: $tenant})
                      -[r:RELATES_TO {relation: $relation}]->
                      (t:Entity {name: $tgt_name, type: $tgt_type, tenant: $tenant})
                SET r.chunk_span_start = $span_start,
                    r.chunk_span_end   = $span_end,
                    r.extraction_model = $extraction_model,
                    r.prompt_version   = $prompt_version
                """,
                src_name=src_name,
                src_type=src_type,
                tgt_name=tgt_name,
                tgt_type=tgt_type,
                tenant=tenant,
                relation=rel.relation,
                span_start=rel.chunk_span_start,
                span_end=rel.chunk_span_end,
                extraction_model=rel.extraction_model,
                prompt_version=rel.prompt_version,
            )

    async def merge_relations_batch(self, rows: list[dict], tenant: str = "default") -> None:
        """Batched equivalent of merge_relation() — one round-trip for the
        whole batch, combining both of merge_relation()'s queries (main edge
        properties + provenance) into a single UNWIND pass since they target
        the same edge. Each row needs the same keys merge_relation() takes as
        kwargs (src_name, src_type, tgt_name, tgt_type, relation, weight,
        confidence, extracted_at, source_doc_id, source_type, constraint_type,
        valid_from, valid_to, span_start, span_end, extraction_model,
        prompt_version).
        """
        if not rows:
            return
        await self.run(
            """
            UNWIND $rows AS row
            MATCH (s:Entity {name: row.src_name, type: row.src_type, tenant: $tenant})
            MATCH (t:Entity {name: row.tgt_name, type: row.tgt_type, tenant: $tenant})
            MERGE (s)-[r:RELATES_TO {relation: row.relation}]->(t)
            ON CREATE SET r.recorded_at = datetime()
            SET r.weight           = row.weight,
                r.extracted_at     = row.extracted_at,
                r.source_doc_id    = row.source_doc_id,
                r.source_type      = row.source_type,
                r.constraint_type  = row.constraint_type,
                r.valid_from       = row.valid_from,
                r.valid_to         = row.valid_to,
                r.tenant           = $tenant,
                r.source_doc_ids   = CASE
                    WHEN r.source_doc_ids IS NULL            THEN [row.source_doc_id]
                    WHEN row.source_doc_id IN r.source_doc_ids THEN r.source_doc_ids
                    ELSE r.source_doc_ids + [row.source_doc_id]
                END,
                r.confidence       = CASE
                    WHEN r.confidence IS NULL THEN row.confidence
                    ELSE 1.0 - (1.0 - r.confidence) * (1.0 - row.confidence)
                END,
                r.chunk_span_start = row.span_start,
                r.chunk_span_end   = row.span_end,
                r.extraction_model = row.extraction_model,
                r.prompt_version   = row.prompt_version
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_community(self, community: Community):
        await self.run(
            """
            MERGE (c:Community {id: $id})
            SET c.level = $level,
                c.summary = $summary,
                c.embedding = $embedding,
                c.member_count = $member_count,
                c.tenant = $tenant
            """,
            id=community.id,
            level=community.level,
            summary=community.summary,
            embedding=community.embedding,
            member_count=community.member_count,
            tenant=community.tenant,
        )
        for entity_id in community.member_entity_ids:
            await self.run(
                """
                MATCH (e:Entity {id: $entity_id})
                MATCH (c:Community {id: $community_id})
                MERGE (e)-[:MEMBER_OF]->(c)
                """,
                entity_id=entity_id,
                community_id=community.id,
            )

    async def clear_communities(self, tenant: str = "default") -> None:
        await self.run(
            """
            MATCH (c:Community)
            WHERE c.tenant = $tenant
            DETACH DELETE c
            """,
            tenant=tenant,
        )

    # ── Retrieval helpers ────────────────────────────────────────────────────────

    async def vector_search_chunks(
        self, embedding: list[float], top_k: int = 10, tenant: str = "default"
    ) -> list[dict]:
        """ANN search over Chunk.embedding using Neo4j vector index.
        Filters by tenant and excludes chunks whose mentioned entities are quarantined.
        """
        return await self.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
            YIELD node AS c, score
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
              AND NOT EXISTS {
                  MATCH (c)-[:MENTIONS]->(e:Entity)
                  WHERE e.quarantined = true
              }
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=embedding,
            tenant=tenant,
        )

    async def get_document_filenames(self, tenant: str = "default") -> list[str]:
        """List distinct document filenames for a tenant (for named-document
        matching against question text — see local_search's named-doc boost).
        """
        rows = await self.run(
            "MATCH (d:Document) WHERE ($tenant = 'default' OR d.tenant = $tenant) "
            "RETURN DISTINCT d.filename AS filename",
            tenant=tenant,
        )
        return [r["filename"] for r in rows if r.get("filename")]

    async def get_chunk_filenames(
        self, chunk_ids: list[str], tenant: str = "default"
    ) -> dict[str, str]:
        """Map chunk_id -> source document filename for a set of chunks.

        Used by the named-document boost to check whether any already-fused
        candidate belongs to the named document before falling back to a
        fresh cosine search.
        """
        if not chunk_ids:
            return {}
        rows = await self.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document)
            WHERE c.id IN $chunk_ids
              AND ($tenant = 'default' OR c.tenant = $tenant)
            RETURN c.id AS chunk_id, d.filename AS filename
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
        )
        return {r["chunk_id"]: r["filename"] for r in rows if r.get("filename")}

    async def get_best_chunk_for_document(
        self, filename: str, embedding: list[float], tenant: str = "default"
    ) -> dict | None:
        """Best chunk (by cosine similarity to `embedding`) belonging to the
        document with this exact filename. Used by the named-document boost:
        when the question explicitly names a document, guarantee its most
        relevant chunk a seed slot even if it didn't survive fused-ranking.
        """
        rows = await self.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {filename: $filename})
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
              AND c.embedding IS NOT NULL
            RETURN c.id AS chunk_id, c.text AS text,
                   vector.similarity.cosine(c.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT 1
            """,
            filename=filename,
            embedding=embedding,
            tenant=tenant,
        )
        return rows[0] if rows else None

    async def vector_search_communities(
        self,
        embedding: list[float],
        top_k: int = 5,
        tenant: str = "default",
    ) -> list[dict]:
        """ANN search over Community.embedding for global search."""
        return await self.run(
            """
            CALL db.index.vector.queryNodes('community_embeddings', $k, $embedding)
            YIELD node AS c, score
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            RETURN c.id AS community_id, c.summary AS summary, c.level AS level, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=embedding,
            tenant=tenant,
        )

    async def get_entity_neighbors(
        self,
        chunk_ids: list[str],
        as_of: str | None = None,
        tenant: str = "default",
    ) -> list[dict]:
        """Expand retrieved chunks to their entity neighbors (1-hop).
        Excludes quarantined entities. Optionally filters edges by valid_to.
        """
        temporal_filter = (
            "AND (r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of)"
            if as_of else ""
        )
        return await self.run(
            f"""
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {{id: cid}})-[:MENTIONS]->(e:Entity)
            WHERE coalesce(e.quarantined, false) = false
            OPTIONAL MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
            OPTIONAL MATCH (src_doc:Document {{id: r.source_doc_id}})
            WHERE coalesce(neighbor.quarantined, false) = false {temporal_filter}
              AND ($tenant = 'default' OR r.source_doc_id IS NULL OR src_doc.tenant = $tenant)
            RETURN e.name AS entity, e.type AS type, e.description AS description,
                   collect(DISTINCT neighbor.name) AS neighbors
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
            **({"as_of": as_of} if as_of else {}),
        )

    async def get_multihop_chunks(
        self,
        chunk_ids: list[str],
        hops: int = 2,
        as_of: str | None = None,
        tenant: str = "default",
        query_embedding: list[float] | None = None,
        semantic_weight: float = 0.0,
    ) -> list[dict]:
        """
        Multi-hop graph traversal with temporal filtering and path quality scoring.

        Returns hop chunks with:
          - path_length: number of RELATES_TO hops taken
          - path_confidence: product of edge confidences along the path
          - via_entity: name of the bridging entity

        Temporal filter: only traverses edges valid at `as_of` datetime.
        Quarantine filter: skips quarantined entities.
        Tenant filter: only returns chunks for the given tenant.

        Semantic blend: when `query_embedding` is given and `semantic_weight`
        > 0, ranking becomes
            (1-w) * (path_confidence / path_length) + w * cos(chunk_emb, query_emb)
        with the cosine computed *inside Neo4j* (`vector.similarity.cosine`),
        so no embeddings cross the wire. The caller caps hop chunks (e.g. at
        50) before GNN scoring — pure topology ranking can push semantically
        relevant chunks below that cap on dense graphs. Chunks without an
        embedding fall back to the pure path score.
        """
        temporal_filter = (
            "AND ALL(r IN relationships(path) WHERE "
            "(r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of))"
            if as_of else ""
        )
        tenant_filter = (
            "AND ALL(r IN relationships(path) WHERE "
            "r.source_doc_id IS NULL OR EXISTS { "
            "MATCH (d:Document {id: r.source_doc_id}) "
            "WHERE d.tenant = $tenant })"
            if tenant != "default" else ""
        )
        use_semantic = query_embedding is not None and semantic_weight > 0
        score_expr = (
            # blend graph-path quality with query similarity; null-safe fallback
            "CASE WHEN sem_sim IS NULL THEN base_score "
            "ELSE (1.0 - $sem_w) * base_score + $sem_w * sem_sim END"
            if use_semantic else "base_score"
        )
        sem_sim_expr = (
            "CASE WHEN neighbor_chunk.embedding IS NULL THEN NULL "
            "ELSE vector.similarity.cosine(neighbor_chunk.embedding, $query_emb) END"
            if use_semantic else "NULL"
        )
        results = await self.run(
            f"""
            UNWIND $chunk_ids AS cid
            CALL {{
                WITH cid
                MATCH (c:Chunk {{id: cid}})-[:MENTIONS]->(e:Entity)
                WHERE coalesce(e.quarantined, false) = false
                MATCH path = (e)-[:RELATES_TO*1..{hops}]-(neighbor:Entity)
                WHERE coalesce(neighbor.quarantined, false) = false {temporal_filter} {tenant_filter}
                  AND ALL(n IN nodes(path) WHERE coalesce(n.quarantined, false) = false)
                MATCH (neighbor_chunk:Chunk)-[:MENTIONS]->(neighbor)
                WHERE NOT neighbor_chunk.id IN $chunk_ids
                  AND ($tenant = 'default' OR neighbor_chunk.tenant = $tenant)
                RETURN DISTINCT
                    neighbor_chunk.id   AS chunk_id,
                    neighbor_chunk.text AS text,
                    neighbor.name       AS via_entity,
                    length(path)        AS path_length,
                    reduce(conf = 1.0, r IN relationships(path) |
                        conf * coalesce(r.confidence, 1.0)) AS path_confidence,
                    {sem_sim_expr} AS sem_sim
                // unordered cap: bounds traversal per seed chunk so a single
                // high-degree hub entity can't blow up the path enumeration
                LIMIT $per_seed_cap
            }}
            // path score: penalise longer paths, reward high-confidence paths
            WITH chunk_id, text, via_entity, path_length, path_confidence, sem_sim,
                 (path_confidence / toFloat(path_length)) AS base_score
            RETURN chunk_id, text, via_entity, path_length, path_confidence,
                   sem_sim, {score_expr} AS path_score
            ORDER BY path_score DESC
            LIMIT $total_cap
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
            per_seed_cap=200,
            total_cap=500,
            **({"as_of": as_of} if as_of else {}),
            **({"query_emb": query_embedding, "sem_w": float(semantic_weight)}
               if use_semantic else {}),
        )
        # Normalise to list[dict] with score field for downstream compatibility
        for row in results:
            row["score"] = float(row.get("path_score") or 0.0)
        return results

    async def bm25_search_chunks(
        self, query: str, top_k: int = 10, tenant: str = "default"
    ) -> list[dict]:
        """BM25 fulltext search over Chunk.text using Neo4j fulltext index.
        Filters by tenant and excludes quarantined entity chunks.
        """
        return await self.run(
            """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
            YIELD node AS c, score
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
              AND NOT EXISTS {
                  MATCH (c)-[:MENTIONS]->(e:Entity)
                  WHERE e.quarantined = true
              }
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
            tenant=tenant,
        )

    async def bm25_search_entities(
        self,
        query: str,
        top_k: int = 10,
        tenant: str = "default",
    ) -> list[dict]:
        """BM25 fulltext search over Entity name + description.
        Excludes quarantined entities.
        """
        return await self.run(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node AS e, score
            WHERE coalesce(e.quarantined, false) = false
            OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
            RETURN DISTINCT c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
            tenant=tenant,
        )

    async def get_chunk_entity_embeddings(
        self, chunk_ids: list[str]
    ) -> list[dict]:
        """Return entity embeddings for all entities mentioned by the given chunks.

        Used by GNNScorer to build the node-feature matrix H.
        Only returns entities that actually have a stored embedding.
        Excludes quarantined entities.
        """
        return await self.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e:Entity)
            WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
              AND coalesce(e.quarantined, false) = false
            RETURN cid          AS chunk_id,
                   e.name       AS entity_name,
                   e.type       AS entity_type,
                   e.embedding  AS embedding,
                   COUNT { (e)-[:RELATES_TO]-() } AS degree
            """,
            chunk_ids=chunk_ids,
        )

    async def get_entity_relations_subgraph(
        self,
        entities: list[dict],
        as_of: str | None = None,
        tenant: str = "default",
    ) -> list[dict]:
        """Return RELATES_TO edges between a set of entities.

        ``entities`` is a list of ``{"name": str, "type": str}`` dicts.
        Using (name, type) pairs rather than names alone avoids ambiguous
        matches when a tenant has two entities with the same name but
        different types (e.g. "Apple" as ORG vs. PRODUCT).

        Used by GNNScorer to build the adjacency matrix A.
        Only intra-subgraph edges are returned (both endpoints in the list).
        Optionally filters edges by temporal validity.
        """
        if not entities:
            return []
        temporal_filter = (
            "AND (r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of)"
            if as_of else ""
        )
        # Build a set-membership key of the form "name:type" for the target
        # side filter so both dimensions are checked without a subquery.
        return await self.run(
            f"""
            UNWIND $entities AS pair
            MATCH (s:Entity {{name: pair.name, type: pair.type, tenant: $tenant}})
                  -[r:RELATES_TO]->
                  (t:Entity {{tenant: $tenant}})
            WHERE (t.name + ':' + t.type) IN $entity_keys {temporal_filter}
              AND coalesce(s.quarantined, false) = false
              AND coalesce(t.quarantined, false) = false
            RETURN s.name                             AS src,
                   s.type                             AS src_type,
                   t.name                             AS tgt,
                   t.type                             AS tgt_type,
                   r.weight                           AS weight,
                   coalesce(r.confidence, 1.0)        AS confidence,
                   r.extracted_at                     AS extracted_at,
                   r.source_doc_id                    AS source_doc_id
            """,
            entities=entities,
            entity_keys=[f"{e['name']}:{e['type']}" for e in entities],
            tenant=tenant,
            **({"as_of": as_of} if as_of else {}),
        )

    async def get_all_entities(self, tenant: str = "default") -> list[dict]:
        return await self.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE ($tenant = 'default' OR c.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            RETURN DISTINCT e.id AS id, e.name AS name, e.type AS type
            """,
            tenant=tenant,
        )

    async def get_all_relations(self, tenant: str = "default") -> list[dict]:
        return await self.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            OPTIONAL MATCH (d:Document {id: r.source_doc_id})
            WHERE coalesce(s.quarantined, false) = false
              AND coalesce(t.quarantined, false) = false
              AND ($tenant = 'default' OR r.source_doc_id IS NULL OR d.tenant = $tenant)
            RETURN s.id AS source_id, t.id AS target_id, r.relation AS relation,
                   coalesce(r.weight, r.confidence, 1.0) AS weight
            """,
            tenant=tenant,
        )


_client: Neo4jClient | None = None


def get_neo4j() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
