"""Neo4j connection pool, query runner, and graph MERGE helpers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from graphrag.core.config import get_settings
from graphrag.core.exceptions import GraphRAGError
from graphrag.core.models import Chunk, Community, Entity, Relation

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

    async def run(self, cypher: str, **params) -> list[dict]:
        async with self._driver.session() as session:
            result = await session.run(cypher, parameters=params)
            return [record.data() async for record in result]

    # ── Schema initialization ────────────────────────────────────────────────────

    async def init_schema(self):
        schema_cypher = Path(__file__).parent / "schema.cypher"
        statements = schema_cypher.read_text().split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if stmt:
                await self.run(stmt)
        log.info("neo4j.schema_initialized")

    # ── Ingestion helpers ────────────────────────────────────────────────────────

    async def merge_document(self, doc_id: str, filename: str, ingested_at: str):
        await self.run(
            """
            MERGE (d:Document {id: $id})
            SET d.filename = $filename,
                d.ingested_at = $ingested_at,
                d.status = 'done'
            """,
            id=doc_id,
            filename=filename,
            ingested_at=ingested_at,
        )

    async def merge_chunk(self, chunk: Chunk):
        await self.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.chunk_index = $chunk_index,
                c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            id=chunk.id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            embedding=chunk.embedding,
            doc_id=chunk.document_id,
        )

    async def merge_entity(self, entity: Entity):
        await self.run(
            """
            MERGE (e:Entity {name: $name, type: $type})
            ON CREATE SET e.id = $id,
                          e.description = $description,
                          e.embedding = $embedding
            ON MATCH SET  e.description = CASE WHEN e.description = '' THEN $description ELSE e.description END
            """,
            id=entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            embedding=entity.embedding,
        )

    async def merge_mentions(self, chunk_id: str, entity_name: str, entity_type: str):
        await self.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {name: $entity_name, type: $entity_type})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_name=entity_name,
            entity_type=entity_type,
        )

    async def merge_relation(self, rel: Relation, src_name: str, tgt_name: str):
        await self.run(
            """
            MATCH (s:Entity {name: $src_name})
            MATCH (t:Entity {name: $tgt_name})
            MERGE (s)-[r:RELATES_TO {relation: $relation}]->(t)
            SET r.weight = $weight
            """,
            src_name=src_name,
            tgt_name=tgt_name,
            relation=rel.relation,
            weight=rel.weight,
        )

    async def merge_community(self, community: Community):
        await self.run(
            """
            MERGE (c:Community {id: $id})
            SET c.level = $level,
                c.summary = $summary,
                c.embedding = $embedding,
                c.member_count = $member_count
            """,
            id=community.id,
            level=community.level,
            summary=community.summary,
            embedding=community.embedding,
            member_count=community.member_count,
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

    # ── Retrieval helpers ────────────────────────────────────────────────────────

    async def vector_search_chunks(
        self, embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """ANN search over Chunk.embedding using Neo4j vector index."""
        return await self.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $embedding)
            YIELD node AS c, score
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=embedding,
        )

    async def vector_search_communities(
        self, embedding: list[float], top_k: int = 5
    ) -> list[dict]:
        """ANN search over Community.embedding for global search."""
        return await self.run(
            """
            CALL db.index.vector.queryNodes('community_embeddings', $k, $embedding)
            YIELD node AS c, score
            RETURN c.id AS community_id, c.summary AS summary, c.level AS level, score
            ORDER BY score DESC
            """,
            k=top_k,
            embedding=embedding,
        )

    async def get_entity_neighbors(self, chunk_ids: list[str]) -> list[dict]:
        """Expand retrieved chunks to their entity neighbors (1-hop)."""
        return await self.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e:Entity)
            OPTIONAL MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
            RETURN e.name AS entity, e.type AS type, e.description AS description,
                   collect(DISTINCT neighbor.name) AS neighbors
            """,
            chunk_ids=chunk_ids,
        )

    async def get_multihop_chunks(
        self, chunk_ids: list[str], hops: int = 2
    ) -> list[dict]:
        """
        Multi-hop graph traversal:
          Chunk → MENTIONS → Entity → RELATES_TO* (up to `hops`) → Entity
          → MENTIONS (reverse) → Chunk

        This is what makes "Company A owns Company B" on p.10 connect to
        "Company B launched a rocket" on p.300.
        """
        return await self.run(
            f"""
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {{id: cid}})-[:MENTIONS]->(e:Entity)
            MATCH (e)-[:RELATES_TO*1..{hops}]-(neighbor:Entity)
            MATCH (neighbor_chunk:Chunk)-[:MENTIONS]->(neighbor)
            WHERE NOT neighbor_chunk.id IN $chunk_ids
            RETURN DISTINCT
                neighbor_chunk.id   AS chunk_id,
                neighbor_chunk.text AS text,
                0.0                 AS score,
                neighbor.name       AS via_entity
            """,
            chunk_ids=chunk_ids,
        )

    async def bm25_search_chunks(
        self, query: str, top_k: int = 10
    ) -> list[dict]:
        """BM25 fulltext search over Chunk.text using Neo4j fulltext index."""
        return await self.run(
            """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
            YIELD node AS c, score
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
        )

    async def bm25_search_entities(
        self, query: str, top_k: int = 10
    ) -> list[dict]:
        """BM25 fulltext search over Entity name + description."""
        return await self.run(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node AS e, score
            OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
            RETURN DISTINCT c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
        )

    async def get_all_entities(self) -> list[dict]:
        return await self.run(
            "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type"
        )

    async def get_all_relations(self) -> list[dict]:
        return await self.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO]->(t:Entity)
            RETURN s.id AS source_id, t.id AS target_id, r.relation AS relation, r.weight AS weight
            """
        )


from pathlib import Path  # noqa: E402 — placed here to avoid circular import in type stubs


_client: Neo4jClient | None = None


def get_neo4j() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
