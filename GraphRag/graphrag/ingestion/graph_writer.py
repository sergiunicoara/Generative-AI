"""Write extracted chunks, entities, and relations into Neo4j."""

from __future__ import annotations

import structlog

from graphrag.core.models import Chunk, Document, Entity, Relation
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class GraphWriter:
    def __init__(self):
        self._neo4j = get_neo4j()

    async def write_document(self, doc: Document):
        await self._neo4j.merge_document(
            doc_id=doc.id,
            filename=doc.filename,
            ingested_at=doc.ingested_at.isoformat(),
        )
        log.info("graph_writer.document_merged", doc_id=doc.id)

    async def write_chunks(self, chunks: list[Chunk]):
        for chunk in chunks:
            await self._neo4j.merge_chunk(chunk)
        log.info("graph_writer.chunks_merged", count=len(chunks))

    async def write_entities(self, entities: list[Entity], chunk: Chunk):
        for entity in entities:
            await self._neo4j.merge_entity(entity)
            await self._neo4j.merge_mentions(chunk.id, entity.name, entity.type)
        log.info(
            "graph_writer.entities_merged",
            count=len(entities),
            chunk_id=chunk.id,
        )

    async def write_relations(
        self,
        relations: list[Relation],
        entity_map: dict[str, Entity],
    ):
        for rel in relations:
            src = entity_map.get(rel.source_entity_id)
            tgt = entity_map.get(rel.target_entity_id)
            if src and tgt:
                await self._neo4j.merge_relation(rel, src.name, tgt.name)
        log.info("graph_writer.relations_merged", count=len(relations))
