"""Agent that orchestrates the full document ingestion pipeline."""

from __future__ import annotations

import asyncio

import structlog

from graphrag.agents.base_agent import BaseGraphRAGAgent
from graphrag.core.config import get_settings
from graphrag.core.models import Document, IngestMessage
from graphrag.ingestion.chunker import chunk_document
from graphrag.ingestion.document_loader import load_document
from graphrag.ingestion.embedder import Embedder
from graphrag.ingestion.extractor import Extractor
from graphrag.ingestion.graph_writer import GraphWriter

log = structlog.get_logger(__name__)


class IngestionAgent(BaseGraphRAGAgent):
    def __init__(self):
        self._embedder = Embedder()
        self._extractor = Extractor()
        self._writer = GraphWriter()
        super().__init__("ingestion_agent")

    def _model(self) -> str:
        return get_settings().groq_model

    def _instruction(self) -> str:
        return (
            "You are a document ingestion agent. Your job is to load documents, "
            "chunk them, extract entities and relations, and write everything to the "
            "knowledge graph. Use the available tools in sequence: "
            "load → chunk → embed → extract → write."
        )

    def _tools(self) -> list:
        try:
            from google.adk.tools import FunctionTool
            from graphrag.agents.tools.neo4j_tools import search_graph
            return [FunctionTool(search_graph)]
        except ImportError:
            return []

    async def run(self, message: IngestMessage) -> dict:
        """Full ingestion pipeline: document → chunks → entities → Neo4j."""
        extracted = await self.extract(message)
        return await self.write(extracted)

    async def extract(self, message: IngestMessage) -> dict:
        """Phase 1 — pure compute, no Neo4j writes.

        Safe to run concurrently across documents: chunking, embedding, and
        LLM entity/relation extraction touch only this document's own data,
        never the shared Entity/alias graph. Call `write()` afterwards,
        serialized across documents, to commit the result.
        """
        doc = message.document
        job_id = message.job_id

        log.info("ingestion_agent.start", job_id=job_id, filename=doc.filename)

        # 1. Chunk
        chunks = chunk_document(doc)
        log.info("ingestion_agent.chunked", job_id=job_id, chunks=len(chunks))

        # 2. Embed chunks
        chunks = await self._embedder.embed_chunks(chunks)

        # 3. Extract entities + relations from each chunk, concurrently
        # bounded by a semaphore (LLM calls only — entity embedding is
        # batched separately below instead of one round-trip per entity).
        concurrency = get_settings().ingestion.get("extraction_concurrency", 5)
        semaphore = asyncio.Semaphore(concurrency)

        async def _extract_one(chunk):
            async with semaphore:
                return await self._extractor.extract(chunk)

        extraction_results = await asyncio.gather(
            *(_extract_one(chunk) for chunk in chunks)
        )

        # 4. Embed all entities across the whole document in one batched call
        # (was: one embed_text() round-trip per entity, serialized). See
        # lesson A131 — mirrors the chunk-embedding batching already in
        # place, applied to the per-entity loop A129#5 left untouched.
        all_entities_flat = [e for entities, _ in extraction_results for e in entities]
        if all_entities_flat:
            entity_embeddings = await self._embedder.embed_texts(
                [f"{e.name} {e.description}" for e in all_entities_flat]
            )
            for entity, emb in zip(all_entities_flat, entity_embeddings):
                entity.embedding = emb

        return {
            "job_id": job_id,
            "doc": doc,
            "chunks": chunks,
            "extraction_results": extraction_results,
        }

    async def write(self, extracted: dict) -> dict:
        """Phase 2 — Neo4j writes. Must run sequentially across documents:
        entity/alias dedup and contradiction detection need a consistent
        view of the shared graph as each document lands.
        """
        job_id = extracted["job_id"]
        doc    = extracted["doc"]
        chunks = extracted["chunks"]

        # 1. Write document node
        await self._writer.write_document(doc)

        # 2. Write chunks to Neo4j
        await self._writer.write_chunks(chunks)

        # 3. Write entities + relations, in chunk order, so AliasRegistry /
        # OntologyRegistry / contradiction-detection see chunks in document order.
        all_entities = []
        all_relations = []
        for chunk, (entities, relations) in zip(chunks, extracted["extraction_results"]):
            entity_map = {e.id: e for e in entities}

            await self._writer.write_entities(entities, chunk)
            await self._writer.write_relations(
                relations, entity_map, doc_id=doc.id, tenant=doc.tenant
            )

            all_entities.extend(entities)
            all_relations.extend(relations)

        maintenance_report = await self._writer.validate_and_check_cycles(
            doc_id=doc.id,
            tenant=doc.tenant,
        )

        # Optional post-write step: ground high-confidence entities in Wikidata.
        # Enabled via WIKIDATA_LINKING=1 env var (default off — avoids rate-limit
        # issues on large ingestion batches and keeps the pipeline fast).
        wikidata_links = 0
        if get_settings().wikidata_linking_enabled:
            try:
                from graphrag.graph.entity_linker import WikidataEntityLinker
                from graphrag.graph.neo4j_client import get_neo4j
                linker = WikidataEntityLinker(get_neo4j())
                # Only link entities with high confidence (≥0.85) to reduce API calls
                high_conf = [e for e in all_entities if e.confidence >= 0.85]
                for entity in high_conf[:20]:   # cap at 20 per document (rate limit)
                    try:
                        linked = await linker.link_entity(
                            entity.name, entity.type, doc.tenant
                        )
                        if linked:
                            wikidata_links += 1
                    except Exception as link_exc:
                        log.debug("ingestion_agent.wikidata_skip",
                                  entity=entity.name, error=str(link_exc)[:80])
                log.info("ingestion_agent.wikidata_linked",
                         job_id=job_id, linked=wikidata_links, candidates=len(high_conf))
            except ImportError:
                log.debug("ingestion_agent.wikidata_import_error")
            except Exception as exc:
                log.warning("ingestion_agent.wikidata_error", error=str(exc)[:120])

        log.info(
            "ingestion_agent.done",
            job_id=job_id,
            chunks=len(chunks),
            entities=len(all_entities),
            relations=len(all_relations),
            wikidata_links=wikidata_links,
            validation_issues=maintenance_report["validation"]["total_issues"],
            new_conflicts=maintenance_report["new_conflicts"],
        )
        return {
            "job_id": job_id,
            "doc_id": doc.id,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
            "wikidata_links": wikidata_links,
            "maintenance": maintenance_report,
        }
