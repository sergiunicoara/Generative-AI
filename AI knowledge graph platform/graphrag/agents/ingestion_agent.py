"""Google ADK agent that orchestrates the full document ingestion pipeline."""

from __future__ import annotations

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
        doc = message.document
        job_id = message.job_id

        log.info("ingestion_agent.start", job_id=job_id, filename=doc.filename)

        # 1. Write document node
        await self._writer.write_document(doc)

        # 2. Chunk
        chunks = chunk_document(doc)
        log.info("ingestion_agent.chunked", job_id=job_id, chunks=len(chunks))

        # 3. Embed chunks
        chunks = await self._embedder.embed_chunks(chunks)

        # 4. Write chunks to Neo4j
        await self._writer.write_chunks(chunks)

        # 5. Extract entities + relations from each chunk
        all_entities = []
        all_relations = []
        for chunk in chunks:
            entities, relations = await self._extractor.extract(chunk)
            entity_map = {e.id: e for e in entities}

            # Embed entities
            for entity in entities:
                entity.embedding = await self._embedder.embed_text(
                    f"{entity.name} {entity.description}",
                    task_type="retrieval_document",
                )

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
