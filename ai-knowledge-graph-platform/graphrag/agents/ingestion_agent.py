"""Agent that orchestrates the full document ingestion pipeline."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from inspect import isawaitable
from time import perf_counter

import structlog

from graphrag.agents.base_agent import BaseGraphRAGAgent
from graphrag.core.config import get_settings
from graphrag.core.models import IngestMessage, IngestionRunManifest, StructuredTable
from graphrag.observability.correlation import current_correlation_id
from graphrag.ingestion.chunker import chunk_document
from graphrag.ingestion.embedder import Embedder
from graphrag.ingestion.extractor import Extractor
from graphrag.ingestion.graph_writer import GraphWriter
from graphrag.ingestion.intelligence import (
    IntelligenceArtifactExtractor,
    mine_explicit_aliases,
    temporal_periods,
)
from graphrag.enterprise.lineage import LineageService
from graphrag.enterprise.metadata_governance import MetadataGovernanceService

log = structlog.get_logger(__name__)


class IngestionAgent(BaseGraphRAGAgent):
    def __init__(self):
        self._embedder = Embedder()
        self._extractor = Extractor()
        self._writer = GraphWriter()
        self._artifact_extractor = IntelligenceArtifactExtractor(self._model())
        self._metadata_governance = MetadataGovernanceService(self._writer.neo4j_client)
        self._lineage = LineageService(self._writer.neo4j_client)
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

    async def run(self, message: IngestMessage) -> dict:
        """Full ingestion pipeline: document → chunks → entities → Neo4j."""
        # This bootstrap is deliberately before extraction: otherwise a cold
        # worker would skip the strict ontology + SHACL mutation gates for its
        # first document and only load the vocabulary at write time.
        await self._writer.ensure_ontology_schema(message.document.tenant)
        extracted = await self.extract(message)
        try:
            return await self.write(extracted)
        except Exception as exc:
            manifest = extracted.get("manifest")
            if manifest and manifest.document_id:
                manifest.status = "failed"
                manifest.error = f"{type(exc).__name__}: {str(exc)[:300]}"
                manifest.completed_at = datetime.now(timezone.utc)
                manifest.integrity_hash = manifest.compute_integrity_hash()
                try:
                    await self._writer.write_ingestion_manifest(manifest)
                except Exception as manifest_exc:  # noqa: BLE001 - preserve original ingestion error
                    log.warning("ingestion_agent.manifest_failure_unrecorded", error=str(manifest_exc)[:120])
            raise

    async def extract(self, message: IngestMessage) -> dict:
        """Phase 1 — pure compute, no Neo4j writes.

        Safe to run concurrently across documents: chunking, embedding, and
        LLM entity/relation extraction touch only this document's own data,
        never the shared Entity/alias graph. Call `write()` afterwards,
        serialized across documents, to commit the result.
        """
        started = perf_counter()
        doc = message.document
        job_id = message.job_id
        manifest = IngestionRunManifest(
            job_id=job_id,
            tenant=doc.tenant,
            filename=doc.filename,
            content_hash=doc.content_hash,
            correlation_id=current_correlation_id(),
            model_provider="configured_router",
            model_version=self._model(),
            prompt_versions={"entity_relation": "v1", "intelligence": "intelligence-v1"},
        )

        log.info("ingestion_agent.start", job_id=job_id, filename=doc.filename)

        # 1. Chunk
        chunks = chunk_document(doc)
        chunk_elapsed_ms = (perf_counter() - started) * 1000
        log.info("ingestion_agent.chunked", job_id=job_id, chunks=len(chunks))

        # 2. Embed chunks
        embed_started = perf_counter()
        chunks = await self._embedder.embed_chunks(chunks)
        embed_elapsed_ms = (perf_counter() - embed_started) * 1000

        # 3. Extract entities + relations from each chunk, concurrently
        # bounded by a semaphore (LLM calls only — entity embedding is
        # batched separately below instead of one round-trip per entity).
        concurrency = get_settings().ingestion.get("extraction_concurrency", 5)
        semaphore = asyncio.Semaphore(concurrency)

        async def _extract_one(chunk):
            async with semaphore:
                return await self._extractor.extract(chunk)

        extraction_started = perf_counter()
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

        artifact_results = [[] for _ in chunks]
        if get_settings().ingestion.get("intelligence_artifacts_enabled", True):
            artifact_extractor = getattr(self, "_artifact_extractor", None)
            if artifact_extractor is not None:
                async def _extract_artifacts(chunk, entities):
                    async with semaphore:
                        return await artifact_extractor.extract(chunk, [entity.name for entity in entities])
                artifact_results = await asyncio.gather(
                    *(_extract_artifacts(chunk, entities) for chunk, (entities, _) in zip(chunks, extraction_results))
                )
        extraction_elapsed_ms = (perf_counter() - extraction_started) * 1000
        alias_results = [
            mine_explicit_aliases(chunk.text, [entity.name for entity in entities])
            for chunk, (entities, _) in zip(chunks, extraction_results)
        ]
        temporal_results = (
            [temporal_periods(chunk.text) for chunk in chunks]
            if get_settings().ingestion.get("temporal_hierarchy_enabled", True)
            else [[] for _ in chunks]
        )
        manifest.stage_metrics = {
            "chunking": {"duration_ms": round(chunk_elapsed_ms, 3), "items": len(chunks), "cost_usd": None},
            "embedding": {"duration_ms": round(embed_elapsed_ms, 3), "items": len(chunks) + len(all_entities_flat), "cost_usd": None},
            "extraction": {"duration_ms": round(extraction_elapsed_ms, 3), "items": len(all_entities_flat), "cost_usd": None},
            "cost_status": {"value": "provider_usage_is_recorded_in_telemetry_when_available; manifest_never_invents_missing_cost"},
        }

        return {
            "job_id": job_id,
            "doc": doc,
            "chunks": chunks,
            "extraction_results": extraction_results,
            "artifact_results": artifact_results,
            "alias_results": alias_results,
            "temporal_results": temporal_results,
            "manifest": manifest,
        }

    async def write(self, extracted: dict) -> dict:
        """Phase 2 — Neo4j writes. Must run sequentially across documents:
        entity/alias dedup and contradiction detection need a consistent
        view of the shared graph as each document lands.
        """
        job_id = extracted["job_id"]
        doc    = extracted["doc"]
        chunks = extracted["chunks"]
        manifest = extracted.get("manifest")

        # Validate before opening the corpus mutation. A collection-specific
        # contract failure must not leave a partial graph revision behind.
        metadata_governance = getattr(self, "_metadata_governance", None)
        if metadata_governance is not None:
            await metadata_governance.validate(doc.metadata_envelope, doc.tenant)

        # Cache readers fail open to live retrieval while this tenant is being
        # mutated. The revision is advanced only after all writes and checks
        # below complete, so no answer can be cached against a partial ingest.
        await self._writer.begin_corpus_update(doc.tenant)

        # 1. Write document node. write_document may return a DIFFERENT id than
        # doc.id came in with — merge_document keys on (tenant, filename), so a
        # re-ingested document resolves to its existing id, not the fresh
        # uuid4() this run generated for `doc`. Every chunk was built during
        # extract() with the old id, so they must be repointed before anything
        # downstream (chunk write, relations, supersession) references them —
        # otherwise those writes target a document node that doesn't exist and
        # silently create a duplicate (see tasks/lessons.md A136).
        original_id = doc.id
        canonical_id = await self._writer.write_document(doc)  # mutates doc.id in place
        for chunk in chunks:
            # Carries source context into entity assertions without trusting a
            # client-supplied query value. Chunker already sets this for normal
            # ingestion; this also covers older queued documents and custom
            # connectors that construct chunks directly.
            chunk.metadata.setdefault("source_system", doc.metadata_envelope.source_system)
        is_reingest = canonical_id != original_id
        if is_reingest:
            for c in chunks:
                c.document_id = canonical_id
            # Existing chunks keep stable identities across re-ingestion. Clear
            # their old mentions and this document's relation evidence before
            # writing the newly extracted evidence, so facts that disappeared
            # from the revised source cannot remain retrievable.
            await self._writer.reconcile_document_evidence(
                doc_id=canonical_id, tenant=doc.tenant
            )

        if manifest is not None:
            manifest.document_id = doc.id
            manifest.integrity_hash = manifest.compute_integrity_hash()
            await self._writer.write_ingestion_manifest(manifest)

        # 2. Write chunks to Neo4j
        await self._writer.write_chunks(chunks)
        if metadata_governance is not None:
            await metadata_governance.record_document(
                doc.id, doc.metadata_envelope, doc.tenant,
            )
        structured_tables = []
        for raw_table in doc.metadata.get("structured_tables", []):
            try:
                table = StructuredTable.model_validate(raw_table)
                table.document_id = doc.id
                table.tenant = doc.tenant
                structured_tables.append(table)
            except (TypeError, ValueError):
                log.warning("ingestion_agent.structured_table_rejected", filename=doc.filename)
        if structured_tables:
            await self._writer.write_structured_tables(structured_tables, tenant=doc.tenant)

        # High-stakes lineage and obligations are intentionally not written as
        # active graph facts by ingestion. Their text evidence is captured in a
        # tenant-scoped review queue; a human approval materialises the edge or
        # obligation register record later.
        lineage_reviews = []
        obligation_reviews = []
        lineage_service = getattr(self, "_lineage", None)
        if lineage_service is not None:
            for assertion in doc.lineage_assertions:
                lineage_reviews.append(await lineage_service.submit_lineage(doc.id, assertion, doc.tenant))
            for draft in doc.obligation_drafts:
                obligation_reviews.append(await lineage_service.submit_obligation(doc.id, draft, doc.tenant))

        # 3. Write entities + relations, in chunk order, so AliasRegistry /
        # OntologyRegistry / contradiction-detection see chunks in document order.
        all_entities = []
        all_relations = []
        all_artifacts = []
        explicit_aliases = 0
        ontology_proposals = 0
        for index, (chunk, (entities, relations)) in enumerate(zip(chunks, extracted["extraction_results"])):
            entity_map = {e.id: e for e in entities}

            proposal_payload = list(chunk.metadata.get("ontology_proposals", []))
            proposal_writer = getattr(self._writer, "write_ontology_proposals", None)
            if proposal_payload and callable(proposal_writer):
                proposal_result = proposal_writer(proposal_payload, chunk)
                # Older minimal writer test doubles predate this optional
                # governance stage and return a plain MagicMock.  Production
                # GraphWriter returns an awaitable list of proposal ids.
                if isawaitable(proposal_result):
                    proposal_result = await proposal_result
                if isinstance(proposal_result, list):
                    ontology_proposals += len(proposal_result)

            await self._writer.write_entities(entities, chunk)
            await self._writer.write_relations(
                relations, entity_map, doc_id=doc.id, tenant=doc.tenant
            )
            alias_results = extracted.get("alias_results")
            if alias_results is not None:
                explicit_aliases += await self._writer.register_explicit_aliases(
                    alias_results[index], entities, chunk,
                )
            artifact_results = extracted.get("artifact_results")
            artifacts = artifact_results[index] if artifact_results is not None else []
            for artifact in artifacts:
                artifact.source_doc_id = doc.id
                artifact.tenant = doc.tenant
            if artifacts:
                await self._writer.write_intelligence_artifacts(artifacts, chunk)
            temporal_results = extracted.get("temporal_results")
            if temporal_results is not None and temporal_results[index]:
                await self._writer.write_temporal_periods(chunk, temporal_results[index])

            all_entities.extend(entities)
            all_relations.extend(relations)
            all_artifacts.extend(artifacts)

        maintenance_report = await self._writer.validate_and_check_cycles(
            doc_id=doc.id,
            tenant=doc.tenant,
            is_reingest=is_reingest,
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

        corpus_revision = await self._writer.complete_corpus_update(doc.tenant)
        # This checkpoint is used by both the bulk and RabbitMQ paths.  It is
        # deliberately written last: retries of interrupted messages must
        # re-run safely rather than skipping a half-written document.
        await self._writer.mark_document_ingest_complete(doc.id, tenant=doc.tenant)
        if manifest is not None:
            manifest.status = "completed"
            manifest.completed_at = datetime.now(timezone.utc)
            manifest.stage_metrics["write"] = {
                "items": len(chunks) + len(all_entities) + len(all_relations) + len(all_artifacts) + len(structured_tables),
                "artifacts": len(all_artifacts),
                "aliases": explicit_aliases,
                "ontology_proposals": ontology_proposals,
                "tables": len(structured_tables),
                "cost_usd": None,
            }
            manifest.integrity_hash = manifest.compute_integrity_hash()
            await self._writer.write_ingestion_manifest(manifest)

        log.info(
            "ingestion_agent.done",
            job_id=job_id,
            chunks=len(chunks),
            entities=len(all_entities),
            relations=len(all_relations),
            artifacts=len(all_artifacts),
            explicit_aliases=explicit_aliases,
            ontology_proposals=ontology_proposals,
            structured_tables=len(structured_tables),
            wikidata_links=wikidata_links,
            validation_issues=maintenance_report["validation"]["total_issues"],
            new_conflicts=maintenance_report["new_conflicts"],
            corpus_revision=corpus_revision,
        )
        return {
            "job_id": job_id,
            "doc_id": doc.id,
            "chunks": len(chunks),
            "entities": len(all_entities),
            "relations": len(all_relations),
            "artifacts": len(all_artifacts),
            "explicit_aliases": explicit_aliases,
            "ontology_proposals": ontology_proposals,
            "structured_tables": len(structured_tables),
            "wikidata_links": wikidata_links,
            "maintenance": maintenance_report,
            "corpus_revision": corpus_revision,
            "lineage_reviews": lineage_reviews,
            "obligation_reviews": obligation_reviews,
            "ingestion_manifest_id": manifest.id if manifest is not None else "",
        }
