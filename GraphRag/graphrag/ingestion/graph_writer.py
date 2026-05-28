"""Write extracted chunks, entities, and relations into Neo4j.

Enhanced with:
- Alias resolution before entity MERGE (prevents duplicate nodes)
- Embedding-based deduplication (catches aliases that escaped name resolution)
- Audit trail on every write
- Provenance (source_doc_id) on every relation
- Source type tagging (document vs inferred)
- Ingestion validation after each batch
- Cycle detection post-write
"""

from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk, Document, Entity, Relation
from graphrag.graph.alias_registry import get_alias_registry
from graphrag.graph.audit_trail import AuditTrail
from graphrag.graph.community_builder import CommunityBuilder
from graphrag.graph.community_manager import CommunityManager
from graphrag.graph.community_summarizer import CommunitySummarizer
from graphrag.graph.contradiction_detector import ContradictionDetector
from graphrag.graph.cycle_detector import CycleDetector
from graphrag.graph.ingestion_validator import IngestionValidator
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.graph.ontology_registry import get_ontology_registry
from graphrag.graph.quarantine import QuarantineService

log = structlog.get_logger(__name__)


class GraphWriter:
    def __init__(self, changed_by: str = "ingestion_worker"):
        self._neo4j                  = get_neo4j()
        self._audit                  = AuditTrail(self._neo4j)
        self._validator              = IngestionValidator(self._neo4j)
        self._cycle_detector         = CycleDetector(self._neo4j)
        self._quarantine             = QuarantineService(self._neo4j)
        self._contradiction          = ContradictionDetector(self._neo4j)
        self._ontology               = get_ontology_registry(self._neo4j)
        self._changed_by             = changed_by
        self._registry_loaded_tenants: set[str] = set()   # per-tenant load tracking
        self._ontology_loaded        = False
        self._cfg                    = get_settings()

    def _get_registry(self, tenant: str):
        """Return the tenant-scoped alias registry (cached pool)."""
        return get_alias_registry(self._neo4j, tenant=tenant)

    async def _ensure_registry(self, tenant: str = "default") -> None:
        """Load the tenant alias registry and ontology on first use per tenant."""
        registry = self._get_registry(tenant)
        if tenant not in self._registry_loaded_tenants:
            await registry.load()
            self._registry_loaded_tenants.add(tenant)
        if not self._ontology_loaded:
            await self._ontology.load(
                self._cfg.ingestion.get(
                    "entity_types",
                    ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"],
                )
            )
            self._ontology_loaded = True

    # ── Document ───────────────────────────────────────────────────────────────

    async def write_document(self, doc: Document) -> None:
        await self._neo4j.merge_document(
            doc_id=doc.id,
            filename=doc.filename,
            ingested_at=doc.ingested_at.isoformat(),
            authority_level=doc.authority_level,
            valid_from=doc.valid_from.isoformat() if doc.valid_from else None,
            valid_to=doc.valid_to.isoformat() if doc.valid_to else None,
            tenant=doc.tenant,
        )

        # Register supersession chains
        if doc.supersedes:
            from graphrag.graph.document_authority import DocumentAuthorityService
            svc = DocumentAuthorityService(self._neo4j)
            await svc.register_supersession(doc.id, doc.supersedes)

        await self._audit.log_document_change(
            doc_id=doc.id,
            operation="create",
            new_values={"filename": doc.filename, "authority_level": doc.authority_level},
            changed_by=self._changed_by,
        )
        log.info("graph_writer.document_merged", doc_id=doc.id)

    # ── Chunks ─────────────────────────────────────────────────────────────────

    async def write_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            await self._neo4j.merge_chunk(chunk, tenant=chunk.tenant)
        log.info("graph_writer.chunks_merged", count=len(chunks))

    # ── Entities ───────────────────────────────────────────────────────────────

    async def write_entities(
        self, entities: list[Entity], chunk: Chunk
    ) -> list[Entity]:
        """
        Write entities with alias resolution and embedding deduplication.

        Returns the list of entities as actually written (some may be
        redirected to canonical entities found via alias resolution).
        """
        tenant = chunk.tenant
        await self._ensure_registry(tenant)
        registry = self._get_registry(tenant)
        written: list[Entity] = []

        for entity in entities:
            # Propagate tenant onto the entity so merge_entity stores it correctly
            entity.tenant = tenant

            # 1. Alias resolution — name-based
            canonical = registry.resolve(entity.name)
            if canonical and canonical[0] != entity.name:
                log.info(
                    "graph_writer.alias_resolved",
                    raw=entity.name,
                    canonical=canonical[0],
                    type=canonical[1],
                    tenant=tenant,
                )
                # Register the new variant as an alias and use the canonical
                await registry.register_alias(
                    raw_value=entity.name,
                    canonical_name=canonical[0],
                    canonical_type=canonical[1],
                    confidence=0.9,
                )
                # Link the chunk to the canonical entity instead
                await self._neo4j.merge_mentions(
                    chunk.id, canonical[0], canonical[1], tenant=tenant
                )
                continue   # don't create a duplicate node

            # 2. Embedding deduplication — catches aliases that slipped through
            if entity.embedding:
                dup = await registry.find_duplicate_by_embedding(
                    embedding=entity.embedding,
                    entity_type=entity.type,
                    exclude_name=entity.name,
                )
                if dup:
                    dup_name, dup_type, similarity = dup
                    log.info(
                        "graph_writer.embedding_dedup",
                        raw=entity.name,
                        canonical=dup_name,
                        similarity=round(similarity, 4),
                        tenant=tenant,
                    )
                    await registry.register_alias(
                        raw_value=entity.name,
                        canonical_name=dup_name,
                        canonical_type=dup_type,
                        confidence=similarity,
                    )
                    await self._neo4j.merge_mentions(
                        chunk.id, dup_name, dup_type, tenant=tenant
                    )
                    continue

            # 3. Genuinely new entity — write it and record in audit trail
            is_new = await self._neo4j.entity_exists(
                entity.name, entity.type, tenant=tenant
            )
            await self._neo4j.merge_entity(entity, tenant=tenant)
            await self._neo4j.merge_mentions(
                chunk.id, entity.name, entity.type, tenant=tenant
            )

            # Register canonical name in alias registry in-memory cache
            registry._exact[
                __import__("re").sub(r"[^a-z0-9\s]", " ", entity.name.lower()).strip()
            ] = (entity.name, entity.type)

            await self._audit.log_entity_change(
                entity_name=entity.name,
                entity_type=entity.type,
                operation="create" if not is_new else "update",
                new_values={"description": entity.description, "type": entity.type},
                changed_by=self._changed_by,
                source_doc_id=chunk.document_id,
            )
            written.append(entity)

        log.info(
            "graph_writer.entities_merged",
            submitted=len(entities),
            written=len(written),
            chunk_id=chunk.id,
            tenant=tenant,
        )
        return written

    # ── Relations ──────────────────────────────────────────────────────────────

    async def write_relations(
        self,
        relations: list[Relation],
        entity_map: dict[str, Entity],
        doc_id: str = "",
        tenant: str = "default",
    ) -> None:
        await self._ensure_registry(tenant)
        registry = self._get_registry(tenant)
        merged_count = 0
        for rel in relations:
            src = entity_map.get(rel.source_entity_id)
            tgt = entity_map.get(rel.target_entity_id)
            if not (src and tgt):
                continue

            # Resolve aliases for src and tgt names within this tenant.
            # Alias resolution can change the canonical type too, so capture
            # both name and type after resolution so merge_relation can match
            # on the full (name, type, tenant) triple.
            src_canonical = registry.resolve(src.name)
            tgt_canonical = registry.resolve(tgt.name)
            src_name = src_canonical[0] if src_canonical else src.name
            src_type = src_canonical[1] if src_canonical else src.type
            tgt_name = tgt_canonical[0] if tgt_canonical else tgt.name
            tgt_type = tgt_canonical[1] if tgt_canonical else tgt.type

            is_valid, normalized_relation = self._ontology.validate_relation_triplet(
                src_type,
                rel.relation,
                tgt_type,
            )
            rel.relation = normalized_relation
            if not is_valid:
                await self._ontology.record_schema_event(
                    event_type="relation_schema_violation",
                    detail=f"{src_type}:{src_name}-{rel.relation}->{tgt_type}:{tgt_name}",
                    source_doc_id=doc_id,
                )
                log.warning(
                    "graph_writer.relation_skipped",
                    src=src_name,
                    src_type=src_type,
                    tgt=tgt_name,
                    tgt_type=tgt_type,
                    relation=rel.relation,
                    tenant=tenant,
                )
                continue

            # Inject provenance
            rel.source_doc_id = doc_id

            await self._neo4j.merge_relation(
                rel, src_name, src_type, tgt_name, tgt_type, tenant=tenant
            )
            merged_count += 1

            await self._audit.log_relation_change(
                src_name=src_name,
                tgt_name=tgt_name,
                relation=rel.relation,
                operation="upsert",
                new_values={
                    "confidence":      rel.confidence,
                    "constraint_type": rel.constraint_type,
                    "source_type":     rel.source_type,
                    "valid_from":      str(rel.valid_from) if rel.valid_from else None,
                    "valid_to":        str(rel.valid_to) if rel.valid_to else None,
                },
                changed_by=self._changed_by,
                source_doc_id=doc_id,
            )

        log.info("graph_writer.relations_merged", count=merged_count, tenant=tenant)

    # ── Post-write validation ──────────────────────────────────────────────────

    async def validate_and_check_cycles(
        self,
        doc_id: str,
        tenant: str = "default",
    ) -> dict:
        """
        Run validation, cycle detection, contradiction detection, and
        auto-quarantine of flagged anomalies after a full document write.
        Returns the combined report.
        """
        validation_report = await self._validator.validate(doc_id=doc_id)
        await self._validator.remove_self_loops()
        cycles = await self._cycle_detector.run()

        # Auto-quarantine entities flagged as degree anomalies
        quarantined = await self._quarantine.auto_quarantine_anomalies(
            doc_id=doc_id,
            validation_report=validation_report,
        )

        # Detect and persist new contradictions introduced by this document,
        # scoped to the tenant so cross-tenant edges are never compared.
        new_conflicts = await self._contradiction.scan(doc_id=doc_id, tenant=tenant)

        community_report = await self._maybe_rebuild_communities(tenant)

        return {
            "validation": validation_report,
            "cycles": len(cycles),
            "auto_quarantined": quarantined,
            "new_conflicts": len(new_conflicts),
            "community_rebuild": community_report,
        }

    async def _maybe_rebuild_communities(self, tenant: str) -> dict:
        graph_cfg = self._cfg.graph

        # Honour the staleness-check toggle — when disabled the check is fully
        # skipped (no Neo4j read, no snapshot write) so ingestion cost stays flat.
        if not graph_cfg.get("community_staleness_check_on_ingest", True):
            return {"checked": False, "rebuilt": False, "community_count": 0}

        manager = CommunityManager(self._neo4j)
        stale = await manager.check_staleness(tenant=tenant)
        report = {"checked": True, **stale, "rebuilt": False, "community_count": 0}

        if not graph_cfg.get("auto_rebuild_communities", True):
            if stale.get("snapshot_recorded_at") is None:
                await manager.snapshot(tenant=tenant)
            report["checked"] = False
            return report

        if not stale.get("should_rebuild"):
            return report

        builder = CommunityBuilder(tenant=tenant)
        communities = await builder.build()
        if communities:
            summarizer = CommunitySummarizer()
            communities = await summarizer.summarize_all(communities)
            for community in communities:
                await self._neo4j.merge_community(community)
        await manager.mark_rebuilt(tenant=tenant)

        report["rebuilt"] = True
        report["community_count"] = len(communities)
        return report
