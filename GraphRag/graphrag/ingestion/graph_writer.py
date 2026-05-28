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

from graphrag.core.models import Chunk, Document, Entity, Relation
from graphrag.graph.alias_registry import get_alias_registry
from graphrag.graph.audit_trail import AuditTrail
from graphrag.graph.cycle_detector import CycleDetector
from graphrag.graph.ingestion_validator import IngestionValidator
from graphrag.graph.neo4j_client import get_neo4j

log = structlog.get_logger(__name__)


class GraphWriter:
    def __init__(self, changed_by: str = "ingestion_worker"):
        self._neo4j          = get_neo4j()
        self._alias_registry = get_alias_registry(self._neo4j)
        self._audit          = AuditTrail(self._neo4j)
        self._validator      = IngestionValidator(self._neo4j)
        self._cycle_detector = CycleDetector(self._neo4j)
        self._changed_by     = changed_by
        self._registry_loaded = False

    async def _ensure_registry(self) -> None:
        if not self._registry_loaded:
            await self._alias_registry.load()
            self._registry_loaded = True

    # ── Document ───────────────────────────────────────────────────────────────

    async def write_document(self, doc: Document) -> None:
        await self._neo4j.merge_document(
            doc_id=doc.id,
            filename=doc.filename,
            ingested_at=doc.ingested_at.isoformat(),
            authority_level=doc.authority_level,
            valid_from=doc.valid_from.isoformat() if doc.valid_from else None,
            valid_to=doc.valid_to.isoformat() if doc.valid_to else None,
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
            await self._neo4j.merge_chunk(chunk)
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
        await self._ensure_registry()
        written: list[Entity] = []

        for entity in entities:
            # 1. Alias resolution — name-based
            canonical = self._alias_registry.resolve(entity.name)
            if canonical and canonical[0] != entity.name:
                log.info(
                    "graph_writer.alias_resolved",
                    raw=entity.name,
                    canonical=canonical[0],
                    type=canonical[1],
                )
                # Register the new variant as an alias and use the canonical
                await self._alias_registry.register_alias(
                    raw_value=entity.name,
                    canonical_name=canonical[0],
                    canonical_type=canonical[1],
                    confidence=0.9,
                )
                # Link the chunk to the canonical entity instead
                await self._neo4j.merge_mentions(chunk.id, canonical[0], canonical[1])
                continue   # don't create a duplicate node

            # 2. Embedding deduplication — catches aliases that slipped through
            if entity.embedding:
                dup = await self._alias_registry.find_duplicate_by_embedding(
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
                    )
                    await self._alias_registry.register_alias(
                        raw_value=entity.name,
                        canonical_name=dup_name,
                        canonical_type=dup_type,
                        confidence=similarity,
                    )
                    await self._neo4j.merge_mentions(chunk.id, dup_name, dup_type)
                    continue

            # 3. Genuinely new entity — write it and record in audit trail
            is_new = await self._neo4j.entity_exists(entity.name, entity.type)
            await self._neo4j.merge_entity(entity)
            await self._neo4j.merge_mentions(chunk.id, entity.name, entity.type)

            # Register canonical name in alias registry
            self._alias_registry._exact[
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
        )
        return written

    # ── Relations ──────────────────────────────────────────────────────────────

    async def write_relations(
        self,
        relations: list[Relation],
        entity_map: dict[str, Entity],
        doc_id: str = "",
    ) -> None:
        for rel in relations:
            src = entity_map.get(rel.source_entity_id)
            tgt = entity_map.get(rel.target_entity_id)
            if not (src and tgt):
                continue

            # Resolve aliases for src and tgt names
            src_canonical = self._alias_registry.resolve(src.name)
            tgt_canonical = self._alias_registry.resolve(tgt.name)
            src_name = src_canonical[0] if src_canonical else src.name
            tgt_name = tgt_canonical[0] if tgt_canonical else tgt.name

            # Inject provenance
            rel.source_doc_id = doc_id

            await self._neo4j.merge_relation(rel, src_name, tgt_name)

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

        log.info("graph_writer.relations_merged", count=len(relations))

    # ── Post-write validation ──────────────────────────────────────────────────

    async def validate_and_check_cycles(self, doc_id: str) -> dict:
        """
        Run validation and cycle detection after a full document write.
        Returns the combined report.
        """
        validation_report = await self._validator.validate(doc_id=doc_id)
        await self._validator.remove_self_loops()
        cycles = await self._cycle_detector.run()

        return {
            "validation": validation_report,
            "cycles": len(cycles),
        }
