"""Publication gate: EXTRACTED -> STAGED -> VALIDATED -> PUBLISHED.

The batch is validated in memory *before* any graph write (plan D1). BLOCKING
records are removed from the batch and written to the durable quarantine; a
BLOCKING document-level failure rejects the whole batch and nothing is written.
WARNING / INFORMATIONAL findings are published and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from graphrag.core.config import get_settings
from graphrag.core.models import Chunk, Document, Entity, IngestionRunManifest, Relation
from graphrag.graph.validation import metrics
from graphrag.graph.validation.batch import (
    RejectedRecord,
    validate_batch,
    validate_document,
)
from graphrag.graph.validation.graph_checks import check_supersedes, run_read_only
from graphrag.graph.validation.quarantine_store import QuarantineRecordStore
from graphrag.graph.validation.report import ValidationReport
from graphrag.graph.validation.rules import PublicationState, Severity

log = structlog.get_logger(__name__)


class PublicationRejected(ValueError):
    """A document-level BLOCKING rule failed; nothing was written."""

    def __init__(self, report: ValidationReport):
        self.report = report
        rules = sorted({v.rule_id for v in report.blocking})
        super().__init__(f"publication rejected by validation rules: {', '.join(rules)}")


@dataclass
class StagedBatch:
    extraction_results: list[tuple[list[Entity], list[Relation]]]
    rejected: list[RejectedRecord] = field(default_factory=list)
    report: ValidationReport | None = None


def set_state(manifest: IngestionRunManifest | None, state: PublicationState,
              report: ValidationReport | None = None, quarantined: int | None = None) -> None:
    if manifest is None:
        return
    entry = dict(manifest.stage_metrics.get("publication", {}))
    entry["state"] = state.value
    if report is not None:
        by_sev = report.counts()["by_severity"]
        entry["blocking"] = by_sev.get(Severity.BLOCKING.value, 0)
        entry["warnings"] = by_sev.get(Severity.WARNING.value, 0)
        entry["informational"] = by_sev.get(Severity.INFORMATIONAL.value, 0)
        entry["records_checked"] = report.records_checked
    if quarantined is not None:
        entry["quarantined"] = quarantined
    manifest.stage_metrics["publication"] = entry


def ontology_hooks(registry) -> tuple[set[str] | None, object | None]:
    """Allowed entity types and triplet validator, only when a registry is loaded."""
    if registry is None or not getattr(registry, "is_loaded", False):
        return None, None
    allowed = getattr(registry, "allowed_types", None)
    return (set(allowed) if allowed else None), getattr(registry, "validate_relation_triplet", None)


def _default_property_checker():
    try:
        from graphrag.graph.property_schema import PropertySchemaValidator
        return PropertySchemaValidator(None)._check_props
    except Exception:  # noqa: BLE001 - property rules are advisory (WARNING only)
        return None


class PublicationGate:
    def __init__(self, neo4j_client, *, store: QuarantineRecordStore | None = None,
                 property_checker=None, semantic_validator=None, source: str = "document_ingestion"):
        self._neo4j = neo4j_client
        self._store = store or QuarantineRecordStore(neo4j_client)
        self._property_checker = property_checker
        self._semantic_validator = semantic_validator
        self.source = source

    @classmethod
    def from_settings(cls, neo4j_client, **kwargs) -> "PublicationGate | None":
        if not get_settings().ingestion.get("publication_gate_enabled", True):
            return None
        kwargs.setdefault("property_checker", _default_property_checker())
        return cls(neo4j_client, **kwargs)

    @property
    def store(self) -> QuarantineRecordStore:
        return self._store

    def _validate(self, entities, relations, *, tenant, chunk_id, registry):
        allowed, triplet = ontology_hooks(registry)
        return validate_batch(
            entities, relations, tenant=tenant, source=self.source, chunk_id=chunk_id,
            allowed_entity_types=allowed, triplet_validator=triplet,
            property_checker=self._property_checker, semantic_validator=self._semantic_validator,
        )

    async def stage(
        self,
        doc: Document,
        chunks: list[Chunk],
        extraction_results: list[tuple[list[Entity], list[Relation]]],
        manifest: IngestionRunManifest | None = None,
        *,
        registry=None,
    ) -> StagedBatch:
        """Validate the whole batch in memory. Raises PublicationRejected
        (after quarantining the document) when a document-level rule blocks."""
        set_state(manifest, PublicationState.STAGED)
        report = validate_document(doc, chunks, source=self.source)
        report.extend(await check_supersedes(
            self._neo4j, tenant=doc.tenant, document_key=doc.filename or doc.id,
            supersedes=list(doc.supersedes or []), source=self.source,
        ))
        if not report.conforms:
            rejected = [RejectedRecord(
                record_kind="document", record_key=doc.filename or doc.id,
                rule_ids=sorted({v.rule_id for v in report.blocking}),
                messages=[v.message for v in report.blocking if v.message],
                payload=doc.model_dump(mode="json", include={
                    "id", "filename", "source_path", "tenant", "content_hash",
                    "valid_from", "valid_to", "supersedes", "authority_level"}),
            )]
            await self.quarantine(rejected, doc=doc, document_id="", manifest=manifest)
            set_state(manifest, PublicationState.REJECTED, report, quarantined=1)
            metrics.record_report(report)
            metrics.record_batch("rejected")
            log.warning("publication_gate.document_rejected", tenant=doc.tenant,
                        filename=doc.filename, rules=[v.rule_id for v in report.blocking])
            raise PublicationRejected(report)

        filtered: list[tuple[list[Entity], list[Relation]]] = []
        rejected: list[RejectedRecord] = []
        for chunk, (entities, relations) in zip(chunks, extraction_results):
            result = self._validate(entities, relations, tenant=doc.tenant,
                                    chunk_id=chunk.id, registry=registry)
            filtered.append((result.entities, result.relations))
            rejected.extend(result.rejected)
            report.extend(result.report)
        set_state(manifest, PublicationState.VALIDATED, report, quarantined=len(rejected))
        metrics.record_report(report)
        return StagedBatch(filtered, rejected, report)

    async def quarantine(self, records: list[RejectedRecord], *, doc: Document, document_id: str,
                         manifest: IngestionRunManifest | None = None) -> list[str]:
        if not records:
            return []
        ids = await self._store.save(
            records, tenant=doc.tenant, source=self.source, document_id=document_id,
            document_key=doc.filename or doc.id, manifest_id=manifest.id if manifest else "",
        )
        for kind in {r.record_kind for r in records}:
            metrics.record_quarantined(kind, sum(1 for r in records if r.record_kind == kind))
        log.info("publication_gate.quarantined", tenant=doc.tenant, filename=doc.filename,
                 count=len(records), rules=sorted({rid for r in records for rid in r.rule_ids}))
        return ids

    @staticmethod
    def published(manifest: IngestionRunManifest | None, quarantined: int) -> None:
        set_state(manifest, PublicationState.PUBLISHED, quarantined=quarantined)
        metrics.record_batch("published_with_quarantine" if quarantined else "published")

    async def retry(self, *, tenant: str, record_id: str, writer, corrected: dict | None = None,
                    registry=None) -> dict:
        """Re-validate a (corrected) quarantined record and publish it if it now passes.

        Publication goes through the normal GraphWriter path, so alias
        resolution and the post-resolution domain/range check still apply.
        """
        from graphrag.graph.corpus_revision import CorpusMutation

        rec = await self._store.get(tenant=tenant, record_id=record_id)
        if rec is None:
            raise LookupError(record_id)
        if rec["record_kind"] == "document":
            raise ValueError("a rejected document is retried by re-ingesting the corrected source")
        if rec["status"] != "QUARANTINED":
            raise ValueError(f"record is {rec['status']}, not QUARANTINED")
        payload = dict(corrected if corrected is not None else rec["payload"])

        if rec["record_kind"] == "entity":
            entity = Entity.model_validate(payload)
            result = self._validate([entity], [], tenant=tenant, chunk_id=rec["chunk_id"], registry=registry)
        else:
            relation = Relation.model_validate(payload)
            endpoints = []
            for side, eid in (("source_endpoint", relation.source_entity_id),
                              ("target_endpoint", relation.target_entity_id)):
                ep = payload.get(side) or {}
                if ep.get("name") and ep.get("type"):
                    endpoints.append(Entity(id=eid, name=ep["name"], type=ep["type"], tenant=tenant))
            result = self._validate(endpoints, [relation], tenant=tenant, chunk_id=rec["chunk_id"],
                                    registry=registry)
            if not result.rejected:
                missing = await self._missing_endpoints(tenant, endpoints)
                if missing:
                    result.rejected.append(RejectedRecord(
                        "relation", rec["record_key"], ["REL-REF-001"],
                        [f"endpoint not in published graph: {m}" for m in missing], payload))

        blocking = result.rejected
        if blocking:
            rule_ids = sorted({rid for r in blocking for rid in r.rule_ids})
            await self._store.mark_retry(tenant=tenant, record_id=record_id, resolved=False,
                                         rule_ids=rule_ids,
                                         messages=[m for r in blocking for m in r.messages],
                                         payload=payload)
            metrics.record_retry("still_invalid")
            return {"id": record_id, "status": "QUARANTINED", "rule_ids": rule_ids}

        async with CorpusMutation(self._neo4j, tenant, "quarantine_retry"):
            if rec["record_kind"] == "entity":
                chunk = Chunk(id=rec["chunk_id"] or record_id, document_id=rec["document_id"] or "",
                              text="", chunk_index=0, tenant=tenant)
                await writer.write_entities(result.entities, chunk)
                late: list[RejectedRecord] = []
            else:
                await writer.write_relations(result.relations, {e.id: e for e in endpoints},
                                             doc_id=rec["document_id"] or "", tenant=tenant)
                drain = getattr(writer, "drain_rejections", None)
                late = drain() if callable(drain) else []
        if late:
            rule_ids = sorted({rid for r in late for rid in r.rule_ids})
            await self._store.mark_retry(tenant=tenant, record_id=record_id, resolved=False,
                                         rule_ids=rule_ids, messages=[m for r in late for m in r.messages],
                                         payload=payload)
            metrics.record_retry("still_invalid")
            return {"id": record_id, "status": "QUARANTINED", "rule_ids": rule_ids}
        await self._store.mark_retry(tenant=tenant, record_id=record_id, resolved=True,
                                     rule_ids=[], messages=[], payload=payload)
        metrics.record_retry("published")
        return {"id": record_id, "status": "RESOLVED", "rule_ids": []}

    async def _missing_endpoints(self, tenant: str, endpoints: list[Entity]) -> list[str]:
        rows = await run_read_only(
            self._neo4j,
            """
            UNWIND $keys AS k
            OPTIONAL MATCH (e:Entity {name: k.name, type: k.type, tenant: $tenant})
            RETURN k.name AS name, k.type AS type, e IS NOT NULL AS present
            """,
            keys=[{"name": e.name, "type": e.type} for e in endpoints], tenant=tenant,
        )
        return [f"{r['type']}:{r['name']}" for r in rows if not r["present"]]
