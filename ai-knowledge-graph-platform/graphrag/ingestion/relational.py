"""Local relational-to-KG ingestion with declarative, tenant-scoped mappings.

This module deliberately has no cloud or vendor dependency.  SQLite is used as
the local reference connector; the mapping and validation contracts are
provider-neutral and can later be backed by another database adapter.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, model_validator

from graphrag.core.models import Chunk, Document, Entity, Relation, SourceType
from graphrag.graph.source_catalog import (
    SourceCatalogRepository,
    SourceEnvelope,
    SourceKind,
    SourceMapping,
    SourceSystem,
)
from graphrag.ingestion.incremental import (
    compute_relational_snapshot_hash,
    should_skip_ingest,
)
from graphrag.ingestion.row_checkpoint import compute_row_hash, diff_rows

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a simple SQL identifier")
    return value


class EntityTableMapping(BaseModel):
    table: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    id_column: str = Field(min_length=1)
    name_column: str = Field(min_length=1)
    description_column: str | None = None
    attributes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_identifiers(self) -> "EntityTableMapping":
        for key in ("table", "id_column", "name_column", "description_column"):
            value = getattr(self, key)
            if value:
                _identifier(value, key)
        for value in self.attributes:
            _identifier(value, "attributes")
        return self


class RelationTableMapping(BaseModel):
    table: str = Field(min_length=1)
    source_table: str = Field(min_length=1)
    target_table: str = Field(min_length=1)
    source_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    relation: str = Field(min_length=2, pattern=r"^[A-Z][A-Z0-9_]{1,49}$")
    confidence_column: str | None = None
    valid_from_column: str | None = None
    valid_to_column: str | None = None

    @model_validator(mode="after")
    def validate_identifiers(self) -> "RelationTableMapping":
        for key in (
            "table", "source_table", "target_table", "source_column", "target_column", "confidence_column",
            "valid_from_column", "valid_to_column",
        ):
            value = getattr(self, key)
            if value:
                _identifier(value, key)
        return self


class RelationalGraphMapping(BaseModel):
    """Declarative mapping from tables to KG entities and relations."""

    id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    entities: list[EntityTableMapping] = Field(min_length=1)
    relations: list[RelationTableMapping] = Field(default_factory=list)
    ontology_version: str = "local/v1"

    def as_source_mapping(self) -> SourceMapping:
        return SourceMapping(
            id=f"{self.id}-{self.version}",
            tenant=self.tenant,
            source_id=self.source_id,
            version=self.version,
            mapping=self.model_dump(mode="json", exclude={"tenant", "source_id"}),
        )


class MappingValidationReport(BaseModel):
    valid: bool
    tenant: str
    source_id: str
    entity_rows: int = 0
    relation_rows: int = 0
    errors: list[str] = Field(default_factory=list)
    shacl_conforms: bool | None = None
    # Set by ingest()'s content-hash checkpoint (graphrag/ingestion/incremental.py).
    # content_hash is always populated once rows are read; skipped is True only
    # when it matched the caller-supplied previous_hash and no write occurred.
    content_hash: str = ""
    skipped: bool = False


class IncrementalIngestReport(MappingValidationReport):
    """`MappingValidationReport` plus the row-level diff
    `RelationalGraphIngestor.ingest_incremental()` computed against the
    last completed checkpoint. `upserted`/`deleted` row keys are
    `"entity:{table}:{key}"`/`"relation:{table}:{source_key}:{target_key}"`
    -- see `graphrag/ingestion/row_checkpoint.py`."""

    upserted: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    unchanged: list[str] = Field(default_factory=list)


class ConcurrentIngestError(RuntimeError):
    """Raised when another ingest run already holds the lease for this
    (tenant, source_id) -- see `Neo4jClient.begin_relational_ingest_run`."""


@dataclass
class _BuildResult:
    """What reading and building a mapping's entities/relations produces --
    shared by `ingest()` and `ingest_incremental()` so both read/build
    identically; only what each does with the result differs."""

    entities: list[Entity]
    relations: list[Relation]
    payload: list[dict[str, Any]]
    relation_payload: list[dict[str, Any]]
    by_source_key: dict[tuple[str, str], Entity]
    # "{entity|relation}:{table}:{key...}" -> per-row content hash, for
    # ingest_incremental()'s row-level diff (graphrag/ingestion/row_checkpoint.py).
    row_hashes: dict[str, str] = field(default_factory=dict)


class SQLiteSourceConnector:
    """Read-only local SQLite connector implementing the source contract."""

    kind = SourceKind.DATABASE

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def uri(self) -> str:
        return str(self.path.resolve())

    async def records(
        self, source: SourceSystem, mapping: SourceMapping, *, cursor: str = ""
    ):
        spec = mapping.mapping
        tables = list(spec.get("entities", [])) + list(spec.get("relations", []))
        for table_spec in tables:
            table = _identifier(str(table_spec["table"]), "table")
            rows = await self.read_table(table)
            for index, row in enumerate(rows):
                external_id = f"{table}:{row.get('id', index)}"
                yield SourceEnvelope(
                    external_id=external_id,
                    content=json.dumps(row, sort_keys=True, default=str),
                    content_type="application/json",
                    metadata={"table": table, "source_id": source.id},
                    cursor=str(index + 1),
                )

    def _read_table(self, table: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        # `with sqlite3.connect(...) as conn:` commits/rolls back on exit --
        # it does NOT close the connection (a well-known sqlite3 gotcha), so
        # every call here used to leak an open connection/file handle until
        # garbage collection got around to it. Harmless on Linux (a file can
        # be deleted while a handle is still open); on Windows it means a
        # caller that deletes this file right after reading it (an ephemeral
        # fixture DB, say) can hit a real PermissionError -- confirmed live
        # while wiring exactly that pattern for the Energy demo.
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
        finally:
            conn.close()

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_table, _identifier(table, "table"))


class PostgreSQLSourceConnector:
    """Read-only PostgreSQL adapter for the same local mapping contract.

    It works with a local Docker PostgreSQL instance through a SQLAlchemy
    async URL. Table names are identifier-validated; all values remain data,
    never executable SQL supplied by a mapping or an agent.
    """

    kind = SourceKind.DATABASE

    def __init__(self, url: str):
        if not url.startswith(("postgresql+asyncpg://", "postgresql://")):
            raise ValueError("PostgreSQL URL must use postgresql+asyncpg:// or postgresql://")
        self.url = url
        self._engine = None

    @property
    def uri(self) -> str:
        """Safe source identity; credentials never enter a persisted KGSource."""
        from sqlalchemy.engine import make_url

        return str(make_url(self.url).render_as_string(hide_password=True))

    def _get_engine(self):
        if self._engine is None:
            from sqlalchemy.ext.asyncio import create_async_engine
            self._engine = create_async_engine(self.url, pool_pre_ping=True)
        return self._engine

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        from sqlalchemy import text

        safe_table = _identifier(table, "table")
        async with self._get_engine().connect() as connection:
            result = await connection.execute(text(f'SELECT * FROM "{safe_table}"'))
            return [dict(row) for row in result.mappings().all()]

    async def records(
        self, source: SourceSystem, mapping: SourceMapping, *, cursor: str = ""
    ):
        spec = mapping.mapping
        tables = list(spec.get("entities", [])) + list(spec.get("relations", []))
        for table_spec in tables:
            table = _identifier(str(table_spec["table"]), "table")
            for index, row in enumerate(await self.read_table(table)):
                yield SourceEnvelope(
                    external_id=f"{table}:{row.get('id', index)}",
                    content=json.dumps(row, sort_keys=True, default=str),
                    content_type="application/json",
                    metadata={"table": table, "source_id": source.id},
                    cursor=str(index + 1),
                )

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


class ExcelWorkbookConnector:
    """Read an Excel workbook through the same mapping contract as databases.

    A worksheet is treated as a table and its first row as identifier-safe
    column names.  This keeps spreadsheet ingestion declarative, SHACL-gated,
    and provenance-preserving instead of introducing a separate ad-hoc path.
    """

    kind = SourceKind.FILE

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def uri(self) -> str:
        return self.path.resolve().as_uri()

    def _read_table(self, sheet_name: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        from openpyxl import load_workbook

        workbook = load_workbook(self.path, read_only=True, data_only=True)
        if sheet_name not in workbook.sheetnames:
            raise ValueError(f"worksheet not found: {sheet_name}")
        rows = workbook[sheet_name].iter_rows(values_only=True)
        headers = next(rows, None)
        if not headers:
            return []
        names = [str(header or "").strip() for header in headers]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError(f"{sheet_name}: headers must be non-empty and unique")
        for name in names:
            _identifier(name, f"{sheet_name} header")
        return [
            dict(zip(names, row, strict=True))
            for row in rows
            if any(value is not None and value != "" for value in row)
        ]

    async def read_table(self, table: str) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_table, table)

    async def records(
        self, source: SourceSystem, mapping: SourceMapping, *, cursor: str = ""
    ):
        spec = mapping.mapping
        sheets = list(spec.get("entities", [])) + list(spec.get("relations", []))
        for sheet_spec in sheets:
            sheet = str(sheet_spec["table"])
            for index, row in enumerate(await self.read_table(sheet)):
                yield SourceEnvelope(
                    external_id=f"{sheet}:{row.get('id', index)}",
                    content=json.dumps(row, sort_keys=True, default=str),
                    content_type="application/json",
                    metadata={"worksheet": sheet, "source_id": source.id},
                    cursor=str(index + 1),
                )


@runtime_checkable
class TabularSourceConnector(Protocol):
    """Connector boundary for ``RelationalGraphIngestor`` specifically.

    Distinct from ``graphrag.graph.source_catalog.SourceConnector``, which
    declares ``records()`` for streaming change envelopes -- a different job.
    ``RelationalGraphIngestor`` never calls ``records()``; it only ever calls
    ``read_table()``, and reads ``kind``/``uri`` (see its ``ingest()``/
    ``validate()`` methods below). Requiring ``records()`` here would
    over-constrain any future tabular connector that has no notion of
    streaming deltas -- a fixed-schema on-disk export, say.

    ``SQLiteSourceConnector``, ``PostgreSQLSourceConnector`` and
    ``ExcelWorkbookConnector`` all already satisfy this structurally; no
    change was needed on any of them.
    """

    kind: SourceKind

    @property
    def uri(self) -> str: ...

    async def read_table(self, table: str) -> list[dict[str, Any]]: ...


class RelationalGraphIngestor:
    """Validate and persist mapped relational rows through ``GraphWriter``."""

    def __init__(self, connector: TabularSourceConnector, graph_writer):
        self.connector = connector
        self.graph_writer = graph_writer

    async def validate(self, mapping: RelationalGraphMapping) -> MappingValidationReport:
        errors: list[str] = []
        entity_keys: set[tuple[str, str]] = set()
        entity_rows = 0
        relation_rows = 0

        for table_map in mapping.entities:
            rows = await self.connector.read_table(table_map.table)
            entity_rows += len(rows)
            for row in rows:
                key = row.get(table_map.id_column)
                name = row.get(table_map.name_column)
                if key in (None, "") or name in (None, ""):
                    errors.append(
                        f"{table_map.table}: every row needs {table_map.id_column} and {table_map.name_column}"
                    )
                else:
                    entity_keys.add((table_map.table, str(key)))

        for table_map in mapping.relations:
            rows = await self.connector.read_table(table_map.table)
            relation_rows += len(rows)
            for row in rows:
                if row.get(table_map.source_column) in (None, ""):
                    errors.append(f"{table_map.table}: missing {table_map.source_column}")
                if row.get(table_map.target_column) in (None, ""):
                    errors.append(f"{table_map.table}: missing {table_map.target_column}")

        return MappingValidationReport(
            valid=not errors,
            tenant=mapping.tenant,
            source_id=mapping.source_id,
            entity_rows=entity_rows,
            relation_rows=relation_rows,
            errors=sorted(set(errors)),
        )

    async def _read_and_build(self, mapping: RelationalGraphMapping) -> _BuildResult:
        """Read every entity/relation table and build `Entity`/`Relation`
        model objects, payload rows (for the whole-snapshot hash), and a
        per-row hash checkpoint map (for `ingest_incremental()`'s row-level
        diff). Shared by `ingest()` and `ingest_incremental()` so both
        read/build identically -- extracted from what used to be inline in
        `ingest()` alone, no behavior change for it."""
        entities: list[Entity] = []
        by_source_key: dict[tuple[str, str], Entity] = {}
        payload: list[dict[str, Any]] = []
        row_hashes: dict[str, str] = {}
        for table_map in mapping.entities:
            rows = await self.connector.read_table(table_map.table)
            for row in rows:
                source_key = str(row[table_map.id_column])
                entity_id = str(uuid5(NAMESPACE_URL, f"{mapping.source_id}:{table_map.table}:{source_key}"))
                entity = Entity(
                    id=entity_id,
                    name=str(row[table_map.name_column]),
                    type=table_map.entity_type,
                    description=str(row.get(table_map.description_column) or "")
                    if table_map.description_column else "",
                    tenant=mapping.tenant,
                    source_type=SourceType.DOCUMENT,
                    source_doc_id=f"relational:{mapping.source_id}",
                )
                entities.append(entity)
                by_source_key[(table_map.table, source_key)] = entity
                payload.append({"table": table_map.table, "row": row})
                row_hashes[f"entity:{table_map.table}:{source_key}"] = compute_row_hash(row)

        relations: list[Relation] = []
        relation_payload: list[dict[str, Any]] = []
        for table_map in mapping.relations:
            for row in await self.connector.read_table(table_map.table):
                relation_payload.append({"table": table_map.table, "row": row})
                source_key = str(row[table_map.source_column])
                target_key = str(row[table_map.target_column])
                source = by_source_key.get((table_map.source_table, source_key))
                target = by_source_key.get((table_map.target_table, target_key))
                if source is None or target is None:
                    raise ValueError(f"{table_map.table}: relation references an unknown entity")
                relation_id = str(uuid5(NAMESPACE_URL, f"{mapping.source_id}:{table_map.table}:{source.id}:{target.id}"))
                relations.append(Relation(
                    id=relation_id,
                    source_entity_id=source.id,
                    target_entity_id=target.id,
                    relation=table_map.relation,
                    confidence=float(row.get(table_map.confidence_column, 1.0) or 1.0)
                    if table_map.confidence_column else 1.0,
                    valid_from=self._timestamp(row.get(table_map.valid_from_column)) if table_map.valid_from_column else None,
                    valid_to=self._timestamp(row.get(table_map.valid_to_column)) if table_map.valid_to_column else None,
                    source_doc_id=f"relational:{mapping.source_id}",
                ))
                row_hashes[f"relation:{table_map.table}:{source_key}:{target_key}"] = compute_row_hash(row)

        return _BuildResult(
            entities=entities, relations=relations, payload=payload,
            relation_payload=relation_payload, by_source_key=by_source_key,
            row_hashes=row_hashes,
        )

    async def ingest(
        self,
        mapping: RelationalGraphMapping,
        previous_hash: str | None = None,
    ) -> MappingValidationReport:
        """Validate, then write, `mapping`'s source through to the graph.

        `previous_hash` is injected rather than fetched internally (e.g. via
        a Neo4j lookup) so this method stays testable without a live graph —
        matching this class's existing style, which already takes a
        `connector`/`graph_writer` pair rather than reaching for either
        itself. A real caller fetches it the same way
        `scripts/ingest_corpus.py` already fetches a document's stored hash
        (see `graphrag/ingestion/incremental.py`'s module docstring) and
        passes it in. When the freshly-read snapshot's hash matches, the
        write is skipped entirely — no SHACL check, no graph write — and the
        returned report has `skipped=True`.

        Whole-snapshot, all-or-nothing, no durable checkpoint of its own —
        see `ingest_incremental()` for row-level diffing, a durable
        checkpoint, crash recovery, and concurrent-run protection.
        """
        report = await self.validate(mapping)
        if not report.valid:
            raise ValueError("relational mapping rejected: " + "; ".join(report.errors))

        built = await self._read_and_build(mapping)
        entities, relations = built.entities, built.relations
        payload, relation_payload = built.payload, built.relation_payload

        # Snapshot hash covers entity AND relation rows -- a source where only
        # a relation table changed (e.g. supplies) must still be detected,
        # not just an entity-table change. Computed here, after both tables
        # are read but before any write, so an unchanged source costs one
        # read pass and a hash, never a SHACL check or a graph write.
        current_hash = compute_relational_snapshot_hash(payload + relation_payload)
        report.content_hash = current_hash
        if should_skip_ingest(previous_hash, current_hash):
            report.skipped = True
            return report

        from graphrag.graph.shacl_validator import SHACLValidator

        conforms, shacl_report = SHACLValidator.validate_relational_batch(
            entities, relations, tenant=mapping.tenant,
        )
        report.shacl_conforms = conforms
        if not conforms:
            raise ValueError("relational mapping rejected by SHACL: " + shacl_report)

        # A Document.source_id is a real foreign-key-like graph contract: make
        # the source and immutable mapping version durable before the document
        # write so `INGESTED_FROM` can be formed atomically by merge_document.
        catalog = SourceCatalogRepository(self.graph_writer.neo4j_client)
        await catalog.upsert_source(SourceSystem(
            id=mapping.source_id,
            tenant=mapping.tenant,
            name=mapping.source_id,
            kind=self.connector.kind,
            uri=self.connector.uri,
            owner="relational-ingestion",
            classification="synthetic" if mapping.tenant == "sustainability" else "internal",
        ))
        await catalog.add_mapping(mapping.as_source_mapping())

        raw = json.dumps(payload, sort_keys=True, default=str)
        # Keyed by source_id alone, not mapping.version: merge_document MERGEs
        # on (tenant, filename) (graphrag/graph/neo4j_client.py), so a stable
        # identity here is what lets a re-ingest of the same source under a
        # newer mapping version update the existing Document in place instead
        # of creating a parallel one every version bump -- mapping_version is
        # still recorded in metadata below, just no longer part of identity.
        document = Document(
            id=str(uuid5(NAMESPACE_URL, f"relational-document:{mapping.source_id}")),
            filename=f"relational://{mapping.source_id}",
            source_path=self.connector.uri,
            raw_text=raw,
            content_hash=current_hash,
            tenant=mapping.tenant,
            source_id=mapping.source_id,
            status="done",
            metadata={"mapping_id": mapping.id, "mapping_version": mapping.version,
                      "ontology_version": mapping.ontology_version,
                      "provenance": "local-relational-source"},
        )
        document_id = await self.graph_writer.write_document(document)
        chunk = Chunk(
            id=str(uuid5(NAMESPACE_URL, f"relational-chunk:{document_id}")),
            document_id=document_id,
            text=raw,
            chunk_index=0,
            tenant=mapping.tenant,
            metadata={"source_id": mapping.source_id, "mapping_version": mapping.version},
        )
        await self.graph_writer.write_chunks([chunk])
        written = await self.graph_writer.write_entities(entities, chunk)
        entity_map = {entity.id: entity for entity in written}
        await self.graph_writer.write_relations(relations, entity_map, doc_id=document_id, tenant=mapping.tenant)
        return report

    async def ingest_incremental(
        self,
        mapping: RelationalGraphMapping,
        *,
        run_id: str,
        lease_seconds: int = 300,
    ) -> IncrementalIngestReport:
        """Durable, row-level incremental ingest.

        Acquires a lease for (tenant, source_id) before doing anything else
        — concurrent-run protection: a second call for the same source
        while a lease is held raises `ConcurrentIngestError` immediately,
        no partial read or write. Then diffs this run's rows against the
        last *completed* checkpoint (row-level upsert/delete detection — an
        incomplete checkpoint from a crashed run is never trusted, same
        reasoning `Neo4jClient.get_document_states()` already documents for
        the document path), writes the current full row set (MERGE-
        idempotent, so this is correct on any change, not just an unchanged
        one — see the module docstring's "Out of scope" note on why this
        isn't a selective/minimal write), tombstones entities whose row
        disappeared, and only then persists the new checkpoint and releases
        the lease.

        Crash recovery and replay: a crash anywhere after the lease is
        acquired leaves it claimed but the checkpoint's `ingest_complete`
        unchanged — deliberately not released on error (fail-safe, not
        fail-open: an immediate retry racing the same broken state is worse
        than a bounded wait). The next call's lease acquisition succeeds
        once `lease_seconds` elapses and safely replays the whole sequence
        from scratch; this works *because* every write and tombstone here
        is idempotent, not because of any special-cased resume logic.

        Known limitation, stated plainly: a deleted *relation*-table row
        (a join disappearing while both entities remain) is reported in
        `deleted` for visibility but is not soft-deleted at the graph level
        in this pass — only entity deletions are tombstoned. Extending
        tombstoning to relationship edges is a real, separate follow-up.
        """
        neo4j = self.graph_writer.neo4j_client
        acquired = await neo4j.begin_relational_ingest_run(
            mapping.tenant, mapping.source_id, run_id, lease_seconds,
        )
        if not acquired:
            raise ConcurrentIngestError(
                f"{mapping.source_id!r}: another ingest run already holds the lease"
            )

        report = await self.validate(mapping)
        if not report.valid:
            raise ValueError("relational mapping rejected: " + "; ".join(report.errors))

        built = await self._read_and_build(mapping)
        current_hash = compute_relational_snapshot_hash(built.payload + built.relation_payload)

        previous_state = await neo4j.get_relational_source_state(mapping.tenant, mapping.source_id)
        previous_row_hashes = (
            previous_state["row_hashes"]
            if previous_state is not None and previous_state["ingest_complete"]
            else {}
        )
        diff = diff_rows(previous_row_hashes, built.row_hashes)

        if not diff.upserted and not diff.deleted and previous_state is not None and previous_state["ingest_complete"]:
            # Row-level diff subsumes the whole-snapshot check ingest() uses
            # -- nothing changed and nothing disappeared, so this is exactly
            # ingest()'s should_skip_ingest() condition, just derived from
            # the row-level checkpoint instead of a separate stored hash.
            # Still refreshes the checkpoint/releases the lease: a skip must
            # not leave a run "in flight" forever.
            await neo4j.complete_relational_ingest_run(
                mapping.tenant, mapping.source_id, run_id, built.row_hashes,
            )
            incremental_report = IncrementalIngestReport(**report.model_dump())
            incremental_report.content_hash = current_hash
            incremental_report.skipped = True
            incremental_report.unchanged = diff.unchanged
            return incremental_report

        from graphrag.graph.shacl_validator import SHACLValidator

        conforms, shacl_report = SHACLValidator.validate_relational_batch(
            built.entities, built.relations, tenant=mapping.tenant,
        )
        report.shacl_conforms = conforms
        if not conforms:
            raise ValueError("relational mapping rejected by SHACL: " + shacl_report)

        catalog = SourceCatalogRepository(neo4j)
        await catalog.upsert_source(SourceSystem(
            id=mapping.source_id,
            tenant=mapping.tenant,
            name=mapping.source_id,
            kind=self.connector.kind,
            uri=self.connector.uri,
            owner="relational-ingestion",
            classification="synthetic" if mapping.tenant == "sustainability" else "internal",
        ))
        await catalog.add_mapping(mapping.as_source_mapping())

        raw = json.dumps(built.payload, sort_keys=True, default=str)
        document = Document(
            id=str(uuid5(NAMESPACE_URL, f"relational-document:{mapping.source_id}")),
            filename=f"relational://{mapping.source_id}",
            source_path=self.connector.uri,
            raw_text=raw,
            content_hash=current_hash,
            tenant=mapping.tenant,
            source_id=mapping.source_id,
            status="done",
            metadata={"mapping_id": mapping.id, "mapping_version": mapping.version,
                      "ontology_version": mapping.ontology_version,
                      "provenance": "local-relational-source"},
        )
        document_id = await self.graph_writer.write_document(document)
        chunk = Chunk(
            id=str(uuid5(NAMESPACE_URL, f"relational-chunk:{document_id}")),
            document_id=document_id,
            text=raw,
            chunk_index=0,
            tenant=mapping.tenant,
            metadata={"source_id": mapping.source_id, "mapping_version": mapping.version},
        )
        await self.graph_writer.write_chunks([chunk])
        written = await self.graph_writer.write_entities(built.entities, chunk)
        entity_map = {entity.id: entity for entity in written}
        await self.graph_writer.write_relations(
            built.relations, entity_map, doc_id=document_id, tenant=mapping.tenant,
        )

        deleted_entity_ids = []
        for key in diff.deleted:
            if not key.startswith("entity:"):
                continue
            _, table, source_key = key.split(":", 2)
            deleted_entity_ids.append(
                str(uuid5(NAMESPACE_URL, f"{mapping.source_id}:{table}:{source_key}"))
            )
        if deleted_entity_ids:
            await neo4j.tombstone_relational_rows(mapping.tenant, mapping.source_id, deleted_entity_ids)

        await neo4j.complete_relational_ingest_run(
            mapping.tenant, mapping.source_id, run_id, built.row_hashes,
        )
        return IncrementalIngestReport(
            **report.model_dump(exclude={"content_hash"}),
            content_hash=current_hash,
            upserted=diff.upserted, deleted=diff.deleted, unchanged=diff.unchanged,
        )

    @staticmethod
    def _timestamp(value: Any) -> datetime | None:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


__all__ = [
    "ConcurrentIngestError", "EntityTableMapping", "RelationTableMapping",
    "RelationalGraphMapping", "IncrementalIngestReport", "MappingValidationReport",
    "TabularSourceConnector", "SQLiteSourceConnector", "PostgreSQLSourceConnector",
    "ExcelWorkbookConnector", "RelationalGraphIngestor",
]
