"""Neo4j connection pool, query runner, and graph MERGE helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver

from graphrag.observability.operational_metrics import (
    record_graph_query, set_graph_pool,
)
from neo4j.exceptions import ServiceUnavailable, TransientError

from graphrag.core.config import get_settings
from graphrag.core.models import (
    Chunk,
    Community,
    Entity,
    IngestionRunManifest,
    IntelligenceArtifact,
    Relation,
    StructuredTable,
)
from graphrag.core.retry import with_retry
from graphrag.enterprise.access import access_params, document_access_predicate, link_access_predicate
from graphrag.enterprise.models import AccessContext, DocumentLink, normalise_document_url

log = structlog.get_logger(__name__)


class Neo4jClient:
    """Thin wrapper around the Neo4j async driver with retry logic."""

    # Every query holds one pooled connection for its duration, so this is
    # also the hard ceiling on graph concurrency for the process. Published as
    # a gauge alongside in-flight count so saturation is visible as a ratio
    # before it shows up as latency.
    MAX_CONNECTION_POOL_SIZE = 50

    @staticmethod
    def _content_access_params(access_context: AccessContext | None = None) -> dict:
        """Build trusted ACL query parameters once per graph call."""
        enabled = bool(get_settings().access_control.get("enabled", False))
        return access_params(access_context, enabled=enabled)

    def __init__(self):
        cfg = get_settings()
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
            max_connection_pool_size=self.MAX_CONNECTION_POOL_SIZE,
            notifications_min_severity="OFF",
        )
        self._filtered_vector_search = False
        self._filtered_vector_indexes: set[str] = set()
        # In-flight query count. The driver does not expose pool occupancy, so
        # this is counted here: it is the same number for our purposes, since
        # a session is held for exactly the span of a query.
        self._in_flight = 0
        set_graph_pool(0, self.MAX_CONNECTION_POOL_SIZE)

    async def detect_capabilities(self) -> dict:
        """Detect Neo4j 2026 tenant-filtered vector indexes explicitly."""
        version_rows = await self.run(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
        )
        version = str(version_rows[0].get("version", "")) if version_rows else ""
        try:
            modern_server = int(version.split(".", 1)[0]) >= 2026
        except ValueError:
            modern_server = False
        filtered_indexes: set[str] = set()
        if modern_server:
            rows = await self.run(
                "SHOW VECTOR INDEXES YIELD name, state, properties, indexProvider "
                "RETURN name, state, properties, indexProvider"
            )
            for row in rows:
                properties = set(row.get("properties") or [])
                if (row.get("state") == "ONLINE" and "tenant" in properties
                        and str(row.get("indexProvider", "")).startswith("vector-2026")):
                    filtered_indexes.add(str(row.get("name")))
        self._filtered_vector_search = {
            "chunk_embeddings", "community_embeddings"
        }.issubset(filtered_indexes)
        self._filtered_vector_indexes = filtered_indexes
        capabilities = {
            "neo4j_version": version,
            "filtered_vector_search": self._filtered_vector_search,
            "filtered_vector_indexes": sorted(filtered_indexes),
        }
        log.info("neo4j.capabilities", **capabilities)
        return capabilities

    async def close(self):
        await self._driver.close()

    @with_retry(exceptions=(TransientError, ServiceUnavailable), max_attempts=3)
    async def run(self, cypher: str, **params) -> list[dict]:
        self._in_flight += 1
        set_graph_pool(self._in_flight, self.MAX_CONNECTION_POOL_SIZE)
        try:
            with record_graph_query():
                async with self._driver.session() as session:
                    result = await session.run(cypher, parameters=params)
                    return [record.data() async for record in result]
        finally:
            self._in_flight -= 1
            set_graph_pool(self._in_flight, self.MAX_CONNECTION_POOL_SIZE)

    # ── Schema initialization ────────────────────────────────────────────────────

    async def init_schema(self):
        version_rows = await self.run(
            "CALL dbms.components() YIELD versions RETURN versions[0] AS version"
        )
        version = str(version_rows[0].get("version", "")) if version_rows else ""
        try:
            modern_server = int(version.split(".", 1)[0]) >= 2026
        except ValueError:
            modern_server = False
        schema_cypher = Path(__file__).parent / "schema.cypher"
        raw = schema_cypher.read_text()
        for fragment in raw.split(";"):
            # Strip comment lines per-fragment (A59: never check the whole fragment
            # for "--" — that skips CREATE statements that follow a comment line)
            lines = [line for line in fragment.splitlines()
                     if not line.strip().startswith("--")]
            stmt = "\n".join(lines).strip()
            if modern_server and stmt.startswith("CREATE VECTOR INDEX"):
                continue
            if stmt:
                result = await self.run(stmt)
                # Consume result so DDL actually executes (A58)
                _ = result
        if modern_server:
            modern_indexes = {
                "chunk_embeddings": "FOR (n:Chunk) ON n.embedding WITH [n.tenant]",
                "community_embeddings": "FOR (n:Community) ON n.embedding WITH [n.tenant]",
                "entity_embeddings": "FOR (n:Entity) ON n.embedding WITH [n.tenant]",
                "community_summary_snapshot_embeddings": (
                    "FOR (n:CommunitySummarySnapshot) ON n.embedding "
                    "WITH [n.tenant, n.valid_from, n.valid_to, n.transaction_from, n.transaction_to]"
                ),
            }
            for name, index_schema in modern_indexes.items():
                await self.run(
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS {index_schema} "
                    "OPTIONS {indexConfig: {`vector.dimensions`: 3072, "
                    "`vector.similarity_function`: 'cosine'}}"
                )
        from graphrag.context_graph.schema import CONTEXT_GRAPH_SCHEMA
        for statement in CONTEXT_GRAPH_SCHEMA:
            await self.run(statement)
        from graphrag.business.schema import BUSINESS_SCHEMA
        for statement in BUSINESS_SCHEMA:
            await self.run(statement)
        await self.detect_capabilities()
        log.info("neo4j.schema_initialized")

    async def get_corpus_state(self, tenant: str = "default") -> dict:
        """Return the tenant's durable retrieval revision and update status."""
        rows = await self.run(
            "OPTIONAL MATCH (s:KGCorpusState {tenant: $tenant}) "
            "RETURN coalesce(s.revision, 0) AS revision, "
            "coalesce(s.updating, false) AS updating, "
            "coalesce(s.active_updates, 0) AS active_updates",
            tenant=tenant,
        )
        return rows[0] if rows else {"revision": 0, "updating": False, "active_updates": 0}

    async def begin_corpus_update(self, tenant: str = "default", *, reason: str = "ingestion") -> None:
        """Disable answer-cache reads while an ingestion mutates this tenant."""
        await self.run(
            "MERGE (s:KGCorpusState {tenant: $tenant}) "
            "ON CREATE SET s.revision = 0, s.active_updates = 0 "
            "SET s.active_updates = coalesce(s.active_updates, 0) + 1, "
            "s.updating = true, s.update_started_at = datetime(), "
            "s.last_reason = $reason",
            tenant=tenant,
            reason=reason,
        )

    async def complete_corpus_update(
        self,
        tenant: str = "default",
        *,
        reason: str = "ingestion",
        outcome: str = "completed",
    ) -> int:
        """Atomically publish a new corpus revision after successful ingestion."""
        rows = await self.run(
            "MERGE (s:KGCorpusState {tenant: $tenant}) "
            "ON CREATE SET s.revision = 0, s.active_updates = 0 "
            "WITH s, CASE WHEN coalesce(s.active_updates, 0) > 0 "
            "THEN s.active_updates - 1 ELSE 0 END AS remaining "
            "SET s.revision = coalesce(s.revision, 0) + 1, "
            "s.active_updates = remaining, s.updating = remaining > 0, "
            "s.updated_at = datetime(), s.last_reason = $reason, "
            "s.last_outcome = $outcome "
            "RETURN s.revision AS revision",
            tenant=tenant,
            reason=reason,
            outcome=outcome,
        )
        return int(rows[0]["revision"]) if rows else 0

    async def advance_corpus_revision(self, tenant: str = "default", *, reason: str) -> int:
        """Publish a completed single-step retrieval-state mutation."""
        rows = await self.run(
            "MERGE (s:KGCorpusState {tenant: $tenant}) "
            "ON CREATE SET s.revision = 0, s.active_updates = 0 "
            "SET s.revision = coalesce(s.revision, 0) + 1, "
            "s.updating = coalesce(s.active_updates, 0) > 0, "
            "s.updated_at = datetime(), s.last_reason = $reason, "
            "s.last_outcome = 'completed' "
            "RETURN s.revision AS revision",
            tenant=tenant,
            reason=reason,
        )
        return int(rows[0]["revision"]) if rows else 0

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
        source_id: str | None = None,
        content_hash: str = "",
        metadata_envelope: dict | None = None,
        access_policy: dict | None = None,
    ) -> str:
        """MERGE on the document's real identity, (tenant, filename) — not on
        doc_id, which is a fresh uuid4() every ingestion run and so can never
        match an existing node. Keying on it made every re-ingest an
        unconditional CREATE: a partial aerospace re-ingest silently duplicated
        4 documents and 38% of that tenant's chunks (see tasks/lessons.md A136).

        Returns the *canonical* id — the one already stored, if this document
        existed before, otherwise doc_id. Callers MUST use the returned id for
        every downstream write (chunks, relations, supersession) instead of the
        id they passed in.
        """
        metadata_envelope = metadata_envelope or {}
        access_policy = access_policy or {}
        rows = await self.run(
            """
            OPTIONAL MATCH (source:KGSource {tenant: $tenant, id: $source_id})
            WITH source
            WHERE $source_id IS NULL OR source IS NOT NULL
            MERGE (d:Document {tenant: $tenant, filename: $filename})
            ON CREATE SET d.id = $id, d.created_at = datetime(), d.recorded_at = datetime()
            SET d.ingested_at     = $ingested_at,
                d.status          = 'done',
                d.authority_level = $authority_level,
                d.valid_from      = $valid_from,
                d.valid_to        = $valid_to,
                d.source_id       = $source_id,
                d.content_hash    = $content_hash,
                // Three-tier governed metadata. Neo4j node properties cannot
                // hold a nested map, so the full envelope is retained as
                // canonical JSON while queryable universal fields are flat.
                d.metadata_envelope_json = $metadata_envelope_json,
                d.collection      = $collection,
                d.metadata_schema_version = $metadata_schema_version,
                d.source_system   = $source_system,
                d.external_id     = $external_id,
                d.source_url      = $source_url,
                d.source_version  = $source_version,
                d.content_type    = $content_type,
                d.classification  = $classification,
                // ACL fields are document-owned and used by every retrieval
                // query. A missing/unknown ACL is denied once enforcement is
                // enabled; enforcement itself is a deployment configuration.
                d.access_mode     = $access_mode,
                d.acl_state       = $acl_state,
                d.allow_principals = $allow_principals,
                d.deny_principals = $deny_principals,
                d.requires_group_resolution = $requires_group_resolution,
                // A queue retry must never mistake an interrupted write for a
                // completed document merely because the content hash landed.
                d.ingest_complete = false,
                // Re-ingesting a file that had been tombstoned resurrects it:
                // the source reappeared on disk, so its chunks must become
                // retrievable again. Clearing here (rather than leaving the
                // operator to notice) keeps tombstone state derived from the
                // corpus rather than drifting from it.
                d.is_deleted      = false,
                d.deleted_at      = null
            FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
              MERGE (d)-[:INGESTED_FROM]->(source))
            RETURN d.id AS doc_id
            """,
            id=doc_id,
            filename=filename,
            ingested_at=ingested_at,
            authority_level=authority_level,
            valid_from=valid_from,
            valid_to=valid_to,
            tenant=tenant,
            source_id=source_id,
            content_hash=content_hash,
            metadata_envelope_json=json.dumps(metadata_envelope, sort_keys=True, default=str),
            collection=str(metadata_envelope.get("collection") or "default"),
            metadata_schema_version=str(metadata_envelope.get("schema_version") or "v1"),
            source_system=str(metadata_envelope.get("source_system") or "manual"),
            external_id=str(metadata_envelope.get("external_id") or ""),
            source_url=normalise_document_url(str(metadata_envelope.get("source_url") or "")),
            source_version=str(metadata_envelope.get("source_version") or ""),
            content_type=str(metadata_envelope.get("content_type") or "text/plain"),
            classification=str(metadata_envelope.get("classification") or ""),
            access_mode=str(access_policy.get("access_mode") or "tenant"),
            acl_state=str(access_policy.get("acl_state") or "known"),
            allow_principals=list(access_policy.get("allow_principals") or []),
            deny_principals=list(access_policy.get("deny_principals") or []),
            requires_group_resolution=bool(access_policy.get("requires_group_resolution", False)),
        )
        if source_id and not rows:
            raise ValueError("document source is missing or belongs to another tenant")
        return rows[0]["doc_id"] if rows else doc_id

    async def mark_document_ingest_complete(self, doc_id: str, tenant: str = "default") -> None:
        """Mark the durable queue checkpoint only after every write stage succeeds."""
        await self.run(
            """
            MATCH (d:Document {id: $doc_id, tenant: $tenant})
            SET d.ingest_complete = true, d.ingest_completed_at = datetime()
            """,
            doc_id=doc_id,
            tenant=tenant,
        )

    async def get_document_states(self, tenant: str = "default") -> dict[str, dict]:
        """Return {filename: {"content_hash": str, "is_deleted": bool}} for a tenant.

        Feeds the bulk ingest CLI's incremental decision: skip a file whose
        stored hash matches what is on disk, re-ingest one whose hash differs,
        ingest one that is absent. Replaces the old binary `ingest_complete`
        checkpoint, which skipped a document forever once ingested — so an
        EDITED source file was never re-ingested without a full --wipe.

        `ingest_complete` is still returned and still matters: merge_document
        writes content_hash at the START of a document's write, so a run that
        crashes midway leaves a hash that already matches the file on disk.
        Hash alone would then mark a PARTIALLY ingested document as
        "unchanged" and skip it forever. A document is only safely skippable
        when its hash matches AND its previous write actually completed.
        """
        rows = await self.run(
            """
            MATCH (d:Document {tenant: $tenant})
            RETURN d.filename AS filename,
                   coalesce(d.content_hash, '') AS content_hash,
                   coalesce(d.is_deleted, false) AS is_deleted,
                   coalesce(d.ingest_complete, false) AS ingest_complete
            """,
            tenant=tenant,
        )
        return {
            r["filename"]: {
                "content_hash": r["content_hash"],
                "is_deleted": r["is_deleted"],
                "ingest_complete": r["ingest_complete"],
            }
            for r in rows if r.get("filename")
        }

    async def tombstone_documents(
        self, filenames: list[str], tenant: str = "default",
    ) -> int:
        """Soft-delete documents whose source file has disappeared.

        Deliberately NOT a physical delete. Erasing data is GDPR erasure's job
        (graphrag/graph/gdpr.py) and is irreversible; a source file vanishing
        from a corpus directory is far more often a sync glitch, a partial
        checkout, or a rename than a genuine deletion request. Tombstoning
        excludes the document's chunks from retrieval while leaving every node
        recoverable — and merge_document clears the flag automatically if the
        file comes back.

        Returns the number of documents newly tombstoned (already-tombstoned
        ones are not recounted, so this is safe to run on every ingest).
        """
        if not filenames:
            return 0
        rows = await self.run(
            """
            UNWIND $filenames AS fname
            MATCH (d:Document {tenant: $tenant, filename: fname})
            WHERE coalesce(d.is_deleted, false) = false
            SET d.is_deleted = true,
                d.deleted_at = datetime()
            RETURN count(d) AS tombstoned
            """,
            filenames=filenames,
            tenant=tenant,
        )
        return rows[0]["tombstoned"] if rows else 0

    async def merge_chunk(self, chunk: Chunk, tenant: str = "default"):
        """MERGE on (tenant, document_id, chunk_index) — stable across re-ingestion and
        re-chunking, unlike chunk.id (fresh uuid4() every run). Also writes
        document_id itself, which chunks never carried before this change
        (see backfill_chunk_document_id.py) — this activates the existing
        chunk_doc index and is what counterfactual.py's document-removal
        simulation queries on, previously matching nothing.
        """
        await self.run(
            """
            MERGE (c:Chunk {tenant: $tenant, document_id: $doc_id, chunk_index: $chunk_index})
            ON CREATE SET c.id = $id
            SET c.text      = $text,
                c.embedding = $embedding
            WITH c
            MATCH (d:Document {id: $doc_id, tenant: $tenant})
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
            MERGE (c:Chunk {tenant: $tenant, document_id: row.doc_id, chunk_index: row.chunk_index})
            ON CREATE SET c.id = row.id
            SET c.text      = row.text,
                c.embedding = row.embedding
            WITH c, row
            MATCH (d:Document {id: row.doc_id, tenant: $tenant})
            MERGE (c)-[:PART_OF]->(d)
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_intelligence_artifacts(
        self, artifacts: list[IntelligenceArtifact], tenant: str = "default",
    ) -> None:
        """Persist source-grounded artifact nodes and their evidence links."""
        if not artifacts:
            return
        rows = [
            {
                "id": artifact.id,
                "artifact_type": artifact.artifact_type,
                "text": artifact.text,
                "evidence_quote": artifact.evidence_quote,
                "confidence": artifact.confidence,
                "source_chunk_id": artifact.source_chunk_id,
                "source_doc_id": artifact.source_doc_id,
                "entity_names": artifact.entity_names,
                "event_start": artifact.event_start.isoformat() if artifact.event_start else None,
                "event_end": artifact.event_end.isoformat() if artifact.event_end else None,
                "extraction_model": artifact.extraction_model,
                "prompt_version": artifact.prompt_version,
            }
            for artifact in artifacts
        ]
        await self.run(
            """
            UNWIND $rows AS row
            MERGE (a:IntelligenceArtifact {tenant: $tenant, id: row.id})
            ON CREATE SET a.created_at = datetime()
            SET a.artifact_type = row.artifact_type,
                a.text = row.text,
                a.evidence_quote = row.evidence_quote,
                a.confidence = row.confidence,
                a.source_chunk_id = row.source_chunk_id,
                a.source_doc_id = row.source_doc_id,
                a.event_start = row.event_start,
                a.event_end = row.event_end,
                a.extraction_model = row.extraction_model,
                a.prompt_version = row.prompt_version,
                a.updated_at = datetime()
            WITH a, row
            MATCH (chunk:Chunk {tenant: $tenant, id: row.source_chunk_id})
            MERGE (a)-[:DERIVED_FROM]->(chunk)
            WITH a, row
            MATCH (document:Document {tenant: $tenant, id: row.source_doc_id})
            MERGE (a)-[:ASSERTED_IN]->(document)
            WITH a, row
            UNWIND row.entity_names AS entity_name
            MATCH (entity:Entity {tenant: $tenant, name: entity_name})
            MERGE (a)-[:ASSERTS_ABOUT]->(entity)
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_document_links(
        self,
        doc_id: str,
        links: list[DocumentLink],
        *,
        tenant: str = "default",
        access_policy: dict | None = None,
    ) -> int:
        """Persist explicit link references and materialise resolved LINKS_TO edges.

        ``DocumentLinkReference`` makes the import order harmless: a source can
        arrive before its target, retain its source-backed provenance, and be
        resolved after a later target ingestion.  The query only links documents
        in the same tenant and carries the source policy snapshot onto the edge.
        """
        policy = access_policy or {}
        rows = [
            {
                "target_url": link.target_url,
                "anchor_text": link.anchor_text,
                "source_locator": link.source_locator,
                "observed_at": link.observed_at.isoformat(),
                "source_system": link.source_system,
                "source_version": link.source_version,
                "link_key": hashlib.sha256(
                    f"{link.target_url}\0{link.anchor_text}\0{link.source_locator}".encode()
                ).hexdigest(),
            }
            for link in links
        ]
        link_keys = [row["link_key"] for row in rows]
        # A re-ingest is a replacement of the source document revision, not an
        # append-only crawl. Remove links that disappeared from that revision
        # before adding current observations; otherwise an obsolete HTML anchor
        # would remain a retrievable path indefinitely.
        await self.run(
            """
            MATCH (source:Document {id: $doc_id, tenant: $tenant})-[link:LINKS_TO {tenant: $tenant}]->()
            WHERE NOT link.link_key IN $link_keys
            DELETE link
            """,
            doc_id=doc_id,
            tenant=tenant,
            link_keys=link_keys,
        )
        await self.run(
            """
            MATCH (source:Document {id: $doc_id, tenant: $tenant})-[declaration:DECLARES_LINK {tenant: $tenant}]->(ref:DocumentLinkReference {tenant: $tenant})
            WHERE NOT ref.link_key IN $link_keys
            DETACH DELETE ref
            """,
            doc_id=doc_id,
            tenant=tenant,
            link_keys=link_keys,
        )
        if not rows:
            return 0
        result = await self.run(
            """
            MATCH (source:Document {id: $doc_id, tenant: $tenant})
            UNWIND $links AS row
            MERGE (ref:DocumentLinkReference {
                tenant: $tenant, source_document_id: $doc_id, link_key: row.link_key
            })
            ON CREATE SET ref.created_at = datetime(), ref.recorded_at = datetime()
            SET ref.target_url = row.target_url,
                ref.anchor_text = row.anchor_text,
                ref.source_locator = row.source_locator,
                ref.observed_at = datetime(row.observed_at),
                ref.source_system = row.source_system,
                ref.source_version = row.source_version,
                ref.access_mode = $access_mode,
                ref.acl_state = $acl_state,
                ref.allow_principals = $allow_principals,
                ref.deny_principals = $deny_principals,
                ref.requires_group_resolution = $requires_group_resolution,
                ref.updated_at = datetime()
            MERGE (source)-[:DECLARES_LINK {tenant: $tenant, link_key: row.link_key}]->(ref)
            WITH source, ref, row
            OPTIONAL MATCH (target:Document {tenant: $tenant, source_url: row.target_url})
            WHERE coalesce(target.is_deleted, false) = false AND target.id <> source.id
            FOREACH (_ IN CASE WHEN target IS NULL THEN [] ELSE [1] END |
              MERGE (source)-[link:LINKS_TO {tenant: $tenant, link_key: row.link_key}]->(target)
              ON CREATE SET link.recorded_at = datetime()
              SET link.target_url = row.target_url,
                  link.anchor_text = row.anchor_text,
                  link.source_locator = row.source_locator,
                  link.observed_at = datetime(row.observed_at),
                  link.source_system = row.source_system,
                  link.source_version = row.source_version,
                  link.provenance_ref = ref.link_key,
                  link.access_mode = $access_mode,
                  link.acl_state = $acl_state,
                  link.allow_principals = $allow_principals,
                  link.deny_principals = $deny_principals,
                  link.requires_group_resolution = $requires_group_resolution,
                  link.updated_at = datetime()
            )
            RETURN count(ref) AS references
            """,
            doc_id=doc_id,
            tenant=tenant,
            links=rows,
            access_mode=str(policy.get("access_mode") or "tenant"),
            acl_state=str(policy.get("acl_state") or "known"),
            allow_principals=list(policy.get("allow_principals") or []),
            deny_principals=list(policy.get("deny_principals") or []),
            requires_group_resolution=bool(policy.get("requires_group_resolution", False)),
        )
        return int(result[0].get("references", 0)) if result else 0

    async def reconcile_document_links(self, doc_id: str, tenant: str = "default") -> int:
        """Resolve any durable explicit reference whose source or target just landed."""
        rows = await self.run(
            """
            MATCH (target:Document {id: $doc_id, tenant: $tenant})
            MATCH (source:Document {tenant: $tenant})-[:DECLARES_LINK]->(ref:DocumentLinkReference {tenant: $tenant})
            WHERE ref.target_url = target.source_url
              AND source.id <> target.id
              AND coalesce(source.is_deleted, false) = false
              AND coalesce(target.is_deleted, false) = false
            MERGE (source)-[link:LINKS_TO {tenant: $tenant, link_key: ref.link_key}]->(target)
            ON CREATE SET link.recorded_at = datetime()
            SET link.target_url = ref.target_url,
                link.anchor_text = ref.anchor_text,
                link.source_locator = ref.source_locator,
                link.observed_at = ref.observed_at,
                link.source_system = ref.source_system,
                link.source_version = ref.source_version,
                link.provenance_ref = ref.link_key,
                link.access_mode = ref.access_mode,
                link.acl_state = ref.acl_state,
                link.allow_principals = ref.allow_principals,
                link.deny_principals = ref.deny_principals,
                link.requires_group_resolution = ref.requires_group_resolution,
                link.updated_at = datetime()
            RETURN count(link) AS resolved
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return int(rows[0].get("resolved", 0)) if rows else 0

    async def merge_structured_tables(
        self, tables: list[StructuredTable], tenant: str = "default",
    ) -> None:
        """Persist structured table payloads without flattening away cell structure."""
        if not tables:
            return
        rows = [
            {
                "id": table.id,
                "document_id": table.document_id,
                "table_index": table.table_index,
                "caption": table.caption,
                "columns": table.columns,
                "rows_json": json.dumps(table.rows, ensure_ascii=False),
                "jsonld": json.dumps(table.as_jsonld(), ensure_ascii=False),
                "source_page": table.source_page,
                "extraction_method": table.extraction_method,
                "source_chunk_id": table.source_chunk_id,
            }
            for table in tables
        ]
        await self.run(
            """
            UNWIND $rows AS row
            MERGE (table:StructuredTable {tenant: $tenant, document_id: row.document_id, table_index: row.table_index})
            ON CREATE SET table.id = row.id, table.created_at = datetime()
            SET table.caption = row.caption,
                table.columns = row.columns,
                table.rows_json = row.rows_json,
                table.jsonld = row.jsonld,
                table.source_page = row.source_page,
                table.extraction_method = row.extraction_method,
                table.source_chunk_id = row.source_chunk_id,
                table.updated_at = datetime()
            WITH table, row
            MATCH (document:Document {tenant: $tenant, id: row.document_id})
            MERGE (table)-[:EXTRACTED_FROM]->(document)
            """,
            rows=rows,
            tenant=tenant,
        )

    async def merge_temporal_periods(
        self, chunk_id: str, periods: list[dict[str, str]], tenant: str = "default",
    ) -> None:
        """Materialise explicit time mentions and their calendar hierarchy."""
        if not periods:
            return
        await self.run(
            """
            MATCH (chunk:Chunk {tenant: $tenant, id: $chunk_id})
            UNWIND $periods AS period
            MERGE (node:TimePeriod {tenant: $tenant, value: period.value})
            ON CREATE SET node.kind = period.kind, node.created_at = datetime()
            SET node.kind = period.kind, node.updated_at = datetime()
            MERGE (chunk)-[:MENTIONS_TIME]->(node)
            WITH node, period
            FOREACH (_ IN CASE WHEN period.parent = '' THEN [] ELSE [1] END |
                MERGE (parent:TimePeriod {tenant: $tenant, value: period.parent})
                ON CREATE SET parent.kind = CASE
                    WHEN period.parent CONTAINS '-Q' THEN 'quarter'
                    WHEN period.parent CONTAINS '-' THEN 'month'
                    ELSE 'year' END,
                    parent.created_at = datetime()
                MERGE (node)-[:IN_PERIOD]->(parent))
            """,
            chunk_id=chunk_id,
            periods=periods,
            tenant=tenant,
        )

    async def upsert_ingestion_manifest(self, manifest: IngestionRunManifest) -> None:
        """Persist an ingestion receipt; unknown provider cost stays explicitly unknown."""
        await self.run(
            """
            MERGE (manifest:IngestionRunManifest {tenant: $tenant, id: $id})
            ON CREATE SET manifest.created_at = datetime()
            SET manifest.job_id = $job_id,
                manifest.document_id = $document_id,
                manifest.filename = $filename,
                manifest.content_hash = $content_hash,
                manifest.correlation_id = $correlation_id,
                manifest.model_provider = $model_provider,
                manifest.model_version = $model_version,
                manifest.prompt_versions_json = $prompt_versions_json,
                manifest.stage_metrics_json = $stage_metrics_json,
                manifest.status = $status,
                manifest.started_at = $started_at,
                manifest.completed_at = $completed_at,
                manifest.error = $error,
                manifest.integrity_hash = $integrity_hash,
                manifest.updated_at = datetime()
            WITH manifest
            OPTIONAL MATCH (document:Document {tenant: $tenant, id: $document_id})
            FOREACH (_ IN CASE WHEN document IS NULL THEN [] ELSE [1] END |
                MERGE (manifest)-[:INGESTS]->(document))
            """,
            id=manifest.id,
            tenant=manifest.tenant,
            job_id=manifest.job_id,
            document_id=manifest.document_id,
            filename=manifest.filename,
            content_hash=manifest.content_hash,
            correlation_id=manifest.correlation_id,
            model_provider=manifest.model_provider,
            model_version=manifest.model_version,
            prompt_versions_json=json.dumps(manifest.prompt_versions, sort_keys=True),
            stage_metrics_json=json.dumps(manifest.stage_metrics, sort_keys=True),
            status=manifest.status,
            started_at=manifest.started_at.isoformat(),
            completed_at=manifest.completed_at.isoformat() if manifest.completed_at else None,
            error=manifest.error,
            integrity_hash=manifest.integrity_hash,
        )

    async def delete_stale_chunks(self, doc_id: str, keep_count: int, tenant: str = "default") -> int:
        """Delete chunks left over from a previous ingestion of doc_id whose
        chunk_index no longer has a counterpart in the current chunk set —
        i.e. a re-chunk that produced fewer chunks than before (the
        section-aware chunker rewrite did exactly this). Without this, the
        surplus chunks stay in the retrieval pool with no owner. DETACH so
        MENTIONS/PART_OF edges go with them. Returns the number deleted.
        """
        rows = await self.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})
            WHERE c.chunk_index >= $keep_count
            WITH collect(c) AS stale, count(c) AS n
            UNWIND stale AS c
            DETACH DELETE c
            RETURN head(collect(n)) AS deleted
            """,
            doc_id=doc_id,
            keep_count=keep_count,
            tenant=tenant,
        )
        deleted = int(rows[0]["deleted"]) if rows and rows[0].get("deleted") else 0
        if deleted:
            log.info("neo4j.stale_chunks_deleted", doc_id=doc_id, deleted=deleted)
        return deleted

    async def reconcile_document_evidence(self, doc_id: str, tenant: str = "default") -> dict[str, int]:
        """Remove a document's old mentions and relation provenance.

        A changed source is re-extracted in place.  Its old chunks retain their
        stable identities, so `MERGE` alone cannot remove facts no longer in
        the revised text.  Relationships supported solely by this document are
        deleted; multi-source relationships keep their other provenance.
        """
        mention_rows = await self.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})-[m:MENTIONS]->()
            DELETE m
            RETURN count(m) AS removed_mentions
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        deleted_rows = await self.run(
            """
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]-()
            WHERE $doc_id IN coalesce(r.source_doc_ids, [r.source_doc_id])
            WITH r, [source IN coalesce(r.source_doc_ids, [r.source_doc_id])
                     WHERE source <> $doc_id] AS remaining_sources
            WHERE size(remaining_sources) = 0
            DELETE r
            RETURN count(*) AS deleted_relations
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        updated_rows = await self.run(
            """
            MATCH ()-[r:RELATES_TO {tenant: $tenant}]-()
            WHERE $doc_id IN coalesce(r.source_doc_ids, [r.source_doc_id])
            WITH r, [source IN coalesce(r.source_doc_ids, [r.source_doc_id])
                     WHERE source <> $doc_id] AS remaining_sources
            WHERE size(remaining_sources) > 0
            SET r.source_doc_ids = remaining_sources,
                r.source_doc_id = remaining_sources[0]
            RETURN count(*) AS retained_relations
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        artifact_rows = await self.run(
            """
            MATCH (a:IntelligenceArtifact {tenant: $tenant, source_doc_id: $doc_id})
            DETACH DELETE a
            RETURN count(*) AS deleted_artifacts
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        table_rows = await self.run(
            """
            MATCH (table:StructuredTable {tenant: $tenant, document_id: $doc_id})
            DETACH DELETE table
            RETURN count(*) AS deleted_tables
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        return {
            "removed_mentions": int(mention_rows[0].get("removed_mentions", 0)) if mention_rows else 0,
            "deleted_relations": int(deleted_rows[0].get("deleted_relations", 0)) if deleted_rows else 0,
            "retained_relations": int(updated_rows[0].get("retained_relations", 0)) if updated_rows else 0,
            "deleted_artifacts": int(artifact_rows[0].get("deleted_artifacts", 0)) if artifact_rows else 0,
            "deleted_tables": int(table_rows[0].get("deleted_tables", 0)) if table_rows else 0,
        }

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
            MATCH (c:Chunk {id: $chunk_id, tenant: $tenant})
            MATCH (e:Entity {name: $entity_name, type: $entity_type, tenant: $tenant})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_name=entity_name,
            entity_type=entity_type,
            tenant=tenant,
        )

    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> list[dict]:
        """Same MERGE semantics as merge_entity(), one round-trip for the batch.

        Returns one row per input entity: {name, type, prior_similarity}.
        prior_similarity is the cosine similarity between the node's embedding
        *before* this write and the incoming embedding — null on first create,
        or when either embedding is missing/empty. A low value on a match
        means the same (name, type, tenant) key was just asked to represent
        two semantically distant things — the "Apple the company" vs. "Apple
        the fruit" collision that (name, type) alone cannot prevent. The
        caller (GraphWriter.write_entities) logs a warning when this drops
        below ingestion.entity_collision_similarity_min so it surfaces for
        review instead of silently blending the two senses into one node.
        """
        if not entities:
            return []
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
        return await self.run(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (existing:Entity {name: row.name, type: row.type, tenant: $tenant})
            WITH row, existing,
                 CASE
                   WHEN existing IS NOT NULL
                        AND existing.embedding IS NOT NULL AND size(existing.embedding) > 0
                        AND row.embedding IS NOT NULL AND size(row.embedding) > 0
                        AND size(existing.embedding) = size(row.embedding)
                   THEN vector.similarity.cosine(existing.embedding, row.embedding)
                   ELSE null
                 END AS prior_similarity
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
            RETURN row.name AS name, row.type AS type, prior_similarity
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
            MATCH (c:Chunk {id: $chunk_id, tenant: $tenant})
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
            // Snapshot the contributing-document list BEFORE the SET below
            // rewrites it, so the confidence guard tests the pre-update state
            // rather than depending on SET clause evaluation order.
            WITH r, coalesce(r.source_doc_ids, []) AS prior_docs
            SET r.weight           = $weight,
                r.extracted_at     = $extracted_at,
                r.source_doc_id    = $source_doc_id,
                r.source_type      = $source_type,
                r.constraint_type  = $constraint_type,
                r.confidence_state = $confidence_state,
                r.valid_from       = $valid_from,
                r.valid_to         = $valid_to,
                r.tenant           = $tenant,
                // Accumulate all contributing document IDs as a list so that
                // contradiction detection can see every source even after
                // multiple merges collapse to a single edge.
                r.source_doc_ids   = CASE
                    WHEN $source_doc_id IN prior_docs THEN prior_docs
                    ELSE prior_docs + [$source_doc_id]
                END,
                // Bayesian accumulation treats each contributing document as an
                // INDEPENDENT observation. Re-ingesting a document is not a new
                // observation, so it must not raise confidence: without this
                // guard, ingesting the same unchanged file at 0.8 twice yields
                // 0.96, then 0.992 — silent corruption that compounds on every
                // re-run. The source_doc_ids write directly above was already
                // guarded against the same repeat; confidence simply wasn't.
                // See docs/context_graph_gap_plan.md F2.
                r.confidence       = CASE
                    WHEN r.confidence IS NULL              THEN $confidence
                    WHEN $source_doc_id IN prior_docs      THEN r.confidence
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
            confidence_state=rel.confidence_state,
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
            // Snapshot the contributing-document list BEFORE the SET below
            // rewrites it, so the confidence guard tests the pre-update state
            // rather than depending on SET clause evaluation order.
            WITH r, row, coalesce(r.source_doc_ids, []) AS prior_docs
            SET r.weight           = row.weight,
                r.extracted_at     = row.extracted_at,
                r.source_doc_id    = row.source_doc_id,
                r.source_type      = row.source_type,
                r.constraint_type  = row.constraint_type,
                r.confidence_state = row.confidence_state,
                r.valid_from       = row.valid_from,
                r.valid_to         = row.valid_to,
                r.tenant           = $tenant,
                r.source_doc_ids   = CASE
                    WHEN row.source_doc_id IN prior_docs THEN prior_docs
                    ELSE prior_docs + [row.source_doc_id]
                END,
                // Re-ingesting a document is not an independent observation,
                // so it must not raise confidence. See the identical guard in
                // merge_relation and docs/context_graph_gap_plan.md F2.
                r.confidence       = CASE
                    WHEN r.confidence IS NULL             THEN row.confidence
                    WHEN row.source_doc_id IN prior_docs  THEN r.confidence
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
            MERGE (c:Community {id: $id, tenant: $tenant})
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
            tenant=community.tenant,
        )
        for entity_id in community.member_entity_ids:
            await self.run(
                """
                MATCH (e:Entity {id: $entity_id, tenant: $tenant})
                MATCH (c:Community {id: $community_id, tenant: $tenant})
                MERGE (e)-[:MEMBER_OF]->(c)
                """,
                entity_id=entity_id,
                community_id=community.id,
                tenant=community.tenant,
            )
        if community.summary and community.embedding:
            await self._snapshot_community_summary(community)

    async def _snapshot_community_summary(self, community: Community) -> str:
        """Append an immutable summary version and its evidence lineage."""
        rows = await self.run(
            """
            MATCH (c:Community {tenant: $tenant, id: $community_id})
            OPTIONAL MATCH (e:Entity {tenant: $tenant})-[:MEMBER_OF]->(c)
            OPTIONAL MATCH (ch:Chunk {tenant: $tenant})-[:MENTIONS]->(e)
            OPTIONAL MATCH (ch)-[:PART_OF]->(d:Document {tenant: $tenant})
            RETURN collect(DISTINCT e.id) AS entity_ids,
                   collect(DISTINCT ch.id) AS chunk_ids,
                   collect(DISTINCT d.id) AS document_ids,
                   collect(DISTINCT coalesce(toString(ch.updated_at), toString(ch.created_at), 'v1')) AS chunk_versions,
                   collect(DISTINCT coalesce(toString(d.updated_at), toString(d.ingested_at), 'v1')) AS document_versions,
                   collect(DISTINCT toString(d.valid_from)) AS valid_froms,
                   collect(DISTINCT toString(d.valid_to)) AS valid_tos
            """,
            tenant=community.tenant,
            community_id=community.id,
        )
        lineage = rows[0] if rows else {}
        canonical = {
            "community_id": community.id,
            "tenant": community.tenant,
            "level": community.level,
            "summary": community.summary,
            "entity_ids": sorted(value for value in lineage.get("entity_ids", []) if value),
            "chunk_ids": sorted(value for value in lineage.get("chunk_ids", []) if value),
            "document_ids": sorted(value for value in lineage.get("document_ids", []) if value),
            "chunk_versions": sorted(value for value in lineage.get("chunk_versions", []) if value),
            "document_versions": sorted(value for value in lineage.get("document_versions", []) if value),
        }
        content_hash = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        snapshot_id = f"{community.id}:{content_hash[:20]}"
        valid_froms = sorted(
            value for value in lineage.get("valid_froms", []) if value and value != "None"
        )
        valid_tos = sorted(
            value for value in lineage.get("valid_tos", []) if value and value != "None"
        )
        now = datetime.now(timezone.utc).isoformat()
        await self.run(
            """
            MATCH (c:Community {tenant: $tenant, id: $community_id})
            OPTIONAL MATCH (c)-[:HAS_SUMMARY_VERSION]->(current:CommunitySummarySnapshot)
            WHERE current.transaction_to IS NULL
            WITH c, current
            FOREACH (_ IN CASE WHEN current IS NOT NULL AND current.content_hash <> $content_hash
                               THEN [1] ELSE [] END |
              SET current.transaction_to = datetime($now),
                  current.valid_to = coalesce(current.valid_to, datetime($valid_from)))
            MERGE (s:CommunitySummarySnapshot {tenant: $tenant, id: $snapshot_id})
            ON CREATE SET s.community_id = $community_id, s.level = $level,
                          s.summary = $summary, s.embedding = $embedding,
                          s.content_hash = $content_hash,
                          s.entity_ids = $entity_ids, s.chunk_ids = $chunk_ids,
                          s.chunk_versions = $chunk_versions,
                          s.document_ids = $document_ids,
                          s.document_versions = $document_versions,
                          s.valid_from = datetime($valid_from),
                          s.valid_to = CASE WHEN $valid_to IS NULL THEN null ELSE datetime($valid_to) END,
                          s.transaction_from = datetime($now), s.transaction_to = null,
                          s.recorded_at = datetime($now)
            MERGE (c)-[:HAS_SUMMARY_VERSION]->(s)
            WITH s
            UNWIND CASE WHEN size($chunk_ids) = 0 THEN [null] ELSE $chunk_ids END AS chunk_id
            OPTIONAL MATCH (ch:Chunk {tenant: $tenant, id: chunk_id})
            FOREACH (_ IN CASE WHEN ch IS NULL THEN [] ELSE [1] END |
              MERGE (s)-[:SUPPORTED_BY]->(ch))
            WITH DISTINCT s
            UNWIND CASE WHEN size($document_ids) = 0 THEN [null] ELSE $document_ids END AS document_id
            OPTIONAL MATCH (d:Document {tenant: $tenant, id: document_id})
            FOREACH (_ IN CASE WHEN d IS NULL THEN [] ELSE [1] END |
              MERGE (s)-[:DERIVED_FROM]->(d))
            RETURN s.id AS snapshot_id
            """,
            tenant=community.tenant,
            community_id=community.id,
            snapshot_id=snapshot_id,
            level=community.level,
            summary=community.summary,
            embedding=community.embedding,
            content_hash=content_hash,
            entity_ids=canonical["entity_ids"],
            chunk_ids=canonical["chunk_ids"],
            chunk_versions=canonical["chunk_versions"],
            document_ids=canonical["document_ids"],
            document_versions=canonical["document_versions"],
            valid_from=valid_froms[-1] if valid_froms else now,
            valid_to=valid_tos[0] if valid_tos else None,
            now=now,
        )
        return snapshot_id

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
        self,
        embedding: list[float],
        top_k: int = 10,
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
        access_context: AccessContext | None = None,
    ) -> list[dict]:
        """ANN search over Chunk.embedding using Neo4j vector index.
        Filters by tenant, source-document temporal boundaries, and excludes
        chunks whose mentioned entities are quarantined.

        Over-fetches before tenant-filtering — same tenant-starvation risk
        as vector_search_communities, see that method's docstring and
        tasks/lessons.md A146.
        """
        if self.__dict__.get("_filtered_vector_search", False) and tenant != "default":
            return await self.run(
                """
                MATCH (c:Chunk)
                  SEARCH c IN (
                    VECTOR INDEX chunk_embeddings
                    FOR $embedding
                    WHERE c.tenant = $tenant
                    LIMIT $top_k
                  ) SCORE AS score
                OPTIONAL MATCH (c)-[:PART_OF]->(d:Document {tenant: $tenant})
                WHERE (d IS NULL OR coalesce(d.is_deleted, false) = false)
                  AND ($valid_at IS NULL OR (
                    d IS NOT NULL
                    AND (d.valid_from IS NULL OR d.valid_from <= datetime($valid_at))
                    AND (d.valid_to IS NULL OR d.valid_to > datetime($valid_at))))
                  AND ($transaction_at IS NULL OR (
                    d IS NOT NULL AND (coalesce(d.recorded_at, d.created_at) IS NULL
                    OR coalesce(d.recorded_at, d.created_at) <= datetime($transaction_at))))
                  AND NOT EXISTS {
                    MATCH (c)-[:MENTIONS]->(e:Entity {tenant: $tenant})
                    WHERE e.quarantined = true
                  }
                  """ + document_access_predicate("d") + """
                RETURN c.id AS chunk_id, c.text AS text, score
                ORDER BY score DESC LIMIT $top_k
                """,
                embedding=embedding,
                tenant=tenant,
                top_k=top_k,
                valid_at=valid_at,
                transaction_at=transaction_at,
                **self._content_access_params(access_context),
            )
        fetch_k = max(top_k * 20, 100)
        return await self.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $fetch_k, $embedding)
            YIELD node AS c, score
            OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
            WHERE (c.tenant = $tenant)
              AND (d IS NULL OR coalesce(d.is_deleted, false) = false)
              AND ($valid_at IS NULL OR (
                  d IS NOT NULL
                  AND (d.valid_from IS NULL OR d.valid_from <= datetime($valid_at))
                  AND (d.valid_to IS NULL OR d.valid_to > datetime($valid_at))
              ))
              AND ($transaction_at IS NULL OR (
                  d IS NOT NULL
                  AND (coalesce(d.recorded_at, d.created_at) IS NULL
                       OR coalesce(d.recorded_at, d.created_at) <= datetime($transaction_at))
              ))
              AND NOT EXISTS {
                  MATCH (c)-[:MENTIONS]->(e:Entity)
                  WHERE e.quarantined = true
              }
              """ + document_access_predicate("d") + """
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            fetch_k=fetch_k,
            embedding=embedding,
            tenant=tenant,
            top_k=top_k,
            valid_at=valid_at,
            transaction_at=transaction_at,
            **self._content_access_params(access_context),
        )

    async def get_document_filenames(
        self, tenant: str = "default", access_context: AccessContext | None = None,
    ) -> list[str]:
        """List distinct document filenames for a tenant (for named-document
        matching against question text — see local_search's named-doc boost).
        """
        rows = await self.run(
            f"MATCH (d:Document) WHERE (d.tenant = $tenant) {document_access_predicate('d')} "
            "RETURN DISTINCT d.filename AS filename",
            tenant=tenant,
            **self._content_access_params(access_context),
        )
        return [r["filename"] for r in rows if r.get("filename")]

    async def get_chunk_filenames(
        self, chunk_ids: list[str], tenant: str = "default", access_context: AccessContext | None = None,
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
              AND (c.tenant = $tenant)
              {document_access_predicate('d')}
            RETURN c.id AS chunk_id, d.filename AS filename
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
            **self._content_access_params(access_context),
        )
        return {r["chunk_id"]: r["filename"] for r in rows if r.get("filename")}

    async def get_linked_document_chunks(
        self,
        seed_chunk_ids: list[str],
        *,
        top_k: int = 5,
        tenant: str = "default",
        query_embedding: list[float] | None = None,
        valid_at: str | None = None,
        transaction_at: str | None = None,
        access_context: AccessContext | None = None,
    ) -> list[dict]:
        """Follow explicit document links from seed evidence to authorised chunks.

        This is a bounded, one-hop topology expansion. It never invents a link
        from similarity, and it evaluates the source document, the ACL snapshot
        captured on ``LINKS_TO``, and the target document before returning any
        target text.
        """
        if not seed_chunk_ids:
            return []
        rows = await self.run(
            """
            UNWIND $seed_chunk_ids AS seed_id
            MATCH (seed:Chunk {id: seed_id, tenant: $tenant})-[:PART_OF]->(source:Document {tenant: $tenant})
                  -[link:LINKS_TO {tenant: $tenant}]->(target:Document {tenant: $tenant})
            MATCH (chunk:Chunk {tenant: $tenant})-[:PART_OF]->(target)
            WHERE coalesce(source.is_deleted, false) = false
              AND coalesce(target.is_deleted, false) = false
              AND ($valid_at IS NULL OR (
                (target.valid_from IS NULL OR target.valid_from <= datetime($valid_at))
                AND (target.valid_to IS NULL OR target.valid_to > datetime($valid_at))
              ))
              AND ($transaction_at IS NULL OR (
                (coalesce(source.recorded_at, source.created_at) IS NULL
                  OR coalesce(source.recorded_at, source.created_at) <= datetime($transaction_at))
                AND (coalesce(target.recorded_at, target.created_at) IS NULL
                  OR coalesce(target.recorded_at, target.created_at) <= datetime($transaction_at))
                AND (link.recorded_at IS NULL OR link.recorded_at <= datetime($transaction_at))
              ))
              """ + document_access_predicate("source") + document_access_predicate("target") + link_access_predicate("link") + """
            WITH chunk, source, target, link,
                 CASE WHEN $query_embedding IS NULL OR chunk.embedding IS NULL
                           OR size(chunk.embedding) = 0 THEN 0.0
                      ELSE vector.similarity.cosine(chunk.embedding, $query_embedding)
                 END AS semantic_score
            RETURN chunk.id AS chunk_id, chunk.text AS text,
                   semantic_score AS score, semantic_score AS path_score,
                   source.filename AS link_source, target.filename AS link_target,
                   link.anchor_text AS anchor_text, link.target_url AS target_url,
                   link.source_system AS source_system,
                   toString(link.observed_at) AS observed_at
            ORDER BY semantic_score DESC, link.observed_at DESC
            LIMIT $top_k
            """,
            seed_chunk_ids=list(dict.fromkeys(seed_chunk_ids)),
            top_k=top_k,
            tenant=tenant,
            query_embedding=query_embedding,
            valid_at=valid_at,
            transaction_at=transaction_at,
            **self._content_access_params(access_context),
        )
        return rows

    async def get_community_source_documents(
        self, community_ids: list[str], tenant: str = "default", limit_per_community: int = 3
    ) -> dict[str, list[str]]:
        """Map community_id -> source document filenames (no extension) for a
        set of communities, via Community <-[:MEMBER_OF]- Entity <-[:MENTIONS]-
        Chunk -[:PART_OF]-> Document.

        Global search answers from community *summaries*, which carry no
        per-fact document provenance forward — so answers synthesized in
        global mode previously returned zero citations, unconditionally,
        regardless of answer quality (ContextBuilder.build() only reads
        citations from local_results' chunks). This lets GlobalSearch attach
        a representative document set per community so global-mode and
        hybrid-mode answers built from community context can still be
        attributed to real source documents.
        """
        if not community_ids:
            return {}
        rows = await self.run(
            """
            UNWIND $community_ids AS cid
            MATCH (c:Community {id: cid, tenant: $tenant})
                  <-[:MEMBER_OF]-(e:Entity {tenant: $tenant})
                  <-[:MENTIONS]-(chunk:Chunk {tenant: $tenant})
                  -[:PART_OF]->(d:Document)
            WHERE coalesce(d.is_deleted, false) = false
            WITH cid, d.filename AS filename, count(*) AS mentions
            ORDER BY cid, mentions DESC
            WITH cid, collect(filename)[0..$limit] AS filenames
            RETURN cid AS community_id, filenames
            """,
            community_ids=community_ids,
            tenant=tenant,
            limit=limit_per_community,
        )
        return {
            r["community_id"]: [f.replace(".txt", "") for f in (r.get("filenames") or []) if f]
            for r in rows
        }

    async def get_best_chunk_for_document(
        self,
        filename: str,
        embedding: list[float],
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
        access_context: AccessContext | None = None,
    ) -> dict | None:
        """Best chunk (by cosine similarity to `embedding`) belonging to the
        document with this exact filename. Used by the named-document boost:
        when the question explicitly names a document, guarantee its most
        relevant chunk a seed slot even if it didn't survive fused-ranking.
        """
        rows = await self.run(
            """
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {{filename: $filename}})
            WHERE (c.tenant = $tenant)
              AND coalesce(d.is_deleted, false) = false
              AND c.embedding IS NOT NULL
              AND ($valid_at IS NULL OR (
                  (d.valid_from IS NULL OR d.valid_from <= datetime($valid_at))
                  AND (d.valid_to IS NULL OR d.valid_to > datetime($valid_at))
              ))
              AND ($transaction_at IS NULL OR (
                  coalesce(d.recorded_at, d.created_at) IS NULL
                  OR coalesce(d.recorded_at, d.created_at) <= datetime($transaction_at)
              ))
              {document_access_predicate('d')}
            RETURN c.id AS chunk_id, c.text AS text,
                   vector.similarity.cosine(c.embedding, $embedding) AS score
            ORDER BY score DESC
            LIMIT 1
            """,
            filename=filename,
            embedding=embedding,
            tenant=tenant,
            valid_at=valid_at,
            transaction_at=transaction_at,
            **self._content_access_params(access_context),
        )
        return rows[0] if rows else None

    async def vector_search_communities(
        self,
        embedding: list[float],
        top_k: int = 5,
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
    ) -> list[dict]:
        """ANN search over Community.embedding for global search.

        Over-fetches before tenant-filtering: db.index.vector.queryNodes
        returns the global top-k across all tenants, and this Neo4j version
        has no native pre-filter — so a tenant can be starved out of a
        small top-k by other tenants' higher-scoring nodes even when it has
        plenty of its own relevant communities (see tasks/lessons.md A146).
        fetch_k gives the tenant filter a much larger candidate pool before
        truncating to top_k.
        """
        if ((valid_at or transaction_at) and tenant != "default"
                and "community_summary_snapshot_embeddings"
                in self.__dict__.get("_filtered_vector_indexes", set())):
            fetch_k = max(top_k * 4, 20)
            return await self.run(
                """
                MATCH (c:CommunitySummarySnapshot)
                  SEARCH c IN (
                    VECTOR INDEX community_summary_snapshot_embeddings
                    FOR $embedding
                    WHERE c.tenant = $tenant
                    LIMIT $fetch_k
                  ) SCORE AS score
                WHERE ($valid_at IS NULL OR c.valid_from IS NULL OR c.valid_from <= datetime($valid_at))
                  AND ($valid_at IS NULL OR c.valid_to IS NULL OR c.valid_to > datetime($valid_at))
                  AND ($transaction_at IS NULL OR c.transaction_from IS NULL
                       OR c.transaction_from <= datetime($transaction_at))
                  AND ($transaction_at IS NULL OR c.transaction_to IS NULL
                       OR c.transaction_to > datetime($transaction_at))
                RETURN c.community_id AS community_id, c.summary AS summary,
                       c.level AS level, score, c.id AS summary_snapshot_id,
                       c.content_hash AS summary_content_hash
                ORDER BY score DESC LIMIT $top_k
                """,
                fetch_k=fetch_k,
                embedding=embedding,
                tenant=tenant,
                top_k=top_k,
                valid_at=valid_at,
                transaction_at=transaction_at,
            )
        if valid_at or transaction_at:
            fetch_k = max(top_k * 20, 100)
            return await self.run(
                """
                CALL db.index.vector.queryNodes(
                  'community_summary_snapshot_embeddings', $fetch_k, $embedding
                )
                YIELD node AS c, score
                WHERE (c.tenant = $tenant)
                  AND ($valid_at IS NULL OR c.valid_from IS NULL OR c.valid_from <= datetime($valid_at))
                  AND ($valid_at IS NULL OR c.valid_to IS NULL OR c.valid_to > datetime($valid_at))
                  AND ($transaction_at IS NULL OR c.transaction_from IS NULL
                       OR c.transaction_from <= datetime($transaction_at))
                  AND ($transaction_at IS NULL OR c.transaction_to IS NULL
                       OR c.transaction_to > datetime($transaction_at))
                RETURN c.community_id AS community_id, c.summary AS summary,
                       c.level AS level, score, c.id AS summary_snapshot_id,
                       c.content_hash AS summary_content_hash
                ORDER BY score DESC LIMIT $top_k
                """,
                fetch_k=fetch_k,
                embedding=embedding,
                tenant=tenant,
                top_k=top_k,
                valid_at=valid_at,
                transaction_at=transaction_at,
            )
        if self.__dict__.get("_filtered_vector_search", False) and tenant != "default":
            return await self.run(
                """
                MATCH (c:Community)
                  SEARCH c IN (
                    VECTOR INDEX community_embeddings
                    FOR $embedding
                    WHERE c.tenant = $tenant
                    LIMIT $top_k
                  ) SCORE AS score
                RETURN c.id AS community_id, c.summary AS summary,
                       c.level AS level, score
                ORDER BY score DESC LIMIT $top_k
                """,
                embedding=embedding,
                tenant=tenant,
                top_k=top_k,
            )
        fetch_k = max(top_k * 20, 100)
        return await self.run(
            """
            CALL db.index.vector.queryNodes('community_embeddings', $fetch_k, $embedding)
            YIELD node AS c, score
            WHERE (c.tenant = $tenant)
              AND ($valid_at IS NULL OR (
                  c.valid_from IS NULL OR c.valid_from <= datetime($valid_at)
              ))
              AND ($valid_at IS NULL OR (
                  c.valid_to IS NULL OR c.valid_to > datetime($valid_at)
              ))
              AND ($transaction_at IS NULL OR (
                  c.recorded_at IS NULL OR c.recorded_at <= datetime($transaction_at)
              ))
            RETURN c.id AS community_id, c.summary AS summary, c.level AS level, score
            ORDER BY score DESC
            LIMIT $top_k
            """,
            fetch_k=fetch_k,
            embedding=embedding,
            tenant=tenant,
            top_k=top_k,
            valid_at=valid_at,
            transaction_at=transaction_at,
        )

    async def get_entity_neighbors(
        self,
        chunk_ids: list[str],
        as_of: str | None = None,
        tenant: str = "default",
        transaction_at: str | None = None,
    ) -> list[dict]:
        """Expand retrieved chunks to their entity neighbors (1-hop).
        Excludes quarantined entities. Optionally filters edges by valid_to.
        """
        temporal_filter = (
            "AND (r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of)"
            if as_of else ""
        )
        transaction_filter = (
            "AND (r.recorded_at IS NULL OR r.recorded_at <= datetime($transaction_at))"
            if transaction_at else ""
        )
        return await self.run(
            f"""
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {{id: cid}})-[:MENTIONS]->(e:Entity)
            WHERE coalesce(e.quarantined, false) = false
            OPTIONAL MATCH (e)-[r:RELATES_TO {{tenant: $tenant}}]-(neighbor:Entity {{tenant: $tenant}})
            WHERE coalesce(neighbor.quarantined, false) = false {temporal_filter} {transaction_filter}
            RETURN e.name AS entity, e.type AS type, e.description AS description,
                   collect(DISTINCT neighbor.name) AS neighbors
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
            **({"as_of": as_of} if as_of else {}),
            **({"transaction_at": transaction_at} if transaction_at else {}),
        )

    async def get_multihop_chunks(
        self,
        chunk_ids: list[str],
        hops: int = 2,
        as_of: str | None = None,
        tenant: str = "default",
        transaction_at: str | None = None,
        query_embedding: list[float] | None = None,
        semantic_weight: float = 0.0,
        per_seed_cap: int = 200,
        total_cap: int = 500,
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
        # These values are interpolated into Cypher's variable-length path and
        # LIMIT clauses, so validate and bound them before interpolation.  The
        # default limits preserve the legacy query shape exactly.
        hops = min(max(int(hops), 1), 8)
        per_seed_cap = min(max(int(per_seed_cap), 1), 1_000)
        total_cap = min(max(int(total_cap), 1), 5_000)
        temporal_filter = (
            "AND ALL(r IN relationships(path) WHERE "
            "(r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of))"
            if as_of else ""
        )
        transaction_filter = (
            "AND ALL(r IN relationships(path) WHERE "
            "r.recorded_at IS NULL OR r.recorded_at <= datetime($transaction_at))"
            if transaction_at else ""
        )
        # Unconditional: this was previously skipped entirely for tenant
        # "default", which turned the multi-hop expansion into a
        # read-every-tenant path traversal.
        tenant_filter = (
            "AND ALL(r IN relationships(path) WHERE r.tenant = $tenant)"
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
                WHERE coalesce(neighbor.quarantined, false) = false {temporal_filter} {transaction_filter} {tenant_filter}
                  AND ALL(n IN nodes(path) WHERE coalesce(n.quarantined, false) = false)
                MATCH (neighbor_chunk:Chunk)-[:MENTIONS]->(neighbor)
                WHERE NOT neighbor_chunk.id IN $chunk_ids
                  AND (neighbor_chunk.tenant = $tenant)
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
            per_seed_cap=per_seed_cap,
            total_cap=total_cap,
            **({"as_of": as_of} if as_of else {}),
            **({"transaction_at": transaction_at} if transaction_at else {}),
            **({"query_emb": query_embedding, "sem_w": float(semantic_weight)}
               if use_semantic else {}),
        )
        # Normalise to list[dict] with score field for downstream compatibility
        for row in results:
            row["score"] = float(row.get("path_score") or 0.0)
        return results

    async def bm25_search_chunks(
        self,
        query: str,
        top_k: int = 10,
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
        access_context: AccessContext | None = None,
    ) -> list[dict]:
        """BM25 fulltext search over Chunk.text using Neo4j fulltext index.
        Filters by tenant and excludes quarantined entity chunks.
        """
        return await self.run(
            """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
            YIELD node AS c, score
            OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
            WHERE (c.tenant = $tenant)
              AND (d IS NULL OR coalesce(d.is_deleted, false) = false)
              AND ($valid_at IS NULL OR (
                  d IS NOT NULL
                  AND (d.valid_from IS NULL OR d.valid_from <= datetime($valid_at))
                  AND (d.valid_to IS NULL OR d.valid_to > datetime($valid_at))
              ))
              AND ($transaction_at IS NULL OR (
                  d IS NOT NULL
                  AND (coalesce(d.recorded_at, d.created_at) IS NULL
                       OR coalesce(d.recorded_at, d.created_at) <= datetime($transaction_at))
              ))
              AND NOT EXISTS {
                  MATCH (c)-[:MENTIONS]->(e:Entity)
                  WHERE e.quarantined = true
              }
              """ + document_access_predicate("d") + """
            RETURN c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
            tenant=tenant,
            valid_at=valid_at,
            transaction_at=transaction_at,
            **self._content_access_params(access_context),
        )

    async def bm25_search_entities(
        self,
        query: str,
        top_k: int = 10,
        tenant: str = "default",
        valid_at: str | None = None,
        transaction_at: str | None = None,
        access_context: AccessContext | None = None,
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
            OPTIONAL MATCH (c)-[:PART_OF]->(d:Document)
            WHERE (c.tenant = $tenant)
              AND (d IS NULL OR coalesce(d.is_deleted, false) = false)
              AND ($valid_at IS NULL OR (
                  d IS NOT NULL
                  AND (d.valid_from IS NULL OR d.valid_from <= datetime($valid_at))
                  AND (d.valid_to IS NULL OR d.valid_to > datetime($valid_at))
              ))
              AND ($transaction_at IS NULL OR (
                  d IS NOT NULL
                  AND (coalesce(d.recorded_at, d.created_at) IS NULL
                       OR coalesce(d.recorded_at, d.created_at) <= datetime($transaction_at))
              ))
              {document_access_predicate('d')}
            RETURN DISTINCT c.id AS chunk_id, c.text AS text, score
            ORDER BY score DESC
            LIMIT $k
            """,
            query=query,
            k=top_k,
            tenant=tenant,
            valid_at=valid_at,
            transaction_at=transaction_at,
            **self._content_access_params(access_context),
        )

    async def get_chunk_entity_embeddings(
        self, chunk_ids: list[str], tenant: str
    ) -> list[dict]:
        """Return entity embeddings for all entities mentioned by the given chunks.

        Used by GNNScorer to build the node-feature matrix H.
        Only returns entities that actually have a stored embedding.
        Excludes quarantined entities.

        Two-phase, cache-backed (see graphrag/graph/embedding_cache.py):
        live profiling found this call cost ~21ms per distinct entity,
        dominated by PackStream deserialization of 3072-dim embeddings over
        Bolt (confirmed via PROFILE: dbHits are trivial; the same query
        minus e.embedding runs in a fraction of the time). A single query
        typically touches a large fraction of a tenant's entities, so
        caching by (tenant, name, type) recovers most of that cost across
        queries in a long-running process.

        ``tenant`` is required (not defaulted) — caching by (name, type)
        alone would silently conflate two tenants' entities that happen to
        share a name and type but have different embeddings.
        """
        from graphrag.graph.embedding_cache import get_embedding_cache
        cache = get_embedding_cache()

        # Phase 1: discover which entities these chunks mention, without
        # fetching the embedding itself — this step can't be skipped or
        # cached, since it's what tells us which entities exist for *this*
        # chunk set. Cheap on its own (confirmed via PROFILE).
        rows = await self.run(
            """
            UNWIND $chunk_ids AS cid
            MATCH (c:Chunk {id: cid})-[:MENTIONS]->(e:Entity)
            WHERE (c.tenant = $tenant)
              AND e.embedding IS NOT NULL AND size(e.embedding) > 0
              AND coalesce(e.quarantined, false) = false
            RETURN cid          AS chunk_id,
                   e.name       AS entity_name,
                   e.type       AS entity_type,
                   COUNT { (e)-[:RELATES_TO]-() } AS degree
            """,
            chunk_ids=chunk_ids,
            tenant=tenant,
        )

    async def merge_contextual_entity_representations(
        self,
        chunk_id: str,
        entities: list[Entity],
        *,
        source_system: str,
        source_doc_id: str,
        tenant: str = "default",
    ) -> int:
        """Attach source-system representations to canonical entities.

        The canonical entity remains tenant scoped for approved aliases, but
        every assertion is written through a ``SystemRepresentation`` keyed by
        source system. Thus ``Customer`` mentioned in CRM and ERP has separate
        representations and assertion paths even when an explicit resolver has
        intentionally mapped both surface forms to one canonical entity.
        """
        if not entities:
            return 0
        rows = [
            {
                "canonical_name": entity.canonical_name or entity.name,
                "canonical_type": entity.canonical_type or entity.type,
                "raw_name": entity.name,
                "raw_type": entity.type,
            }
            for entity in entities
        ]
        result = await self.run(
            """
            MATCH (chunk:Chunk {id: $chunk_id, tenant: $tenant})
            UNWIND $rows AS row
            MATCH (entity:Entity {
                name: row.canonical_name, type: row.canonical_type, tenant: $tenant
            })
            MERGE (representation:SystemRepresentation {
                tenant: $tenant,
                source_system: $source_system,
                canonical_name: row.canonical_name,
                canonical_type: row.canonical_type
            })
            ON CREATE SET representation.id = randomUUID(),
                          representation.created_at = datetime(),
                          representation.recorded_at = datetime()
            SET representation.raw_names = CASE
                  WHEN row.raw_name IN coalesce(representation.raw_names, [])
                  THEN representation.raw_names
                  ELSE coalesce(representation.raw_names, []) + row.raw_name
                END,
                representation.updated_at = datetime()
            MERGE (entity)-[:HAS_SYSTEM_REPRESENTATION {tenant: $tenant, source_system: $source_system}]->(representation)
            MERGE (assertion:ContextualAssertion {
                tenant: $tenant, chunk_id: $chunk_id,
                canonical_name: row.canonical_name, canonical_type: row.canonical_type,
                source_system: $source_system
            })
            ON CREATE SET assertion.id = randomUUID(), assertion.created_at = datetime(),
                          assertion.source_doc_id = $source_doc_id
            SET assertion.raw_name = row.raw_name, assertion.raw_type = row.raw_type,
                assertion.updated_at = datetime()
            MERGE (chunk)-[:ASSERTS_IN_CONTEXT {tenant: $tenant}]->(assertion)
            MERGE (assertion)-[:ASSERTS_REPRESENTATION {tenant: $tenant}]->(representation)
            RETURN count(assertion) AS assertions
            """,
            chunk_id=chunk_id,
            rows=rows,
            source_system=source_system,
            source_doc_id=source_doc_id,
            tenant=tenant,
        )
        return int(result[0].get("assertions", 0)) if result else 0
        if not rows:
            return []

        # Phase 2: split into cache hits (skip Neo4j entirely) and misses
        # (need one batched fetch for just the missing entities).
        miss_pairs: list[tuple[str, str]] = []
        seen_misses: set[tuple[str, str]] = set()
        for r in rows:
            key = (r["entity_name"], r["entity_type"])
            if cache.get(tenant, *key) is None and key not in seen_misses:
                seen_misses.add(key)
                miss_pairs.append({"name": key[0], "type": key[1]})

        if miss_pairs:
            fetched = await self.run(
                """
                UNWIND $pairs AS pair
                MATCH (e:Entity {name: pair.name, type: pair.type, tenant: $tenant})
                WHERE e.embedding IS NOT NULL AND size(e.embedding) > 0
                RETURN e.name AS entity_name, e.type AS entity_type, e.embedding AS embedding
                """,
                pairs=miss_pairs, tenant=tenant,
            )
            for f in fetched:
                cache.set(tenant, f["entity_name"], f["entity_type"], f["embedding"])

        # Merge: attach each row's embedding from the (now fully warmed for
        # this batch) cache. A row whose entity has no embedding after
        # phase 2 (e.g. quarantined/deleted between phase 1 and phase 2) is
        # dropped, matching the old query's e.embedding IS NOT NULL filter.
        results = []
        for r in rows:
            emb = cache.get(tenant, r["entity_name"], r["entity_type"])
            if emb is None:
                continue
            results.append({
                "chunk_id":    r["chunk_id"],
                "entity_name": r["entity_name"],
                "entity_type": r["entity_type"],
                "embedding":   emb,
                "degree":      r["degree"],
            })
        return results

    async def get_entity_relations_subgraph(
        self,
        entities: list[dict],
        as_of: str | None = None,
        tenant: str = "default",
        transaction_at: str | None = None,
    ) -> list[dict]:
        """Return RELATES_TO edges between a set of entities.

        ``entities`` is a list of ``{"name": str, "type": str}`` dicts.
        Using (name, type) pairs rather than names alone avoids ambiguous
        matches when a tenant has two entities with the same name but
        different types (e.g. "Apple" as ORG vs. PRODUCT).

        Used by GNNScorer to build the adjacency matrix A, and by
        ContextBuilder to surface labeled graph facts (e.g. a transitively
        inferred SUPERSEDES edge) in the synthesis prompt — see INF-01/CON-02
        in evals/golden_set.json: chunk text alone states only pairwise
        supersession, and without the relation label + inferred/asserted
        distinction here, the LLM has no textual anchor to affirm a
        transitive fact it's told to answer "ONLY" from context.
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
        transaction_filter = (
            "AND (r.recorded_at IS NULL OR r.recorded_at <= datetime($transaction_at))"
            if transaction_at else ""
        )
        # Build a set-membership key of the form "name:type" for the target
        # side filter so both dimensions are checked without a subquery.
        return await self.run(
            f"""
            UNWIND $entities AS pair
            MATCH (s:Entity {{name: pair.name, type: pair.type, tenant: $tenant}})
                  -[r:RELATES_TO]->
                  (t:Entity {{tenant: $tenant}})
            WHERE (t.name + ':' + t.type) IN $entity_keys {temporal_filter} {transaction_filter}
              AND coalesce(s.quarantined, false) = false
              AND coalesce(t.quarantined, false) = false
            RETURN s.name                             AS src,
                   s.type                             AS src_type,
                   t.name                             AS tgt,
                   t.type                             AS tgt_type,
                   r.relation                         AS relation,
                   r.weight                           AS weight,
                   coalesce(r.confidence, 1.0)        AS confidence,
                   r.extracted_at                     AS extracted_at,
                   r.source_doc_id                    AS source_doc_id,
                   coalesce(r.source_type, 'asserted') AS source_type,
                   r.inferred_by                      AS inferred_by
            """,
            entities=entities,
            entity_keys=[f"{e['name']}:{e['type']}" for e in entities],
            tenant=tenant,
            **({"as_of": as_of} if as_of else {}),
            **({"transaction_at": transaction_at} if transaction_at else {}),
        )

    async def get_relations_for_entity(
        self,
        name: str,
        type: str,
        tenant: str = "default",
        as_of: str | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Return RELATES_TO edges (either direction) touching a single entity.

        Not to be confused with ``get_entity_neighbors`` (chunk_ids -> 1-hop
        entity expansion, used by LocalSearch's retrieval path) or
        ``get_entity_relations_subgraph`` (edges *between* members of an
        explicit set — both endpoints must be in the passed-in list, so a
        1-item list only matches self-loops). This answers "what is this one
        named entity connected to" — the shape a single-entity lookup needs.
        Excludes quarantined entities on either side.
        """
        temporal_filter = (
            "AND (r.valid_from IS NULL OR r.valid_from <= $as_of) "
            "AND (r.valid_to IS NULL OR r.valid_to > $as_of)"
            if as_of else ""
        )
        return await self.run(
            f"""
            MATCH (e:Entity {{name: $name, type: $type, tenant: $tenant}})
                  -[r:RELATES_TO]-(other:Entity {{tenant: $tenant}})
            WHERE coalesce(e.quarantined, false) = false
              AND coalesce(other.quarantined, false) = false
              {temporal_filter}
            RETURN other.name AS name, other.type AS type,
                   r.weight AS weight,
                   coalesce(r.confidence, 1.0) AS confidence,
                   r.extracted_at AS extracted_at,
                   r.source_doc_id AS source_doc_id,
                   CASE WHEN startNode(r) = e THEN 'outgoing' ELSE 'incoming' END AS direction
            ORDER BY confidence DESC
            LIMIT $limit
            """,
            name=name, type=type, tenant=tenant, limit=limit,
            **({"as_of": as_of} if as_of else {}),
        )

    async def get_all_entities(self, tenant: str = "default") -> list[dict]:
        return await self.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE (c.tenant = $tenant)
              AND coalesce(e.quarantined, false) = false
            RETURN DISTINCT e.id AS id, e.name AS name, e.type AS type
            """,
            tenant=tenant,
        )

    async def get_all_relations(self, tenant: str = "default") -> list[dict]:
        return await self.run(
            """
            MATCH (s:Entity)-[r:RELATES_TO {tenant: $tenant}]->(t:Entity)
            WHERE coalesce(s.quarantined, false) = false
              AND coalesce(t.quarantined, false) = false
            RETURN s.id AS source_id, t.id AS target_id, r.relation AS relation,
                   coalesce(r.weight, r.confidence, 1.0) AS weight
            """,
            tenant=tenant,
        )

    # ── PageRank centrality (GDS) ────────────────────────────────────────────────

    async def run_pagerank(
        self,
        tenant: str = "default",
        damping_factor: float = 0.85,
        max_iterations: int = 20,
    ) -> list[dict]:
        """Project the tenant's Entity/RELATES_TO subgraph in-memory via GDS,
        run PageRank, drop the projection, return entities sorted by score desc.

        Uses a Cypher projection (not native gds.graph.project) so relationship
        weight can fall back to 1.0 when confidence is null — same null-handling
        as CommunityBuilder._build_networkx_graph's weight resolution.
        """
        graph_name = f"pagerank_{tenant}"
        await self.run("CALL gds.graph.drop($name, false)", name=graph_name)  # idempotent cleanup
        await self.run(
            """
            CALL gds.graph.project.cypher(
              $name,
              'MATCH (e:Entity {tenant: $tenant}) WHERE coalesce(e.quarantined,false)=false RETURN id(e) AS id',
              'MATCH (a:Entity {tenant: $tenant})-[r:RELATES_TO {tenant: $tenant}]->(b:Entity {tenant: $tenant})
               RETURN id(a) AS source, id(b) AS target, coalesce(r.weight, r.confidence, 1.0) AS weight',
              {parameters: {tenant: $tenant}}
            )
            """,
            name=graph_name,
            tenant=tenant,
        )
        try:
            rows = await self.run(
                """
                CALL gds.pageRank.stream($name, {
                  dampingFactor: $damping, maxIterations: $iters,
                  relationshipWeightProperty: 'weight'
                })
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).id AS entity_id,
                       gds.util.asNode(nodeId).name AS name,
                       gds.util.asNode(nodeId).type AS type,
                       score
                ORDER BY score DESC
                """,
                name=graph_name,
                damping=damping_factor,
                iters=max_iterations,
            )
        finally:
            await self.run("CALL gds.graph.drop($name, false)", name=graph_name)
        return rows

    async def write_pagerank_scores(self, tenant: str, scores: list[dict]) -> None:
        """Persist score onto each Entity node (UNWIND-batched, per A131-A132 pattern)."""
        await self.run(
            """
            UNWIND $rows AS row
            MATCH (e:Entity {id: row.entity_id, tenant: $tenant})
            SET e.pagerank = row.score, e.pagerank_computed_at = datetime()
            """,
            tenant=tenant,
            rows=scores,
        )

    async def get_top_entities_by_pagerank(
        self, tenant: str = "default", top_k: int = 20
    ) -> list[dict]:
        return await self.run(
            """
            MATCH (e:Entity {tenant: $tenant})
            WHERE e.pagerank IS NOT NULL
            RETURN e.id AS entity_id, e.name AS name, e.type AS type, e.pagerank AS score
            ORDER BY e.pagerank DESC LIMIT $top_k
            """,
            tenant=tenant,
            top_k=top_k,
        )

    async def get_pagerank_by_entity_names(
        self, entity_names: list[str], tenant: str = "default"
    ) -> dict[str, float]:
        """Return {entity_name: pagerank} for the given names, tenant-scoped.

        Used only by the low-confidence-retrieval PageRank tiebreak (see
        local_search.py) — a small, targeted lookup fired on a rare path, not
        part of the per-query hot path. Entities with no computed pagerank
        (coverage is currently partial/stale across tenants — see
        tasks/lessons.md) are simply absent from the returned dict; callers
        must treat a missing key as "no signal," not zero.
        """
        if not entity_names:
            return {}
        rows = await self.run(
            """
            UNWIND $names AS name
            MATCH (e:Entity {name: name, tenant: $tenant})
            WHERE e.pagerank IS NOT NULL
            RETURN e.name AS name, e.pagerank AS score
            """,
            names=entity_names,
            tenant=tenant,
        )
        return {r["name"]: float(r["score"]) for r in rows}


_client: Neo4jClient | None = None


def get_neo4j() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


async def close_neo4j() -> None:
    """Close and reset the process singleton when it was initialized."""
    global _client
    client, _client = _client, None
    if client is not None:
        await client.close()
