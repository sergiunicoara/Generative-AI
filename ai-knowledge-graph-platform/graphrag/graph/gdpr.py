"""GDPR / right-to-be-forgotten — soft delete with cascade and audit trail.

Problem solved
--------------
`quarantine` hides an entity from retrieval but keeps all data in the graph.
That is correct for quality-control purposes (suspected bad data).
It is not correct for a GDPR erasure request:
  - The entity node must be deleted.
  - All RELATES_TO / NEGATIVE_RELATES_TO / MENTIONS edges must be removed.
  - The entity name must be redacted from chunk text and document metadata.
  - The deletion must be auditable: who requested it, when, what was removed.

Architecture
------------
GDPRService implements two operations:

forget_entity(name, type, tenant)
  1. Record deletion intent in DeletionAudit node.
  2. Collect stats (edge count, chunk mention count) for the audit record.
  3. Redact the entity name from all Chunk.text fields that mention it
     (replace with [REDACTED]).
  4. Delete MENTIONS, RELATES_TO, NEGATIVE_RELATES_TO edges.
  5. Delete the Entity node itself.
  6. Update DeletionAudit with completion timestamp and counts.

forget_document(doc_id, tenant)
  1. Find all entities introduced exclusively by this document (no other
     source_doc_id on their MENTIONS chain).
  2. Call forget_entity for each exclusive entity.
  3. Delete all Chunk nodes for the document.
  4. Delete the Document node.
  5. Record in DeletionAudit.

IMPORTANT: Cypher DELETE is permanent.  This is intentional for GDPR.
The DeletionAudit node records what was removed but not the original values.
ChangeLog entries for the deleted entity are also deleted (right-to-be-forgotten
requires the audit trail itself to be scrubbed for the subject's data).
"""

from __future__ import annotations

import re
from uuid import uuid4

import structlog

from graphrag.core.tenancy import require_tenant
from graphrag.graph.corpus_revision import CorpusMutation

log = structlog.get_logger(__name__)


class GDPRService:
    """
    GDPR-compliant entity and document erasure.

    Usage::

        svc = GDPRService(neo4j_client)

        # Erase a specific entity (e.g. an individual's record)
        report = await svc.forget_entity(
            entity_name="John Smith",
            entity_type="PERSON",
            tenant="acme",
            requested_by="dpo@acme.com",
            request_id="SAR-2024-001",
        )

        # Erase all data from a specific document
        report = await svc.forget_document(
            doc_id="doc_abc",
            tenant="acme",
            requested_by="dpo@acme.com",
        )

        # List all erasure records
        records = await svc.deletion_audit_log(tenant="acme")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Public entry points ────────────────────────────────────────────────────
    #
    # Erasure changes what retrieval can return, so it is a corpus mutation:
    # it must advance the corpus revision (every cached answer is keyed on it,
    # so older entries become unreachable) and hold cache reads off while it
    # runs. Without this, a question answered before the erasure kept being
    # served -- raw chunk text included -- until the cache TTL expired.
    # Wrapped here, not in the routes, so every caller gets it
    # (audit-2026-10-01.md, H3).

    async def forget_entity(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str,
        requested_by: str = "dpo",
        request_id: str = "",
    ) -> dict:
        """Erase all data for a named entity. See `_forget_entity` for the steps."""
        tenant = require_tenant(tenant)
        async with CorpusMutation(self._neo4j, tenant, reason="gdpr_erasure"):
            report = await self._forget_entity(
                entity_name, entity_type, tenant, requested_by, request_id,
            )
        await self._evict_cached_answers(tenant)
        return report

    async def forget_document(
        self,
        doc_id: str,
        tenant: str,
        requested_by: str = "dpo",
        request_id: str = "",
    ) -> dict:
        """Erase data sourced exclusively from a document. See `_forget_document`."""
        tenant = require_tenant(tenant)
        async with CorpusMutation(self._neo4j, tenant, reason="gdpr_erasure"):
            report = await self._forget_document(doc_id, tenant, requested_by, request_id)
        await self._evict_cached_answers(tenant)
        return report

    async def _evict_cached_answers(self, tenant: str) -> None:
        """Delete the tenant's cached answers rather than leave them to expire.

        The revision bump already makes them unreachable; this removes the
        stored text itself (a cached QueryResult carries the raw chunk
        contexts). Best-effort: a cache outage must not fail an erasure that
        has already happened, but it is logged because stored-but-unreachable
        personal data is still stored.
        """
        try:
            from graphrag.retrieval.query_cache import get_query_cache

            cache = await get_query_cache()
            await cache.flush_tenant(tenant=tenant)
        except Exception as exc:  # noqa: BLE001
            log.warning("gdpr.cache_eviction_failed", tenant=tenant, error=str(exc)[:200])

    # ── Entity erasure ─────────────────────────────────────────────────────────

    async def _forget_entity(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str,
        requested_by: str = "dpo",
        request_id: str = "",
    ) -> dict:
        """
        Erase all data for a named entity (GDPR right-to-be-forgotten).

        Steps:
        1. Record erasure intent.
        2. Redact entity name from chunk text.
        3. Remove all edges (RELATES_TO, NEGATIVE_RELATES_TO, MENTIONS).
        4. Remove audit/change-log entries for this entity.
        5. Delete the entity node.
        6. Complete audit record.

        Returns a report dict with counts of what was removed.

        `tenant` is required (no default) -- an erasure request is the most
        consequential write in this codebase; a caller that omits it should
        get a loud error, not a silent fall-through to some default tenant's
        data. See graphrag.core.tenancy.require_tenant.
        """
        tenant = require_tenant(tenant)
        audit_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (a:DeletionAudit {
                id:             $id,
                subject_name:   $name,
                subject_type:   $type,
                tenant:         $tenant,
                requested_by:   $requested_by,
                request_id:     $request_id,
                status:         'in_progress',
                requested_at:   datetime()
            })
            """,
            id=audit_id,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
            requested_by=requested_by,
            request_id=request_id,
        )

        # Count what we're about to remove
        stats = await self._count_entity_data(entity_name, entity_type, tenant)

        # Step 1: Redact entity name from chunk text
        redacted_chunks = await self._redact_mentions_in_chunks(
            entity_name, entity_type, tenant
        )

        # Step 2: Remove MENTIONS edges
        await self._neo4j.run(
            """
            MATCH (c:Chunk)-[m:MENTIONS]->(e:Entity {name: $name, type: $type, tenant: $tenant})
            DELETE m
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )

        # Step 3: Remove RELATES_TO and NEGATIVE_RELATES_TO edges (both directions)
        for edge_type in ("RELATES_TO", "NEGATIVE_RELATES_TO"):
            await self._neo4j.run(
                f"""
                MATCH (e:Entity {{name: $name, type: $type, tenant: $tenant}})
                      -[r:{edge_type}]-()
                DELETE r
                """,
                name=entity_name,
                type=entity_type,
                tenant=tenant,
            )
            await self._neo4j.run(
                f"""
                MATCH ()-[r:{edge_type}]->(e:Entity {{name: $name, type: $type, tenant: $tenant}})
                DELETE r
                """,
                name=entity_name,
                type=entity_type,
                tenant=tenant,
            )

        # Step 4: Remove Statement nodes (reified relations)
        await self._neo4j.run(
            """
            MATCH (stmt:Statement {tenant: $tenant})
            WHERE stmt.src_name = $name OR stmt.tgt_name = $name
            DETACH DELETE stmt
            """,
            name=entity_name,
            tenant=tenant,
        )

        # Step 5: Remove ChangeLog entries for this entity.
        # Was `MATCH (cl:ChangeLog {target_label:'Entity'}) WHERE cl.target_id
        # = $name` -- two bugs: no tenant filter (deleted ANY tenant's
        # ChangeLog row with a matching target_id), and target_id is never
        # equal to $name in the first place -- audit_trail.py stores
        # `target_id: e.id` (the entity's internal id), not its name, so
        # this predicate never matched anything it was meant to. Traversing
        # the HAS_CHANGE edge audit_trail.py actually creates fixes both:
        # it's tenant-scoped via the entity match, and doesn't depend on a
        # field that was never populated the way this assumed.
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
                  -[:HAS_CHANGE]->(cl:ChangeLog {tenant: $tenant})
            DELETE cl
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )

        # Step 6: Delete the entity node
        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            DETACH DELETE e
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )

        # Complete audit record
        await self._neo4j.run(
            """
            MATCH (a:DeletionAudit {id: $id})
            SET a.status             = 'complete',
                a.completed_at       = datetime(),
                a.edges_removed      = $edges,
                a.chunks_redacted    = $chunks
            """,
            id=audit_id,
            edges=stats.get("total_edges", 0),
            chunks=redacted_chunks,
        )

        report = {
            "audit_id":       audit_id,
            "entity_name":    entity_name,
            "entity_type":    entity_type,
            "tenant":         tenant,
            "edges_removed":  stats.get("total_edges", 0),
            "chunks_redacted": redacted_chunks,
            "status":         "complete",
        }
        log.info("gdpr.entity_forgotten", **{k: v for k, v in report.items() if k != "audit_id"})
        return report

    # ── Document erasure ───────────────────────────────────────────────────────

    async def _forget_document(
        self,
        doc_id: str,
        tenant: str,
        requested_by: str = "dpo",
        request_id: str = "",
    ) -> dict:
        """
        Erase all data sourced exclusively from a specific document.

        Entities that also appear in other documents are NOT deleted —
        only their association with this document is removed.
        Entities whose sole evidence comes from this document are fully erased.

        `tenant` is required (no default) -- see forget_entity's docstring.
        """
        tenant = require_tenant(tenant)
        audit_id = str(uuid4())

        # Entities exclusively from this document. `other:Chunk` in the
        # NOT EXISTS subquery previously had no tenant filter, so a
        # same-named/id'd chunk in a DIFFERENT tenant could make an
        # otherwise-exclusive entity look non-exclusive -- an under-deletion
        # (fails safe, doesn't leak data), but still wrong: a genuine GDPR
        # erasure could silently fail to fully erase an entity because of
        # data it can't even see.
        exclusive_rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})-[:MENTIONS]->(e:Entity {tenant: $tenant})
            WHERE NOT EXISTS {
                MATCH (other:Chunk {tenant: $tenant})-[:MENTIONS]->(e)
                WHERE other.document_id <> $doc_id
            }
            RETURN DISTINCT e.name AS name, e.type AS type
            """,
            doc_id=doc_id,
            tenant=tenant,
        )

        erased_entities = []
        for row in exclusive_rows:
            await self.forget_entity(
                entity_name=row["name"],
                entity_type=row["type"],
                tenant=tenant,
                requested_by=requested_by,
                request_id=request_id,
            )
            erased_entities.append({"name": row["name"], "type": row["type"]})

        # Delete all chunks for this document. Was matched by document_id
        # alone -- a document_id collision (or reuse) across tenants would
        # DETACH DELETE another tenant's chunks entirely.
        chunk_rows = await self._neo4j.run(
            "MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant}) DETACH DELETE c RETURN count(c) AS n",
            doc_id=doc_id,
            tenant=tenant,
        )
        chunks_deleted = chunk_rows[0].get("n", 0) if chunk_rows else 0

        # Delete the document node itself. Same class of bug -- id-only match.
        await self._neo4j.run(
            "MATCH (d:Document {id: $doc_id, tenant: $tenant}) DETACH DELETE d",
            doc_id=doc_id,
            tenant=tenant,
        )

        report = {
            "audit_id":        audit_id,
            "doc_id":          doc_id,
            "tenant":          tenant,
            "entities_erased": len(erased_entities),
            "chunks_deleted":  chunks_deleted,
            "erased_entities": erased_entities,
            "status":          "complete",
        }
        log.info("gdpr.document_forgotten", doc_id=doc_id, tenant=tenant,
                 entities=len(erased_entities))
        return report

    # ── Audit log ──────────────────────────────────────────────────────────────

    async def deletion_audit_log(
        self,
        tenant: str,
        limit: int = 100,
    ) -> list[dict]:
        """Return all erasure records for a tenant, newest first.

        `tenant` required for consistency with forget_entity/forget_document
        -- this query was already correctly scoped, but a default here would
        leave one of three public methods on this service silently exempt
        from the "no unscoped GDPR operation" rule the other two now enforce.
        """
        tenant = require_tenant(tenant)
        return await self._neo4j.run(
            """
            MATCH (a:DeletionAudit)
            WHERE (a.tenant = $tenant)
            RETURN a.id             AS audit_id,
                   a.subject_name   AS subject_name,
                   a.subject_type   AS subject_type,
                   a.requested_by   AS requested_by,
                   a.request_id     AS request_id,
                   a.status         AS status,
                   a.edges_removed  AS edges_removed,
                   a.chunks_redacted AS chunks_redacted,
                   a.requested_at   AS requested_at,
                   a.completed_at   AS completed_at
            ORDER BY a.requested_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _count_entity_data(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str,
    ) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            OPTIONAL MATCH (e)-[r1:RELATES_TO]-()
            OPTIONAL MATCH ()-[r2:RELATES_TO]->(e)
            OPTIONAL MATCH (e)-[n1:NEGATIVE_RELATES_TO]-()
            OPTIONAL MATCH ()-[n2:NEGATIVE_RELATES_TO]->(e)
            OPTIONAL MATCH (c)-[:MENTIONS]->(e)
            RETURN count(DISTINCT r1) + count(DISTINCT r2) AS pos_edges,
                   count(DISTINCT n1) + count(DISTINCT n2) AS neg_edges,
                   count(DISTINCT c)                       AS chunk_count
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        r = rows[0] if rows else {}
        return {
            "total_edges":  (r.get("pos_edges") or 0) + (r.get("neg_edges") or 0),
            "chunk_count":  r.get("chunk_count", 0),
        }

    async def _redact_mentions_in_chunks(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str,
    ) -> int:
        """Replace entity_name in chunk text with [REDACTED]."""
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {tenant: $tenant})-[:MENTIONS]->(e:Entity {name: $name, type: $type, tenant: $tenant})
            RETURN c.id AS chunk_id, c.text AS text
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        pattern = re.compile(re.escape(entity_name), re.IGNORECASE)
        count = 0
        for row in rows:
            original = row.get("text") or ""
            redacted = pattern.sub("[REDACTED]", original)
            if redacted != original:
                # The read above (this function's first query) is already
                # tenant-scoped; this write wasn't -- matched by chunk_id
                # alone, so a chunk_id collision across tenants would
                # redact another tenant's chunk text.
                await self._neo4j.run(
                    "MATCH (c:Chunk {id: $chunk_id, tenant: $tenant}) SET c.text = $text, c.redacted = true",
                    chunk_id=row["chunk_id"],
                    tenant=tenant,
                    text=redacted,
                )
                count += 1
        return count
