"""Audit trail — records every graph mutation with full before/after state.

Problem solved
--------------
Without an audit trail, there is no answer to:
  - Who changed the status of entity X and when?
  - What was the confidence of this edge before last night's ingestion?
  - Which ingestion run introduced this relation?

Every write that goes through GraphWriter passes through AuditTrail,
which creates a ChangeLog node in Neo4j linked to the mutated entity.

The trail is append-only — ChangeLog nodes are never deleted,
only queried.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)


class AuditTrail:
    """
    Append-only audit log for all graph mutations.

    Usage::

        trail = AuditTrail(neo4j_client)
        await trail.log_entity_change(
            entity_name="SpaceX",
            entity_type="ORG",
            operation="update",
            old_values={"description": ""},
            new_values={"description": "Rocket company"},
            changed_by="ingestion_worker",
            source_doc_id="doc_abc",
        )
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def log_entity_change(
        self,
        entity_name: str,
        entity_type: str,
        operation: str,
        old_values: dict[str, Any] | None = None,
        new_values: dict[str, Any] | None = None,
        changed_by: str = "system",
        source_doc_id: str = "",
    ) -> None:
        try:
            await self._neo4j.run(
                """
                MATCH (e:Entity {name: $name, type: $type})
                WITH e LIMIT 1
                CREATE (e)-[:HAS_CHANGE]->(cl:ChangeLog {
                    id:           $log_id,
                    target_label: 'Entity',
                    target_id:    e.id,
                    operation:    $operation,
                    changed_by:   $changed_by,
                    changed_at:   datetime(),
                    old_values:   $old_values,
                    new_values:   $new_values,
                    source_doc_id: $source_doc_id
                })
                """,
                name=entity_name,
                type=entity_type,
                log_id=str(uuid4()),
                operation=operation,
                changed_by=changed_by,
                old_values=str(old_values or {}),
                new_values=str(new_values or {}),
                source_doc_id=source_doc_id,
            )
        except Exception as exc:
            log.warning("audit_trail.entity_log_failed",
                        entity=entity_name, operation=operation, error=str(exc)[:120])

    async def log_relation_change(
        self,
        src_name: str,
        tgt_name: str,
        relation: str,
        operation: str,
        old_values: dict[str, Any] | None = None,
        new_values: dict[str, Any] | None = None,
        changed_by: str = "system",
        source_doc_id: str = "",
    ) -> None:
        try:
            await self._neo4j.run(
                """
                MATCH (s:Entity {name: $src})-[r:RELATES_TO {relation: $relation}]->(t:Entity {name: $tgt})
                WITH r LIMIT 1
                CREATE (cl:ChangeLog {
                    id:            $log_id,
                    target_label:  'Relation',
                    target_id:     $src + '->' + $tgt + ':' + $relation,
                    operation:     $operation,
                    changed_by:    $changed_by,
                    changed_at:    datetime(),
                    old_values:    $old_values,
                    new_values:    $new_values,
                    source_doc_id: $source_doc_id
                })
                """,
                src=src_name,
                tgt=tgt_name,
                relation=relation,
                log_id=str(uuid4()),
                operation=operation,
                changed_by=changed_by,
                old_values=str(old_values or {}),
                new_values=str(new_values or {}),
                source_doc_id=source_doc_id,
            )
        except Exception as exc:
            log.warning("audit_trail.relation_log_failed",
                        src=src_name, tgt=tgt_name, operation=operation, error=str(exc)[:120])

    async def log_document_change(
        self,
        doc_id: str,
        operation: str,
        old_values: dict[str, Any] | None = None,
        new_values: dict[str, Any] | None = None,
        changed_by: str = "system",
    ) -> None:
        try:
            await self._neo4j.run(
                """
                MATCH (d:Document {id: $doc_id})
                WITH d LIMIT 1
                CREATE (d)-[:HAS_CHANGE]->(cl:ChangeLog {
                    id:           $log_id,
                    target_label: 'Document',
                    target_id:    $doc_id,
                    operation:    $operation,
                    changed_by:   $changed_by,
                    changed_at:   datetime(),
                    old_values:   $old_values,
                    new_values:   $new_values,
                    source_doc_id: $doc_id
                })
                """,
                doc_id=doc_id,
                log_id=str(uuid4()),
                operation=operation,
                changed_by=changed_by,
                old_values=str(old_values or {}),
                new_values=str(new_values or {}),
            )
        except Exception as exc:
            log.warning("audit_trail.document_log_failed",
                        doc_id=doc_id, operation=operation, error=str(exc)[:120])

    async def get_history(
        self,
        entity_name: str,
        entity_type: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return the N most recent changes for an entity."""
        return await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type})-[:HAS_CHANGE]->(cl:ChangeLog)
            RETURN cl.operation     AS operation,
                   cl.changed_by   AS changed_by,
                   cl.changed_at   AS changed_at,
                   cl.old_values   AS old_values,
                   cl.new_values   AS new_values,
                   cl.source_doc_id AS source_doc_id
            ORDER BY cl.changed_at DESC
            LIMIT $limit
            """,
            name=entity_name,
            type=entity_type,
            limit=limit,
        )
