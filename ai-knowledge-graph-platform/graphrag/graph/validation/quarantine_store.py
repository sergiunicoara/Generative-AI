"""Durable quarantine for records the publication gate refused.

A ``:QuarantinedRecord`` is not an ``:Entity`` and has no edges into the
published graph, so no retrieval path can see it. It keeps the violated rule
ids, the source document, the chunk and the full payload, so it can be fixed
and retried. Ids are deterministic: re-ingesting the same bad record updates
the existing row instead of piling up duplicates.
"""
from __future__ import annotations

import hashlib
import json

from graphrag.graph.validation.batch import RejectedRecord
from graphrag.graph.validation.graph_checks import run_read_only

QUARANTINED = "QUARANTINED"
RESOLVED = "RESOLVED"


def quarantine_id(tenant: str, document_key: str, rec: RejectedRecord) -> str:
    raw = "|".join((tenant, document_key, rec.record_kind, rec.record_key, rec.chunk_id))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


_UPSERT = """
UNWIND $rows AS row
MERGE (q:QuarantinedRecord {tenant: $tenant, id: row.id})
ON CREATE SET q.created_at = datetime(), q.attempts = 0
SET q.record_kind  = row.record_kind,
    q.record_key   = row.record_key,
    q.rule_ids     = row.rule_ids,
    q.messages     = row.messages,
    q.payload_json = row.payload_json,
    q.chunk_id     = row.chunk_id,
    q.source       = $source,
    q.document_id  = $document_id,
    q.document_key = $document_key,
    q.manifest_id  = $manifest_id,
    q.schema_version = $schema_version,
    q.status       = 'QUARANTINED',
    q.updated_at   = datetime()
WITH q
OPTIONAL MATCH (m:IngestionRunManifest {tenant: $tenant, id: $manifest_id})
FOREACH (_ IN CASE WHEN m IS NULL THEN [] ELSE [1] END | MERGE (m)-[:QUARANTINED]->(q))
"""

_FIELDS = """q.id AS id, q.record_kind AS record_kind, q.record_key AS record_key,
       q.rule_ids AS rule_ids, q.messages AS messages, q.status AS status,
       q.source AS source, q.document_id AS document_id, q.document_key AS document_key,
       q.chunk_id AS chunk_id, q.manifest_id AS manifest_id, q.attempts AS attempts,
       q.schema_version AS schema_version, toString(q.created_at) AS created_at,
       toString(q.updated_at) AS updated_at"""


class QuarantineRecordStore:
    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def save(
        self,
        records: list[RejectedRecord],
        *,
        tenant: str,
        source: str,
        document_id: str,
        document_key: str,
        manifest_id: str = "",
        schema_version: str | None = None,
    ) -> list[str]:
        if not records:
            return []
        rows = [{
            "id": quarantine_id(tenant, document_key, r),
            "record_kind": r.record_kind,
            "record_key": r.record_key,
            "rule_ids": r.rule_ids,
            "messages": r.messages,
            "payload_json": json.dumps(r.payload, sort_keys=True, default=str),
            "chunk_id": r.chunk_id,
        } for r in records]
        await self._neo4j.run(
            _UPSERT, rows=rows, tenant=tenant, source=source, document_id=document_id,
            document_key=document_key, manifest_id=manifest_id, schema_version=schema_version,
        )
        return [row["id"] for row in rows]

    async def list(self, *, tenant: str, status: str | None = None, rule_id: str | None = None,
                   limit: int = 100) -> list[dict]:
        return await run_read_only(
            self._neo4j,
            f"""
            MATCH (q:QuarantinedRecord {{tenant: $tenant}})
            WHERE ($status IS NULL OR q.status = $status)
              AND ($rule_id IS NULL OR $rule_id IN q.rule_ids)
            RETURN {_FIELDS}
            ORDER BY q.updated_at DESC
            LIMIT $limit
            """,
            tenant=tenant, status=status, rule_id=rule_id, limit=max(1, min(int(limit), 1000)),
        )

    async def get(self, *, tenant: str, record_id: str) -> dict | None:
        rows = await run_read_only(
            self._neo4j,
            f"""
            MATCH (q:QuarantinedRecord {{tenant: $tenant, id: $id}})
            RETURN {_FIELDS}, q.payload_json AS payload_json
            """,
            tenant=tenant, id=record_id,
        )
        if not rows:
            return None
        row = dict(rows[0])
        row["payload"] = json.loads(row.pop("payload_json") or "{}")
        return row

    async def summary(self, *, tenant: str) -> dict:
        """Record counts by status and source; open-record counts by rule."""
        records = await run_read_only(
            self._neo4j,
            """
            MATCH (q:QuarantinedRecord {tenant: $tenant})
            RETURN q.status AS status, q.source AS source, count(*) AS n
            """,
            tenant=tenant,
        )
        rules = await run_read_only(
            self._neo4j,
            """
            MATCH (q:QuarantinedRecord {tenant: $tenant, status: 'QUARANTINED'})
            UNWIND q.rule_ids AS rule_id
            RETURN rule_id, count(*) AS n
            """,
            tenant=tenant,
        )
        out: dict[str, dict[str, int]] = {"by_status": {}, "by_source": {}, "by_rule": {}}
        for r in records:
            out["by_status"][r["status"]] = out["by_status"].get(r["status"], 0) + r["n"]
            out["by_source"][r["source"]] = out["by_source"].get(r["source"], 0) + r["n"]
        for r in rules:
            out["by_rule"][r["rule_id"]] = r["n"]
        return out

    async def mark_retry(self, *, tenant: str, record_id: str, resolved: bool,
                         rule_ids: list[str], messages: list[str], payload: dict) -> None:
        await self._neo4j.run(
            """
            MATCH (q:QuarantinedRecord {tenant: $tenant, id: $id})
            SET q.attempts     = coalesce(q.attempts, 0) + 1,
                q.status       = $status,
                q.rule_ids     = $rule_ids,
                q.messages     = $messages,
                q.payload_json = $payload_json,
                q.updated_at   = datetime(),
                q.resolved_at  = CASE WHEN $status = 'RESOLVED' THEN datetime() ELSE q.resolved_at END
            """,
            tenant=tenant, id=record_id, status=RESOLVED if resolved else QUARANTINED,
            rule_ids=rule_ids, messages=messages,
            payload_json=json.dumps(payload, sort_keys=True, default=str),
        )
