"""Versioned metadata schema registry and collection coverage reporting."""

from __future__ import annotations

import json

from graphrag.core.config import get_settings
from graphrag.enterprise.models import CollectionSchema, MetadataEnvelope
from graphrag.graph.neo4j_client import get_neo4j


class MetadataGovernanceService:
    def __init__(self, neo4j_client=None):
        self._neo4j = neo4j_client or get_neo4j()

    async def register_schema(self, schema: CollectionSchema) -> dict:
        """Upsert a versioned collection contract without overwriting history."""
        rows = await self._neo4j.run(
            """
            MERGE (s:CollectionMetadataSchema {
                tenant: $tenant, collection: $collection, version: $version
            })
            ON CREATE SET s.id = $id, s.created_at = datetime()
            SET s.required_fields = $required_fields,
                s.allowed_fields = $allowed_fields,
                s.status = $status,
                s.updated_at = datetime()
            RETURN s.id AS id, s.collection AS collection, s.version AS version,
                   s.status AS status
            """,
            id=schema.id,
            tenant=schema.tenant,
            collection=schema.collection,
            version=schema.version,
            required_fields=schema.required_fields,
            allowed_fields=schema.allowed_fields,
            status=schema.status,
        )
        return rows[0] if rows else {}

    async def validate(self, envelope: MetadataEnvelope, tenant: str) -> None:
        """Validate the governed collection tier before ingestion writes data.

        No active schema means the open-discovery tier remains usable unless
        the deployment explicitly requires every collection to be governed.
        """
        rows = await self._neo4j.run(
            """
            MATCH (s:CollectionMetadataSchema {
                tenant: $tenant, collection: $collection, version: $version,
                status: 'active'
            })
            RETURN s.required_fields AS required_fields, s.allowed_fields AS allowed_fields
            LIMIT 1
            """,
            tenant=tenant,
            collection=envelope.collection,
            version=envelope.schema_version,
        )
        if not rows:
            if get_settings().metadata_governance.get("require_active_collection_schema", False):
                raise ValueError(
                    f"no active metadata schema for {envelope.collection!r} "
                    f"version {envelope.schema_version!r}"
                )
            return

        schema = rows[0]
        fields = envelope.collection_metadata
        missing = [field for field in schema.get("required_fields", []) if not fields.get(field)]
        if missing:
            raise ValueError(f"missing required collection metadata: {', '.join(sorted(missing))}")
        allowed = set(schema.get("allowed_fields", []))
        unexpected = sorted(set(fields) - allowed) if allowed else []
        if unexpected:
            raise ValueError(f"collection metadata fields not allowed by schema: {', '.join(unexpected)}")

    async def record_document(self, document_id: str, envelope: MetadataEnvelope, tenant: str) -> None:
        """Link document to the exact schema version used for auditability."""
        await self._neo4j.run(
            """
            MATCH (d:Document {id: $document_id, tenant: $tenant})
            OPTIONAL MATCH (s:CollectionMetadataSchema {
                tenant: $tenant, collection: $collection, version: $version
            })
            FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
                MERGE (d)-[:METADATA_CONFORMS_TO]->(s))
            SET d.collection_metadata_json = $collection_metadata_json,
                d.discovery_metadata_json = $discovery_metadata_json
            """,
            document_id=document_id,
            tenant=tenant,
            collection=envelope.collection,
            version=envelope.schema_version,
            collection_metadata_json=json.dumps(envelope.collection_metadata, sort_keys=True, default=str),
            discovery_metadata_json=json.dumps(envelope.discovery_metadata, sort_keys=True),
        )

    async def coverage(self, tenant: str, collection: str | None = None) -> list[dict]:
        """Return per-collection metadata coverage for operational dashboards."""
        return await self._neo4j.run(
            """
            MATCH (d:Document {tenant: $tenant})
            WHERE $collection IS NULL OR d.collection = $collection
            WITH coalesce(d.collection, 'default') AS collection,
                 count(d) AS documents,
                 sum(CASE WHEN coalesce(d.external_id, '') <> '' THEN 1 ELSE 0 END) AS external_ids,
                 sum(CASE WHEN coalesce(d.source_version, '') <> '' THEN 1 ELSE 0 END) AS versions,
                 sum(CASE WHEN coalesce(d.classification, '') <> '' THEN 1 ELSE 0 END) AS classifications,
                 sum(CASE WHEN coalesce(d.metadata_schema_version, '') <> '' THEN 1 ELSE 0 END) AS schema_versions,
                 sum(CASE WHEN coalesce(d.acl_state, 'unknown') = 'known' THEN 1 ELSE 0 END) AS known_acls
            RETURN collection, documents, external_ids, versions, classifications,
                   schema_versions, known_acls
            ORDER BY collection
            """,
            tenant=tenant,
            collection=collection,
        )
