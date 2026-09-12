"""Tenant-scoped source catalog and versioned ingestion mapping contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from collections.abc import AsyncIterator
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class SourceKind(StrEnum):
    FILE = "file"
    API = "api"
    DATABASE = "database"
    EVENT_STREAM = "event_stream"
    REPOSITORY = "repository"


class SourceStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRED = "retired"


class SourceEnvelope(BaseModel):
    """Provider-neutral record emitted by a source connector."""

    external_id: str = Field(min_length=1)
    content: str
    content_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    cursor: str = ""


class SourceConnector(Protocol):
    """Connector boundary; implementations keep credentials outside Neo4j."""

    kind: SourceKind

    async def records(
        self, source: "SourceSystem", mapping: "SourceMapping", *, cursor: str = "",
    ) -> AsyncIterator[SourceEnvelope]: ...


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[SourceKind, SourceConnector] = {}

    def register(self, connector: SourceConnector) -> None:
        if connector.kind in self._connectors:
            raise ValueError(f"connector already registered for {connector.kind.value}")
        self._connectors[connector.kind] = connector

    def get(self, kind: SourceKind) -> SourceConnector:
        try:
            return self._connectors[kind]
        except KeyError as exc:
            raise LookupError(f"no connector registered for {kind.value}") from exc


class SourceSystem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant: str = Field(min_length=1)
    name: str = Field(min_length=1)
    kind: SourceKind
    uri: str = ""
    owner: str = ""
    classification: str = "internal"
    refresh_sla_seconds: int | None = Field(default=None, ge=0)
    status: SourceStatus = SourceStatus.ACTIVE
    schema_version: str = "kg-source/v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # `uri` deliberately has NO scheme guard (unlike TripleStoreTarget.base_url
    # and RESTAPISourceConfig.base_url/token_url -- see
    # graphrag/core/connector_url_safety.py). It is a general source
    # identifier, not necessarily something this codebase will ever fetch
    # over HTTP: graphrag/ingestion/relational.py's own ingestion path sets
    # it to a connector's DSN (sqlite:///..., postgresql://...), which an
    # http(s)-only allow-list would reject outright. Confirmed by tracing
    # that call site while wiring this guard -- an http(s)-only validator
    # here would have been a real regression, not a hardening.


class SourceMapping(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    tenant: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    mapping: dict[str, Any]
    config_digest: str = ""
    schema_version: str = "kg-source-mapping/v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def populate_or_validate_digest(self) -> "SourceMapping":
        sensitive = {"password", "secret", "token", "api_key", "authorization", "credential"}

        def _contains_secret(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    any(marker in str(key).lower() for marker in sensitive)
                    or _contains_secret(item)
                    for key, item in value.items()
                )
            if isinstance(value, list):
                return any(_contains_secret(item) for item in value)
            return False

        if _contains_secret(self.mapping):
            raise ValueError("source mappings must reference secrets, not persist credential values")
        expected = self.canonical_digest()
        if self.config_digest and self.config_digest != expected:
            raise ValueError("source mapping digest does not match canonical mapping")
        self.config_digest = expected
        return self

    def canonical_digest(self) -> str:
        payload = json.dumps(self.mapping, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def with_digest(self) -> "SourceMapping":
        self.config_digest = self.canonical_digest()
        return self


class SourceCatalogRepository:
    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def upsert_source(self, source: SourceSystem) -> str:
        props = source.model_dump(mode="json")
        await self._neo4j.run(
            """
            MERGE (s:KGSource {tenant: $tenant, id: $id})
            ON CREATE SET s += $props
            ON MATCH SET s.name = $props.name, s.owner = $props.owner,
                         s.classification = $props.classification,
                         s.refresh_sla_seconds = $props.refresh_sla_seconds,
                         s.status = $props.status
            """,
            tenant=source.tenant,
            id=source.id,
            props=props,
        )
        return source.id

    async def add_mapping(self, mapping: SourceMapping) -> str:
        props = mapping.model_dump(mode="json")
        props["mapping"] = json.dumps(mapping.mapping, sort_keys=True, separators=(",", ":"))
        rows = await self._neo4j.run(
            """
            MATCH (s:KGSource {tenant: $tenant, id: $source_id})
            MERGE (m:KGSourceMapping {tenant: $tenant, id: $id})
            ON CREATE SET m += $props
            MERGE (s)-[:HAS_MAPPING]->(m)
            RETURN m.id AS id
            """,
            tenant=mapping.tenant,
            source_id=mapping.source_id,
            id=mapping.id,
            props=props,
        )
        if not rows:
            raise ValueError("source mapping references a missing or cross-tenant source")
        return mapping.id

    async def list_sources(self, tenant: str) -> list[dict]:
        return await self._neo4j.run(
            """
            MATCH (s:KGSource {tenant: $tenant})
            OPTIONAL MATCH (s)-[:HAS_MAPPING]->(m:KGSourceMapping {tenant: $tenant})
            WITH s, m ORDER BY m.created_at DESC
            RETURN s {.*} AS source, collect(m {.*}) AS mappings
            ORDER BY s.name
            """,
            tenant=tenant,
        )

    async def get_source(self, source_id: str, tenant: str) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (s:KGSource {tenant: $tenant, id: $source_id})
            OPTIONAL MATCH (s)-[:HAS_MAPPING]->(m:KGSourceMapping {tenant: $tenant})
            RETURN s {.*} AS source, collect(m {.*}) AS mappings
            """,
            tenant=tenant,
            source_id=source_id,
        )
        return dict(rows[0]) if rows else {}
