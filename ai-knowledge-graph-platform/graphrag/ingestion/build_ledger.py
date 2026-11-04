"""Per-build lineage for relational ingestion.

Every relational build gets a record: the tables it read (row count and a
content hash each), the code version, the mapping version, the expectation
results, the outcome (``published``, ``blocked``, ``skipped``, ``failed``) and
the counts it wrote. ``BuildLedger`` stores records as ``IngestionBuild``
nodes; a published build is linked to the source ``Document`` it wrote
(``PUBLISHED``), so the lineage of an entity resolves through its chunk and
document to the latest build that published it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def code_version() -> str:
    """``GRAPHRAG_CODE_VERSION``, else ``GITHUB_SHA``, else ``git rev-parse HEAD``, else ``unknown``."""
    for name in ("GRAPHRAG_CODE_VERSION", "GITHUB_SHA"):
        if os.getenv(name):
            return str(os.getenv(name))
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5, check=True).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def table_digest(rows: list[dict[str, Any]]) -> str:
    canonical = sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BuildRecord:
    tenant: str
    source_id: str
    source_uri: str
    mapping_id: str
    mapping_version: str
    code_version: str
    build_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_now)
    finished_at: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    expectations: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = "running"
    outputs: dict[str, int] = field(default_factory=dict)
    document_id: str = ""
    error: str = ""

    @classmethod
    def start(cls, *, tenant: str, source_id: str, source_uri: str, mapping_id: str, mapping_version: str,
              version: str | None = None) -> "BuildRecord":
        return cls(tenant=tenant, source_id=source_id, source_uri=source_uri, mapping_id=mapping_id,
                   mapping_version=mapping_version, code_version=version or code_version())

    def read(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self.inputs = [{"table": name, "rows": len(rows), "sha256": table_digest(rows)}
                       for name, rows in sorted(tables.items())]

    def finish(self, outcome: str, *, outputs: dict[str, int] | None = None, document_id: str = "",
               error: str = "") -> "BuildRecord":
        self.outcome, self.finished_at = outcome, _now()
        self.outputs = dict(outputs or {})
        self.document_id, self.error = document_id, error[:2000]
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BuildLedger:
    """Persist build records in the graph, tenant-scoped."""

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def record(self, build: BuildRecord) -> None:
        await self._neo4j.run(
            """
            MERGE (b:IngestionBuild {tenant: $tenant, id: $id})
            SET b.source_id = $source_id, b.source_uri = $source_uri, b.mapping_id = $mapping_id,
                b.mapping_version = $mapping_version, b.code_version = $code_version,
                b.started_at = $started_at, b.finished_at = $finished_at, b.outcome = $outcome,
                b.inputs_json = $inputs_json, b.expectations_json = $expectations_json,
                b.outputs_json = $outputs_json, b.error = $error
            WITH b
            OPTIONAL MATCH (d:Document {tenant: $tenant, id: $document_id})
            FOREACH (_ IN CASE WHEN d IS NULL OR $outcome <> 'published' THEN [] ELSE [1] END |
                MERGE (b)-[:PUBLISHED]->(d))
            """,
            tenant=build.tenant, id=build.build_id, source_id=build.source_id, source_uri=build.source_uri,
            mapping_id=build.mapping_id, mapping_version=build.mapping_version, code_version=build.code_version,
            started_at=build.started_at, finished_at=build.finished_at, outcome=build.outcome,
            inputs_json=json.dumps(build.inputs, sort_keys=True), document_id=build.document_id,
            expectations_json=json.dumps(build.expectations, sort_keys=True, default=str),
            outputs_json=json.dumps(build.outputs, sort_keys=True), error=build.error,
        )

    async def lineage(self, tenant: str, entity_id: str) -> dict[str, Any] | None:
        """The latest published build behind an entity (via chunk and document)."""
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {tenant: $tenant, id: $entity_id})<-[:MENTIONS]-(c:Chunk {tenant: $tenant})
                  -[:PART_OF]->(d:Document {tenant: $tenant})
                  <-[:PUBLISHED]-(b:IngestionBuild {tenant: $tenant, outcome: 'published'})
            RETURN DISTINCT b.id AS build_id, b.source_id AS source_id, b.mapping_version AS mapping_version,
                   b.code_version AS code_version, b.finished_at AS finished_at, b.inputs_json AS inputs_json,
                   d.id AS document_id
            ORDER BY finished_at DESC LIMIT 1
            """,
            tenant=tenant, entity_id=entity_id,
        )
        rows = list(rows or [])
        if not rows:
            return None
        row = dict(rows[0])
        row["inputs"] = json.loads(row.pop("inputs_json") or "[]")
        return row

    async def builds(self, tenant: str, source_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = await self._neo4j.run(
            """
            MATCH (b:IngestionBuild {tenant: $tenant, source_id: $source_id})
            RETURN b.id AS build_id, b.outcome AS outcome, b.started_at AS started_at,
                   b.finished_at AS finished_at, b.code_version AS code_version,
                   b.mapping_version AS mapping_version, b.expectations_json AS expectations_json
            ORDER BY b.started_at DESC LIMIT $limit
            """,
            tenant=tenant, source_id=source_id, limit=limit,
        )
        out = []
        for row in rows or []:
            row = dict(row)
            row["expectations"] = json.loads(row.pop("expectations_json") or "[]")
            out.append(row)
        return out


__all__ = ["BuildLedger", "BuildRecord", "code_version", "table_digest"]
