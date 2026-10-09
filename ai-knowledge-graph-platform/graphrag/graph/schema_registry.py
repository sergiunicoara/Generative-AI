"""Versioned schema registry (plan Phase 2, decision D2).

Evolves the existing ``:OntologyVersion`` node into the portable model

    (:Dataset)-[:CONFORMS_TO]->(:SchemaVersion)-[:USES_VALIDATION_PROFILE]->(:ValidationProfile)
    (:SchemaVersion)-[:IMPORTS]->(:SchemaVersion)

An ontology version node carries both labels (``OntologyVersion`` for the
existing proposal / event links, ``SchemaVersion`` for the registry model), so
nothing that already points at it changes. Every node carries ``tenant``; every
query filters on it, so one tenant can never read, activate or drift-check
another tenant's schema.

Semantics
---------
* A schema version is identified by a deterministic ``content_hash`` over the
  whole effective contract: entity types, domain/range rules, vocabulary,
  migration map, the ontology YAML sections and version, and the SHACL
  validation profile (hash of the shape files). Any change yields a new version.
* At most one version per (tenant, dataset) is ``active``. Older versions are
  retained, deactivated, for reproducibility.
* Drift = the schema the running code computes differs from the active
  registered version. ``mode: auto`` (default, backward compatible) records the
  drift and activates the new version. ``mode: enforce`` refuses to run on a
  schema that is not the active registered one.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover
    Counter = None

_drift = Counter(
    "graphrag_schema_drift_events_total",
    "Schema load outcomes: match, version_change, rollback, blocked",
    ["outcome"],
) if Counter else None


def record_drift(outcome: str) -> None:
    if _drift is not None:
        _drift.labels(outcome=outcome).inc()

AUTO = "auto"
ENFORCE = "enforce"
DEFAULT_PROFILE_NAME = "platform-shacl-shapes"
_SHAPE_FILES = ("ingestion.shapes.ttl", "export.shapes.ttl")
_YAML_SECTIONS = (
    "type_hierarchy", "relation_rules", "inference_rules",
    "exclusive_state_pairs", "functional_relations", "vocabulary",
)


class SchemaError(RuntimeError):
    pass


class UnknownSchemaError(SchemaError):
    """The computed schema is not a registered version of the dataset."""


class SchemaDriftError(SchemaError):
    """The computed schema differs from the dataset's active registered version."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def profile_hash(shapes_dir: Path) -> str:
    """Hash of the SHACL validation profile (sorted shape-file bytes)."""
    h = hashlib.sha256()
    for name in _SHAPE_FILES:
        path = shapes_dir / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes() if path.exists() else b"<missing>")
    return h.hexdigest()


def compute_content_hash(
    *,
    allowed_types,
    domain_rules: dict[str, set[tuple[str, str]]],
    builtin_rules: dict[str, set[tuple[str, str]]],
    vocabulary: dict,
    migration_map: dict[str, str],
    ontology_doc: dict | None,
    profile_hash_: str,
) -> str:
    """Deterministic over everything that changes what is valid or what a term means."""
    def pairs(rules):
        return {r: sorted(f"{s}:{t}" for s, t in p) for r, p in sorted(rules.items())}

    doc = ontology_doc or {}
    return _digest({
        "types": sorted(allowed_types),
        "domain_rules": pairs(domain_rules),
        "builtin_rules": pairs(builtin_rules),
        "vocabulary": vocabulary,
        "migration_map": dict(sorted(migration_map.items())),
        "ontology": doc.get("ontology") or {},
        "yaml": {k: doc.get(k) for k in _YAML_SECTIONS if k in doc},
        "profile": profile_hash_,
    })


@dataclass(frozen=True)
class SchemaIdentity:
    """What a registry load computed: enough to register and to label answers."""
    tenant: str
    dataset_id: str
    name: str
    version: str
    content_hash: str
    profile_hash: str
    source_uri: str = ""
    profile_version: str = "1"
    types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}#{self.content_hash[:12]}"


def dataset_id_for(tenant: str, cfg: dict | None = None) -> str:
    """Dataset a tenant's ontology belongs to; defaults to one dataset per tenant."""
    mapping = ((cfg or {}).get("schema_registry") or {}).get("datasets") or {}
    return str(mapping.get(tenant) or tenant)


# One statement: upsert dataset/profile/version, deactivate the dataset's other
# versions, activate this one, repoint CONFORMS_TO. Legacy rows (no dataset_id)
# of the tenant are treated as belonging to the dataset being loaded.
UPSERT_AND_ACTIVATE = """
MERGE (d:Dataset {tenant: $tenant, id: $dataset_id})
  ON CREATE SET d.name = $dataset_name, d.created_at = datetime($now)
MERGE (p:ValidationProfile {tenant: $tenant, content_hash: $profile_hash})
  ON CREATE SET p.id = $profile_id, p.name = $profile_name,
                p.version = $profile_version, p.created_at = datetime($now)
MERGE (o:OntologyVersion:SchemaVersion {tenant: $tenant, dataset_id: $dataset_id,
                                        content_hash: $content_hash})
  ON CREATE SET o.id = $id, o.schema_hash = $legacy_hash, o.entity_types = $types,
                o.name = $name, o.version = $version, o.source_uri = $source_uri,
                o.created_at = datetime($now), o.active = false
WITH d, p, o, o.active AS was_active, (o.created_at = datetime($now)) AS created
OPTIONAL MATCH (other:OntologyVersion {tenant: $tenant})
  WHERE other <> o AND coalesce(other.dataset_id, $dataset_id) = $dataset_id
    AND other.active = true
WITH d, p, o, was_active, created, collect(other) AS others
FOREACH (x IN others | SET x.active = false, x.deactivated_at = datetime($now))
SET o.active = true, o.loaded_at = datetime($now),
    o.activated_at = CASE WHEN was_active = true THEN o.activated_at ELSE datetime($now) END
MERGE (d)-[:CONFORMS_TO]->(o)
MERGE (o)-[:USES_VALIDATION_PROFILE]->(p)
WITH o, was_active, created, others
OPTIONAL MATCH (:Dataset {tenant: $tenant, id: $dataset_id})-[c:CONFORMS_TO]->(old:SchemaVersion)
  WHERE old <> o
DELETE c
RETURN DISTINCT o.id AS version_id, created AS created, was_active AS was_active,
       [x IN others | coalesce(x.content_hash, x.schema_hash)] AS prior_hashes
"""

_LOOKUP = """
MATCH (o:OntologyVersion {tenant: $tenant})
WHERE coalesce(o.dataset_id, $dataset_id) = $dataset_id
RETURN o.id AS id, coalesce(o.content_hash, o.schema_hash) AS content_hash,
       o.name AS name, o.version AS version, coalesce(o.active, false) AS active,
       o.source_uri AS source_uri, toString(o.created_at) AS created_at,
       toString(o.activated_at) AS activated_at, toString(o.deactivated_at) AS deactivated_at
ORDER BY coalesce(o.activated_at, o.created_at) DESC
"""


class SchemaRegistry:
    """Operations on the registry. All reads and writes are tenant-scoped."""

    _label_cache: dict[tuple[int, str, str], tuple[float, str]] = {}
    LABEL_TTL_SECONDS = 30.0

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── reads ────────────────────────────────────────────────────────────────
    async def versions(self, tenant: str, dataset_id: str) -> list[dict]:
        run = getattr(self._neo4j, "run_read", None) or self._neo4j.run
        return await run(_LOOKUP, tenant=tenant, dataset_id=dataset_id)

    async def get_active(self, tenant: str, dataset_id: str) -> dict | None:
        return next((v for v in await self.versions(tenant, dataset_id) if v["active"]), None)

    async def check(self, identity: SchemaIdentity) -> dict:
        """Compare the computed schema to the registry without changing it.

        status: ``match`` (active version has this hash), ``drift`` (a different
        version is active; ``known`` says whether this hash was registered
        before), ``unregistered`` (dataset has no active version yet).
        """
        versions = await self.versions(identity.tenant, identity.dataset_id)
        active = next((v for v in versions if v["active"]), None)
        known = any(v["content_hash"] == identity.content_hash for v in versions)
        if active is None:
            status = "unregistered"
        elif active["content_hash"] == identity.content_hash:
            status = "match"
        else:
            status = "drift"
        return {"status": status, "known": known, "active": active,
                "computed_hash": identity.content_hash}

    async def active_label(self, tenant: str, dataset_id: str | None = None) -> str | None:
        """Human-readable active schema version for answer provenance (short TTL cache)."""
        dataset_id = dataset_id or tenant
        key = (id(self._neo4j), tenant, dataset_id)
        hit = self._label_cache.get(key)
        now = time.monotonic()
        if hit and now - hit[0] < self.LABEL_TTL_SECONDS:
            return hit[1] or None
        active = await self.get_active(tenant, dataset_id)
        label = ""
        if active:
            label = f"{active.get('name') or dataset_id}@{active.get('version') or '0'}#{active['content_hash'][:12]}"
        self._label_cache[key] = (now, label)
        return label or None

    @classmethod
    def invalidate_labels(cls) -> None:
        cls._label_cache.clear()

    # ── writes ───────────────────────────────────────────────────────────────
    async def register_and_activate(self, identity: SchemaIdentity, *, now_iso: str | None = None) -> dict:
        """Idempotent. Re-registering the active hash changes nothing; registering a
        previously known hash is a rollback (reactivation)."""
        from datetime import datetime, timezone

        rows = await self._neo4j.run(
            UPSERT_AND_ACTIVATE,
            tenant=identity.tenant,
            dataset_id=identity.dataset_id,
            dataset_name=identity.dataset_id,
            content_hash=identity.content_hash,
            legacy_hash=identity.content_hash[:16],
            profile_hash=identity.profile_hash,
            profile_id=str(uuid4()),
            profile_name=DEFAULT_PROFILE_NAME,
            profile_version=identity.profile_version,
            id=str(uuid4()),
            types=list(identity.types),
            name=identity.name,
            version=identity.version,
            source_uri=identity.source_uri,
            now=now_iso or datetime.now(timezone.utc).isoformat(),
        )
        self.invalidate_labels()
        row = rows[0] if rows else {}
        return {
            "version_id": row.get("version_id", ""),
            "created": bool(row.get("created", False)),
            "was_active": bool(row.get("was_active", False)),
            "prior_hashes": [h for h in row.get("prior_hashes", []) if h],
        }

    async def deactivate(self, tenant: str, dataset_id: str) -> int:
        """Deactivate the active version. The dataset then has no active schema, so
        an enforcing load refuses to run until one is activated."""
        rows = await self._neo4j.run(
            """
            MATCH (o:OntologyVersion {tenant: $tenant})
            WHERE coalesce(o.dataset_id, $dataset_id) = $dataset_id AND o.active = true
            SET o.active = false, o.deactivated_at = datetime()
            WITH collect(o) AS gone
            OPTIONAL MATCH (:Dataset {tenant: $tenant, id: $dataset_id})-[c:CONFORMS_TO]->(:SchemaVersion)
            DELETE c
            RETURN size(gone) AS n
            """,
            tenant=tenant, dataset_id=dataset_id,
        )
        self.invalidate_labels()
        return int(rows[0]["n"]) if rows else 0

    async def activate(self, tenant: str, dataset_id: str, version_id: str) -> dict:
        """Activate an already-registered version of THIS tenant's dataset."""
        versions = await self.versions(tenant, dataset_id)
        target = next((v for v in versions if v["id"] == version_id), None)
        if target is None:
            raise UnknownSchemaError(f"version {version_id!r} is not registered for dataset {dataset_id!r}")
        if target["active"]:
            return {"version_id": version_id, "changed": False}
        rows = await self._neo4j.run(
            """
            MATCH (o:OntologyVersion {tenant: $tenant, id: $id})
            WHERE coalesce(o.dataset_id, $dataset_id) = $dataset_id
            OPTIONAL MATCH (other:OntologyVersion {tenant: $tenant})
              WHERE other <> o AND coalesce(other.dataset_id, $dataset_id) = $dataset_id AND other.active = true
            WITH o, collect(other) AS others
            FOREACH (x IN others | SET x.active = false, x.deactivated_at = datetime())
            SET o.active = true, o.activated_at = datetime(),
                o.dataset_id = coalesce(o.dataset_id, $dataset_id)
            MERGE (d:Dataset {tenant: $tenant, id: $dataset_id})
              ON CREATE SET d.name = $dataset_id, d.created_at = datetime()
            MERGE (d)-[:CONFORMS_TO]->(o)
            WITH o, d
            OPTIONAL MATCH (d)-[c:CONFORMS_TO]->(old:SchemaVersion) WHERE old <> o
            DELETE c
            RETURN DISTINCT o.id AS id
            """,
            tenant=tenant, dataset_id=dataset_id, id=version_id,
        )
        self.invalidate_labels()
        return {"version_id": version_id, "changed": bool(rows)}

    async def rollback(self, tenant: str, dataset_id: str) -> dict:
        """Reactivate the version that was active before the current one."""
        versions = await self.versions(tenant, dataset_id)
        previous = [v for v in versions if not v["active"] and v.get("deactivated_at")]
        if not previous:
            raise UnknownSchemaError(f"dataset {dataset_id!r} has no earlier version to roll back to")
        previous.sort(key=lambda v: v["deactivated_at"], reverse=True)
        return await self.activate(tenant, dataset_id, previous[0]["id"])

    async def add_import(self, tenant: str, schema_id: str, imported_id: str) -> bool:
        """``(:SchemaVersion)-[:IMPORTS]->(:SchemaVersion)``; both ends must be this tenant's."""
        if schema_id == imported_id:
            raise SchemaError("a schema cannot import itself")
        rows = await self._neo4j.run(
            """
            MATCH (a:SchemaVersion {tenant: $tenant, id: $a}), (b:SchemaVersion {tenant: $tenant, id: $b})
            MERGE (a)-[:IMPORTS]->(b)
            RETURN count(*) AS n
            """,
            tenant=tenant, a=schema_id, b=imported_id,
        )
        return bool(rows and rows[0]["n"])

    # ── gate used by OntologyRegistry.load ──────────────────────────────────
    async def enforce(self, identity: SchemaIdentity, *, mode: str) -> dict:
        """Apply the drift policy before activation.

        ``auto``: never raises; returns the check result so the caller can record drift.
        ``enforce``: a dataset with no active version is bootstrapped (nothing to drift
        from); otherwise the computed hash must equal the active one.
        """
        result = await self.check(identity)
        if mode == ENFORCE and result["status"] == "drift":
            active = result["active"]
            raise (SchemaDriftError if result["known"] else UnknownSchemaError)(
                f"dataset {identity.dataset_id!r}: computed schema {identity.content_hash[:12]} "
                f"is not the active registered version {active['content_hash'][:12]} "
                f"({'registered but inactive' if result['known'] else 'unregistered'}); "
                f"activate it explicitly with scripts/schema_registry.py"
            )
        return result


async def startup_drift_check(neo4j=None) -> list[dict]:
    """Report, never fix, tenants whose ontology files no longer match the active version.

    Runs at API/worker startup. It is read-only and never raises: a drifted
    tenant is logged and counted so operators see it before the first ingest;
    ``mode: enforce`` then blocks that tenant's ingestion in ``OntologyRegistry.load``.
    The tenant list is the set of tenants that already have an active version
    (a system-level read; results are reported per tenant, not merged).
    """
    from graphrag.core.config import get_settings
    from graphrag.graph.neo4j_client import get_neo4j
    from graphrag.graph.ontology_registry import OntologyRegistry

    results: list[dict] = []
    try:
        neo4j = neo4j or get_neo4j()
        cfg = get_settings()
        entity_types = cfg.ingestion.get(
            "entity_types", ["PERSON", "ORG", "PRODUCT", "CONCEPT", "LOCATION", "EVENT"])
        run = getattr(neo4j, "run_read", None) or neo4j.run
        rows = await run("MATCH (o:OntologyVersion) WHERE o.active = true RETURN DISTINCT o.tenant AS tenant")
        for tenant in sorted({r["tenant"] for r in rows if r.get("tenant")}):
            try:
                res = await OntologyRegistry(neo4j, tenant=tenant).check_schema_drift(list(entity_types))
            except Exception as exc:  # noqa: BLE001 - one tenant must not hide the others
                log.warning("schema_registry.startup_check_failed", error=str(exc)[:120])
                continue
            res["tenant"] = tenant
            results.append(res)
            if res["status"] == "drift":
                record_drift("startup_drift")
                log.warning("schema_registry.startup_drift", tenant=tenant, dataset=res["dataset_id"],
                            computed=res["computed_hash"][:12], known=res["known"],
                            active=(res["active"] or {}).get("content_hash", "")[:12])
    except Exception as exc:  # noqa: BLE001
        log.warning("schema_registry.startup_check_unavailable", error=str(exc)[:120])
    return results
