"""Ontology registry — versioned entity type and relation schema enforcement.

Problems solved
---------------
1. Schema drift — entity types and relation names accumulate inconsistent
   meanings across ingestion runs. "CEO_OF" extracted in 2023 via one prompt
   version may be semantically different from "CEO_OF" in 2025.

2. Uncontrolled type proliferation — without a registry, every new ingestion
   can introduce new entity types (e.g. "EXEC", "EXECUTIVE", "C_SUITE") that
   fragment the graph and break alias resolution.

Architecture
------------
- OntologyVersion node in Neo4j records each schema snapshot with a hash.
- Validation runs on every extractor output: unknown types are flagged,
  relation names with invalid format are corrected.
- Schema changes are persisted and queryable for audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

# Canonical relation name format: UPPER_SNAKE_CASE
_RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,49}$")
_RELATION_RULES: dict[str, set[tuple[str, str]]] = {
    "FOUNDED": {("PERSON", "ORG"), ("PERSON", "PRODUCT")},
    "FOUNDED_BY": {("ORG", "PERSON"), ("PRODUCT", "PERSON")},
    "CEO_OF": {("PERSON", "ORG")},
    "OWNS": {("PERSON", "ORG"), ("ORG", "ORG"), ("ORG", "PRODUCT")},
    "ACQUIRED": {("ORG", "ORG"), ("ORG", "PRODUCT")},
    "MANUFACTURES": {("ORG", "PRODUCT")},
    "LAUNCHED": {("ORG", "PRODUCT"), ("PERSON", "PRODUCT")},
    "LOCATED_IN": {
        ("ORG", "LOCATION"),
        ("PERSON", "LOCATION"),
        ("EVENT", "LOCATION"),
    },
    "WORKS_AT": {("PERSON", "ORG")},
    "PART_OF": {("PRODUCT", "ORG"), ("ORG", "ORG"), ("EVENT", "ORG")},
    "USES": {("ORG", "PRODUCT"), ("PRODUCT", "PRODUCT"), ("PERSON", "PRODUCT")},
    "RELATED_TO": set(),
}


class OntologyRegistry:
    """
    Versioned ontology registry for entity types and relation schemas.

    Usage::

        registry = OntologyRegistry(neo4j_client)
        await registry.load(entity_types=["PERSON", "ORG", ...])
        result = registry.validate_extraction(entities, relations)
        if result["drift_detected"]:
            log.warning("New types found", new_types=result["new_types"])
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client
        self._allowed_types: set[str] = set()
        self._known_relations: set[str] = set()
        self._migration_map: dict[str, str] = {}   # deprecated → canonical name
        self._version_id: str = ""
        self._loaded: bool = False

    async def load(self, entity_types: list[str]) -> None:
        """Load registry from settings and seed Neo4j OntologyVersion."""
        self._allowed_types = set(entity_types)

        # Load migration map from settings (deprecated relation name → canonical)
        try:
            from graphrag.core.config import get_settings
            onto_cfg = getattr(get_settings(), "ontology", {}) or {}
            raw_map = onto_cfg.get("migration_map", {}) or {}
            self._migration_map = {
                str(k).upper(): str(v).upper() for k, v in raw_map.items()
            }
        except Exception:
            self._migration_map = {}

        # Load known relation types from existing graph
        rows = await self._neo4j.run(
            "MATCH ()-[r:RELATES_TO]->() RETURN DISTINCT r.relation AS rel"
        )
        self._known_relations = {r["rel"] for r in rows if r.get("rel")}

        # Compute version hash from current allowed types
        schema_hash = hashlib.sha256(
            json.dumps(sorted(entity_types)).encode()
        ).hexdigest()[:16]

        # Upsert OntologyVersion node
        result = await self._neo4j.run(
            """
            MERGE (o:OntologyVersion {schema_hash: $hash})
            ON CREATE SET o.id           = $id,
                          o.entity_types = $types,
                          o.created_at   = datetime(),
                          o.active       = true
            RETURN o.id AS version_id
            """,
            hash=schema_hash,
            id=str(uuid4()),
            types=entity_types,
        )
        self._version_id = result[0]["version_id"] if result else ""
        self._loaded = True
        log.info(
            "ontology_registry.loaded",
            types=len(self._allowed_types),
            relations=len(self._known_relations),
            version=self._version_id,
        )

    def validate_extraction(
        self,
        entities: list,
        relations: list,
    ) -> dict:
        """
        Validate extracted entities and relations against the current schema.

        Returns a report with:
          - unknown_types: entity types not in the allowed list
          - malformed_relations: relation names that don't match UPPER_SNAKE_CASE
          - new_relations: relation types never seen before (drift signal)
          - drift_detected: True if any new types or relations were found
          - corrected_relations: how many relation names were auto-corrected
        """
        unknown_types: list[str] = []
        malformed_relations: list[str] = []
        new_relations: list[str] = []
        corrected = 0
        invalid_relation_pairs: list[str] = []

        for entity in entities:
            if entity.type not in self._allowed_types:
                unknown_types.append(f"{entity.name}:{entity.type}")
                # Fallback to CONCEPT rather than creating a new type
                entity.type = "CONCEPT"

        for relation in relations:
            rel_name = relation.relation
            # Auto-correct: uppercase and replace spaces/hyphens
            corrected_name = re.sub(r"[\s\-]+", "_", rel_name.strip()).upper()
            if corrected_name != rel_name:
                relation.relation = corrected_name
                corrected += 1

            # Apply migration map: rename deprecated relation names to canonical ones
            if relation.relation in self._migration_map:
                old_name = relation.relation
                relation.relation = self._migration_map[old_name]
                log.info(
                    "ontology_registry.relation_migrated",
                    old=old_name,
                    new=relation.relation,
                )
                corrected += 1

            if not _RELATION_RE.match(relation.relation):
                malformed_relations.append(relation.relation)
                relation.relation = "RELATED_TO"   # safe fallback

            if relation.relation not in self._known_relations:
                new_relations.append(relation.relation)
                self._known_relations.add(relation.relation)

            src = next((e for e in entities if e.id == relation.source_entity_id), None)
            tgt = next((e for e in entities if e.id == relation.target_entity_id), None)
            if src and tgt:
                allowed_pairs = _RELATION_RULES.get(relation.relation, set())
                if allowed_pairs and (src.type, tgt.type) not in allowed_pairs:
                    invalid_relation_pairs.append(
                        f"{src.name}:{src.type}-{relation.relation}->{tgt.name}:{tgt.type}"
                    )
                    relation.relation = "RELATED_TO"

        drift_detected = bool(unknown_types or new_relations or invalid_relation_pairs)

        if drift_detected:
            log.warning(
                "ontology_registry.drift_detected",
                unknown_types=unknown_types,
                new_relations=new_relations,
                invalid_relation_pairs=invalid_relation_pairs,
            )

        return {
            "unknown_types": unknown_types,
            "malformed_relations": malformed_relations,
            "new_relations": new_relations,
            "invalid_relation_pairs": invalid_relation_pairs,
            "corrected_relations": corrected,
            "drift_detected": drift_detected,
            "version_id": self._version_id,
        }

    def validate_relation_triplet(
        self,
        source_type: str,
        relation: str,
        target_type: str,
    ) -> tuple[bool, str]:
        normalized = re.sub(r"[\s\-]+", "_", relation.strip()).upper()
        # Apply migration rename before domain/range check
        normalized = self._migration_map.get(normalized, normalized)
        if not _RELATION_RE.match(normalized):
            normalized = "RELATED_TO"
        allowed_pairs = _RELATION_RULES.get(normalized, set())
        if not allowed_pairs:
            return True, normalized
        return (source_type, target_type) in allowed_pairs, normalized

    async def persist_migration(
        self,
        old_relation: str,
        new_relation: str,
        migrated_count: int = 0,
    ) -> None:
        """Persist a migration event as an OntologyMigration node for audit."""
        from uuid import uuid4
        await self._neo4j.run(
            """
            CREATE (m:OntologyMigration {
                id:             $id,
                old_relation:   $old,
                new_relation:   $new,
                migrated_count: $count,
                migrated_at:    datetime()
            })
            """,
            id=str(uuid4()),
            old=old_relation,
            new=new_relation,
            count=migrated_count,
        )

    async def apply_graph_migrations(self) -> dict[str, int]:
        """
        Apply migration_map renames to existing RELATES_TO edges in the graph.
        Edges with deprecated relation names are updated to canonical names.
        Returns a count of edges updated per migration rule.
        """
        results: dict[str, int] = {}
        for old_rel, new_rel in self._migration_map.items():
            rows = await self._neo4j.run(
                """
                MATCH ()-[r:RELATES_TO {relation: $old}]->()
                WITH count(r) AS total
                RETURN total
                """,
                old=old_rel,
            )
            count = int(rows[0]["total"]) if rows else 0
            if count > 0:
                await self._neo4j.run(
                    """
                    MATCH ()-[r:RELATES_TO {relation: $old}]->()
                    SET r.relation = $new, r.migrated_from = $old
                    """,
                    old=old_rel,
                    new=new_rel,
                )
                await self.persist_migration(old_rel, new_rel, count)
                log.info(
                    "ontology_registry.graph_migration_applied",
                    old=old_rel,
                    new=new_rel,
                    edges_updated=count,
                )
            results[old_rel] = count
        return results

    async def record_schema_event(
        self,
        event_type: str,        # "new_type" | "new_relation" | "type_correction"
        detail: str,
        source_doc_id: str = "",
    ) -> None:
        """Persist a schema drift event to the OntologyVersion node."""
        if not self._version_id:
            return
        await self._neo4j.run(
            """
            MATCH (o:OntologyVersion {id: $version_id})
            CREATE (o)-[:HAS_EVENT]->(e:OntologyEvent {
                id:           $event_id,
                event_type:   $event_type,
                detail:       $detail,
                source_doc_id: $source_doc_id,
                recorded_at:  datetime()
            })
            """,
            version_id=self._version_id,
            event_id=str(uuid4()),
            event_type=event_type,
            detail=detail,
            source_doc_id=source_doc_id,
        )

    async def get_schema_history(self) -> list[dict]:
        """Return all ontology versions ordered by creation date."""
        return await self._neo4j.run(
            """
            MATCH (o:OntologyVersion)
            OPTIONAL MATCH (o)-[:HAS_EVENT]->(e:OntologyEvent)
            RETURN o.id          AS version_id,
                   o.schema_hash AS hash,
                   o.entity_types AS entity_types,
                   o.created_at  AS created_at,
                   count(e)      AS event_count
            ORDER BY o.created_at DESC
            """
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_registry: OntologyRegistry | None = None


def get_ontology_registry(neo4j_client=None, tenant: str = "default") -> OntologyRegistry:
    global _registry
    if _registry is None:
        if neo4j_client is None:
            from graphrag.graph.neo4j_client import get_neo4j
            neo4j_client = get_neo4j()
        _registry = OntologyRegistry(neo4j_client)
    return _registry
