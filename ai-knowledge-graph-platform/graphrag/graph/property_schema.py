"""Property schema — cardinality and type constraints on entity attributes.

Problem solved
--------------
The ontology validates relation domain/range (CEO_OF: PERSON→ORG) but says
nothing about entity attribute constraints:
  - A PERSON should have at most one BIRTH_DATE.
  - An ORG should have at most one FOUNDED_YEAR and at most one HQ_LOCATION.
  - A PRODUCT must have a non-empty NAME.

Without these constraints multiple ingestion runs can write conflicting
attribute values (two BIRTH_DATEs for the same person from two documents)
and the system has no way to detect or report the violation.

Architecture
------------
PropertySchemaValidator holds a dict of per-type rules:
    PROPERTY_RULES[entity_type][property_name] = PropertyRule

PropertyRule defines:
    cardinality  : "single" (at most one value) | "multi" (list OK) | "required"
    value_type   : "str" | "int" | "float" | "datetime" | any
    allowed_values : optional set of permitted values (enum constraint)

Validation runs post-ingestion (wired into IngestionValidator) by reading
all property values written for a given entity and comparing against rules.

Violations are logged as warnings and returned as structured issue dicts
(same format as IngestionValidator).  They do NOT block ingestion — property
values are untyped in Neo4j and the constraint is advisory.

Conflict detection
------------------
For "single" cardinality properties, if two source documents write different
values for the same property (e.g. BIRTH_DATE from doc A = 1970, from doc B
= 1971), the validator surfaces this as a `property_conflict` issue.
The AuditTrail already records old_values/new_values on every write, so the
conflict is reconstructable from the ChangeLog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class PropertyRule:
    """Constraint on a single entity attribute."""
    cardinality: str        # "single" | "multi" | "required"
    value_type:  str = "str"          # expected Python type name
    allowed_values: set[Any] = field(default_factory=set)  # empty = any value OK
    description: str = ""


# ── Default rules ──────────────────────────────────────────────────────────────

PROPERTY_RULES: dict[str, dict[str, PropertyRule]] = {
    "PERSON": {
        "birth_date":    PropertyRule("single", "str",   description="ISO date of birth"),
        "nationality":   PropertyRule("single", "str"),
        "gender":        PropertyRule("single", "str",   allowed_values={"M", "F", "NB", "unknown"}),
        "email":         PropertyRule("single", "str"),
        "description":   PropertyRule("single", "str"),
    },
    "ORG": {
        "founded_year":  PropertyRule("single", "int",   description="4-digit year"),
        "hq_location":   PropertyRule("single", "str"),
        "org_type":      PropertyRule("single", "str",
                                      allowed_values={"corp", "gov", "ngo", "edu", "startup", "other"}),
        "description":   PropertyRule("single", "str"),
        "ticker":        PropertyRule("single", "str"),
    },
    "PRODUCT": {
        "part_number":   PropertyRule("required", "str"),
        "manufacturer":  PropertyRule("single",   "str"),
        "material":      PropertyRule("single",   "str"),
        "spec_revision": PropertyRule("single",   "str"),
        "description":   PropertyRule("single",   "str"),
    },
    "LOCATION": {
        "country_code":  PropertyRule("single", "str"),
        "latitude":      PropertyRule("single", "float"),
        "longitude":     PropertyRule("single", "float"),
    },
    "EVENT": {
        "event_date":    PropertyRule("single", "str"),
        "event_type":    PropertyRule("single", "str"),
        "location":      PropertyRule("single", "str"),
        "description":   PropertyRule("single", "str"),
    },
}


class PropertySchemaValidator:
    """
    Validate entity attribute constraints and detect property-level conflicts.

    Usage::

        validator = PropertySchemaValidator(neo4j_client)

        # Validate a single entity
        issues = await validator.validate_entity("Elon Musk", "PERSON", tenant="default")

        # Validate all entities for a document
        issues = await validator.validate_document("doc_abc", tenant="default")

        # Register a custom rule at runtime
        validator.add_rule("PRODUCT", "weight_kg", PropertyRule("single", "float"))
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client
        # Mutable copy so callers can add rules at runtime
        self._rules: dict[str, dict[str, PropertyRule]] = {
            et: dict(rules) for et, rules in PROPERTY_RULES.items()
        }

    def add_rule(self, entity_type: str, prop_name: str, rule: PropertyRule) -> None:
        """Register or overwrite a property rule at runtime."""
        self._rules.setdefault(entity_type, {})[prop_name] = rule
        log.info("property_schema.rule_added", entity_type=entity_type, prop=prop_name)

    def get_rules(self, entity_type: str) -> dict[str, PropertyRule]:
        """Return all rules for a given entity type."""
        return self._rules.get(entity_type, {})

    # ── Validation ─────────────────────────────────────────────────────────────

    async def validate_entity(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
    ) -> list[dict]:
        """
        Check all attribute constraints for a single entity.

        Returns list of issue dicts:
            type             : "missing_required" | "invalid_value" | "property_conflict"
            entity           : entity name
            entity_type      : entity type
            property         : property name
            value            : current value
            constraint       : violated rule description
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            RETURN e {.*} AS props
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        if not rows:
            return []

        props = dict(rows[0].get("props") or {})
        return self._check_props(entity_name, entity_type, props)

    async def validate_document(
        self,
        doc_id: str,
        tenant: str = "default",
    ) -> list[dict]:
        """
        Validate all entities introduced or updated by a document.
        Returns all property-level violations as a flat list.
        """
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})
                  -[:MENTIONS]->(e:Entity {tenant: $tenant})
            RETURN DISTINCT e.name AS name, e.type AS type, e {.*} AS props
            """,
            doc_id=doc_id,
            tenant=tenant,
        )
        issues: list[dict] = []
        for row in rows:
            issues += self._check_props(
                row["name"],
                row["type"],
                dict(row.get("props") or {}),
            )
        return issues

    async def detect_property_conflicts(
        self,
        entity_type: str = "",
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """
        Find entities where the ChangeLog shows different values written for a
        "single" cardinality property by different source documents.

        These are property-level contradictions: doc A says birth_date=1970,
        doc B says birth_date=1971.
        """
        type_filter = "AND e.type = $entity_type" if entity_type else ""
        params: dict = {"tenant": tenant, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type

        # Find single-cardinality props from rules
        single_props = []
        for et, rules in self._rules.items():
            if entity_type and et != entity_type:
                continue
            for prop, rule in rules.items():
                if rule.cardinality in ("single", "required"):
                    single_props.append(prop)
        if not single_props:
            return []

        rows = await self._neo4j.run(
            f"""
            MATCH (cl1:ChangeLog)-[:TARGETS]->(e:Entity)
            WHERE e.tenant = $tenant {type_filter}
            MATCH (cl2:ChangeLog)-[:TARGETS]->(e)
            WHERE cl1.source_doc_id <> cl2.source_doc_id
              AND cl1.source_doc_id IS NOT NULL
              AND cl2.source_doc_id IS NOT NULL
            RETURN e.name AS entity, e.type AS entity_type,
                   cl1.source_doc_id AS doc_a, cl2.source_doc_id AS doc_b,
                   cl1.new_values AS vals_a, cl2.new_values AS vals_b
            LIMIT $limit
            """,
            **params,
        )

        conflicts: list[dict] = []
        for row in rows:
            vals_a = row.get("vals_a") or {}
            vals_b = row.get("vals_b") or {}
            for prop in single_props:
                va = vals_a.get(prop)
                vb = vals_b.get(prop)
                if va and vb and va != vb:
                    conflicts.append({
                        "type":        "property_conflict",
                        "entity":      row["entity"],
                        "entity_type": row["entity_type"],
                        "property":    prop,
                        "doc_a":       row["doc_a"],
                        "value_a":     va,
                        "doc_b":       row["doc_b"],
                        "value_b":     vb,
                        "constraint":  "single cardinality — only one value allowed",
                    })
        return conflicts

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_props(
        self,
        name: str,
        entity_type: str,
        props: dict,
    ) -> list[dict]:
        """Apply all rules for entity_type to the given property dict."""
        rules = self._rules.get(entity_type, {})
        issues: list[dict] = []

        for prop_name, rule in rules.items():
            value = props.get(prop_name)

            # Required check
            if rule.cardinality == "required" and (value is None or value == ""):
                issues.append({
                    "type":       "missing_required",
                    "entity":     name,
                    "entity_type": entity_type,
                    "property":   prop_name,
                    "value":      value,
                    "constraint": f"property '{prop_name}' is required for {entity_type}",
                })
                continue

            if value is None:
                continue

            # Allowed-values check
            if rule.allowed_values and str(value) not in rule.allowed_values:
                issues.append({
                    "type":       "invalid_value",
                    "entity":     name,
                    "entity_type": entity_type,
                    "property":   prop_name,
                    "value":      value,
                    "constraint": f"must be one of {sorted(rule.allowed_values)}",
                })

        return issues
