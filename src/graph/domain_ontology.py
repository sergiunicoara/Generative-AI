# Ported from ai-knowledge-graph-platform (graphrag/graph/domain_ontology.py) — domain-agnostic as-is, unmodified except where noted.

"""Domain ontology loader — reads config/ontologies/*.yml and applies to registry/taxonomy.

This module bridges the YAML domain ontology files and the in-memory registries
(OntologyRegistry, TypeTaxonomy, ForwardChainingEngine) so domain-specific
knowledge models can be deployed without code changes.

Usage
-----
The ontology path is set in config/settings.yml → ontology.domain_ontology_path.
Loaded automatically by OntologyRegistry.load() and passed to TypeTaxonomy.load()
as extra_pairs.

Domain ontology files define:
  type_hierarchy     — [(child, parent)] SUBCLASS_OF extensions
  relation_rules     — domain/range constraints (extends built-in _RELATION_RULES)
  inference_rules    — Datalog rules for ForwardChainingEngine
  exclusive_state_pairs — contradiction detection pairs
  functional_relations  — single-valued relation constraints
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re

import structlog

log = structlog.get_logger(__name__)

_SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,49}$")
_ONTOLOGY_SECTIONS = {
    "type_hierarchy", "relation_rules", "inference_rules",
    "exclusive_state_pairs", "functional_relations",
}


class OntologyValidationError(ValueError):
    """Raised when an ontology cannot safely become the active schema."""

    def __init__(self, errors: list[str], source: str = "ontology"):
        self.errors = errors
        self.source = source
        super().__init__(f"{source} failed validation: {'; '.join(errors)}")


def validate_ontology_document(
    ontology: dict,
    *,
    previous: dict | None = None,
    source: str = "ontology",
) -> dict:
    """Validate lifecycle metadata and all graph-facing ontology sections.

    The returned report is deliberately plain data so it can be used by CI,
    YAML loaders, and future RDF/property-graph adapters.  A schema may only
    be activated through :func:`assert_valid_ontology`, which turns errors into
    a fail-fast exception.
    """
    errors: list[str] = []
    if not isinstance(ontology, dict):
        raise OntologyValidationError(["document must be a mapping"], source)

    meta = ontology.get("ontology")
    if not isinstance(meta, dict):
        errors.append("missing 'ontology' lifecycle metadata")
        meta = {}

    ontology_id = str(meta.get("id", "")).strip()
    version = str(meta.get("version", "")).strip()
    status = str(meta.get("status", "")).strip().lower()
    compatible_with = str(meta.get("compatible_with", "")).strip()
    if not ontology_id:
        errors.append("ontology.id is required")
    if not _SEMVER_RE.match(version):
        errors.append("ontology.version must use MAJOR.MINOR.PATCH")
    if status not in {"draft", "active", "deprecated"}:
        errors.append("ontology.status must be draft, active, or deprecated")
    if not compatible_with:
        errors.append("ontology.compatible_with is required")

    deprecated_types = meta.get("deprecated_types", [])
    deprecated_relations = meta.get("deprecated_relations", [])
    if not isinstance(deprecated_types, list) or not isinstance(deprecated_relations, list):
        errors.append("deprecated_types and deprecated_relations must be lists")
        deprecated_types = []
        deprecated_relations = []
    migration_map = dict(ontology.get("migration_map") or {})
    migration_map.update(dict(meta.get("migration_map") or {}))
    migration_map = {str(k).upper(): str(v).upper() for k, v in migration_map.items()}
    for old in deprecated_types:
        if str(old).upper() not in {k.upper() for k in migration_map}:
            errors.append(f"deprecated type '{old}' has no migration_map entry")
    for old in deprecated_relations:
        if str(old).upper() not in migration_map:
            errors.append(f"deprecated relation '{old}' has no migration_map entry")

    hierarchy = ontology.get("type_hierarchy", [])
    if not isinstance(hierarchy, list):
        errors.append("type_hierarchy must be a list")
        hierarchy = []
    pairs: list[tuple[str, str]] = []
    for index, entry in enumerate(hierarchy):
        if isinstance(entry, dict):
            child = str(entry.get("child", "")).upper()
            parent = str(entry.get("parent", "")).upper()
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            child, parent = str(entry[0]).upper(), str(entry[1]).upper()
        else:
            errors.append(f"type_hierarchy[{index}] must define child and parent")
            continue
        if not child or not parent:
            errors.append(f"type_hierarchy[{index}] has an empty child or parent")
        else:
            pairs.append((child, parent))

    # A hierarchy cycle makes taxonomy traversal non-terminating.
    parents = {child: parent for child, parent in pairs}
    for start in parents:
        seen: set[str] = set()
        node = start
        while node in parents:
            if node in seen:
                errors.append(f"type_hierarchy contains a cycle at '{node}'")
                break
            seen.add(node)
            node = parents[node]

    relation_rules = ontology.get("relation_rules", {})
    if not isinstance(relation_rules, dict):
        errors.append("relation_rules must be a mapping")
        relation_rules = {}
    relation_names = set()
    for relation, rule in relation_rules.items():
        rel = str(relation).upper()
        relation_names.add(rel)
        if not _RELATION_RE.match(rel):
            errors.append(f"relation name '{relation}' is not UPPER_SNAKE_CASE")
        if not isinstance(rule, dict):
            errors.append(f"relation_rules.{relation} must be a mapping")
            continue
        if not rule.get("domain") or not rule.get("target"):
            errors.append(f"relation_rules.{relation} needs non-empty domain and target")

    for index, rule in enumerate(ontology.get("inference_rules", []) or []):
        if not isinstance(rule, dict) or not rule.get("name") or not rule.get("relation"):
            errors.append(f"inference_rules[{index}] needs name and relation")
        elif not _RELATION_RE.match(str(rule["relation"]).upper()):
            errors.append(f"inference_rules[{index}].relation is invalid")

    unknown_sections = set(ontology) - _ONTOLOGY_SECTIONS - {"ontology", "domain", "version", "migration_map"}
    # Unknown keys are warnings rather than errors to allow provenance fields.
    warnings = [f"unknown top-level section '{key}'" for key in sorted(unknown_sections)]

    if previous:
        prev_meta = previous.get("ontology", {}) if isinstance(previous, dict) else {}
        prev_version = str(prev_meta.get("version", "0.0.0"))
        if _SEMVER_RE.match(version) and _SEMVER_RE.match(prev_version):
            if version.split(".")[0] != prev_version.split(".")[0]:
                errors.append(f"incompatible major version change {prev_version} -> {version}")

    return {
        "valid": not errors,
        "id": ontology_id,
        "version": version,
        "status": status,
        "relations": sorted(relation_names),
        "type_pairs": len(pairs),
        "warnings": warnings,
        "errors": errors,
    }


def assert_valid_ontology(
    ontology: dict,
    *,
    previous: dict | None = None,
    source: str = "ontology",
) -> dict:
    report = validate_ontology_document(ontology, previous=previous, source=source)
    if not report["valid"]:
        raise OntologyValidationError(report["errors"], source)
    return report


def validate_ontology_yaml(path: str | Path, previous: dict | None = None) -> dict:
    """Strictly load and validate an ontology file for CI/deployment gates."""
    ontology = load_domain_ontology(path)
    if not ontology:
        raise OntologyValidationError(["file is missing or empty"], str(path))
    return assert_valid_ontology(ontology, previous=previous, source=str(path))


def get_ontology_path_for_tenant(
    tenant: str,
    ontologies_dir: str | Path = "config/ontologies",
) -> Path | None:
    """
    Resolve a tenant name to a domain ontology file by naming convention.

    Looks for ``{ontologies_dir}/{tenant}*.yml`` (e.g. tenant "automotive"
    matches "automotive_iatf.yml", tenant "aerospace" matches
    "aerospace_regulatory.yml"). Returns ``None`` if no match exists — callers
    should treat this the same as a missing ontology (built-in rules only).

    This lets each tenant ship its own ontology file without any code change:
    add ``config/ontologies/<tenant>_<name>.yml`` and it is picked up
    automatically.
    """
    d = Path(ontologies_dir)
    if not d.is_dir():
        return None
    matches = sorted(d.glob(f"{tenant.lower()}*.yml"))
    return matches[0] if matches else None


def load_domain_ontology(path: str | Path) -> dict:
    """
    Load a domain ontology YAML file and return the parsed dict.

    Returns an empty dict if the file does not exist or cannot be parsed —
    callers should treat missing domain ontologies as "use built-in rules only."
    """
    p = Path(path)
    if not p.exists():
        log.warning("domain_ontology.file_not_found", path=str(p))
        return {}
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        log.info("domain_ontology.loaded", path=str(p))
        return data or {}
    except Exception as exc:
        log.error("domain_ontology.load_error", path=str(p), error=str(exc))
        return {}


def get_type_hierarchy_pairs(ontology: dict) -> list[tuple[str, str]]:
    """
    Extract (child, parent) type pairs from the ontology's type_hierarchy section.

    These are passed as ``extra_pairs`` to ``TypeTaxonomy.load()`` to extend the
    default hierarchy without modifying source code.

    Example YAML::

        type_hierarchy:
          - [AIRWORTHINESS_DIRECTIVE, REGULATION]
          - [REGULATION,              CONCEPT]
    """
    pairs: list[tuple[str, str]] = []
    for entry in ontology.get("type_hierarchy", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            child, parent = str(entry[0]).upper(), str(entry[1]).upper()
            pairs.append((child, parent))
        elif isinstance(entry, dict):
            child  = str(entry.get("child",  "")).upper()
            parent = str(entry.get("parent", "")).upper()
            if child and parent:
                pairs.append((child, parent))
    return pairs


@lru_cache(maxsize=32)
def _type_hierarchy_pairs_for_path(path_str: str) -> tuple[tuple[str, str], ...]:
    """Cached load+parse — ontology files don't change at runtime, and this
    would otherwise re-read+re-parse the same YAML on every extracted chunk."""
    ontology = load_domain_ontology(path_str)
    return tuple(get_type_hierarchy_pairs(ontology))


def get_entity_types_for_tenant(tenant: str, base_types: list[str]) -> list[str]:
    """
    Return the entity type vocabulary to offer the extraction LLM: the generic
    ``base_types`` plus every domain-specific type from the tenant's
    ``config/ontologies/{tenant}*.yml`` type_hierarchy, if one exists.

    Rationale: the entity MERGE key is (name, type, tenant) — see
    neo4j_client.merge_entities_batch. Two same-name entities with different
    meanings (e.g. "Apple" the company vs. "Apple" the fruit) stay distinct
    nodes only if the extractor assigns them different types. A flat generic
    type list (ORG/PRODUCT/CONCEPT/...) makes that collision likely whenever
    both meanings would plausibly get the same generic type. Domain-specific
    types (TECHNOLOGY_COMPANY vs. AGRICULTURAL_PRODUCT) make the collision
    far less likely because the LLM has a more specific type to reach for.

    Falls back to ``base_types`` unchanged when the tenant has no ontology
    file (e.g. tenant "default").
    """
    path = get_ontology_path_for_tenant(tenant)
    domain_types: set[str] = set()
    if path is not None:
        for child, _parent in _type_hierarchy_pairs_for_path(str(path)):
            domain_types.add(child)
    # Base types first (stable, familiar), then domain types sorted for a
    # deterministic prompt — extraction is cached keyed on the full prompt
    # text (see Extractor._generate), so a stable order matters for cache hits.
    return list(dict.fromkeys(list(base_types) + sorted(domain_types)))


def get_relation_rules(ontology: dict) -> dict[str, dict]:
    """
    Extract relation_rules from the ontology dict.

    Returns a dict of::

        {
          "SUPERSEDES": {"domain": ["REGULATION", ...], "target": [...], "note": "..."},
          ...
        }

    Passed to ``OntologyRegistry.add_domain_range_rules()``.
    """
    return ontology.get("relation_rules", {}) or {}


def get_inference_rules(ontology: dict) -> list[dict]:
    """
    Extract inference_rules list for ForwardChainingEngine.

    Each dict maps to an ``InferenceRule`` dataclass field.
    """
    return ontology.get("inference_rules", []) or []


def get_exclusive_state_pairs(ontology: dict) -> list[tuple[str, str]]:
    """Extract exclusive state pairs for contradiction detection."""
    pairs: list[tuple[str, str]] = []
    for entry in ontology.get("exclusive_state_pairs", []):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((str(entry[0]).upper(), str(entry[1]).upper()))
    return pairs


def get_functional_relations(ontology: dict) -> list[str]:
    """Extract functional (one-to-one) relation names."""
    return [str(r).upper() for r in ontology.get("functional_relations", [])]


def build_inference_rules_from_ontology(ontology: dict):
    """
    Convert inference_rules YAML entries to InferenceRule dataclass instances.

    Returns a list of InferenceRule objects ready to pass to
    ``ForwardChainingEngine(rules=...)`` or ``engine.add_rule(...)``.
    """
    from src.graph.inference_engine import InferenceRule

    rules = []
    for entry in get_inference_rules(ontology):
        try:
            rule = InferenceRule(
                name             = entry["name"],
                rule_type        = entry.get("rule_type", "transitivity"),
                relation         = entry["relation"].upper(),
                derived_relation = entry.get("derived_relation", "").upper(),
                body_relation_2  = entry.get("body_relation_2", "").upper(),
                max_depth        = int(entry.get("max_depth", 3)),
                confidence_decay = float(entry.get("confidence_decay", 0.9)),
            )
            rules.append(rule)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("domain_ontology.invalid_inference_rule",
                        entry=entry, error=str(exc))
    return rules
