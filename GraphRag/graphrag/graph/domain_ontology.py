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

from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


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
    from graphrag.graph.inference_engine import InferenceRule

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
