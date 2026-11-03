"""Deterministic OWL, SHACL, and Neo4j compilation with loss diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from graphrag.semantic_model.erd import render_erd_mermaid
from graphrag.semantic_model.gql import render_gql
from graphrag.semantic_model.models import (
    PropertySpec, SemanticModel, SemanticModelError, load_model, statement_class,
)

_XSD = {
    "string": "xsd:string", "integer": "xsd:integer", "decimal": "xsd:decimal",
    "boolean": "xsd:boolean", "date": "xsd:date", "dateTime": "xsd:dateTime",
}
_GENERATED = "# DO NOT EDIT — generated from {source}.\n"


class Target(str, Enum):
    OWL = "owl"
    SHACL = "shacl"
    NEO4J = "neo4j"
    GQL = "gql"


class Capability(str, Enum):
    CLASSES = "classes"
    INHERITANCE = "inheritance"
    PROPERTIES = "properties"
    CARDINALITY = "cardinality"
    KEYS = "keys"
    DEPRECATION = "deprecation"
    DATATYPES = "datatypes"
    NODE_KIND = "node_kind"
    CLOSED_WORLD_VALIDATION = "closed_world_validation"
    UNIQUE_KEYS = "unique_keys"


TARGET_CAPABILITIES: dict[Target, frozenset[Capability]] = {
    Target.OWL: frozenset({
        Capability.CLASSES, Capability.INHERITANCE, Capability.PROPERTIES,
        Capability.CARDINALITY, Capability.KEYS, Capability.DEPRECATION,
        Capability.DATATYPES,
    }),
    Target.SHACL: frozenset({
        Capability.CLASSES, Capability.PROPERTIES, Capability.CARDINALITY,
        Capability.DATATYPES, Capability.NODE_KIND,
        Capability.CLOSED_WORLD_VALIDATION,
    }),
    Target.NEO4J: frozenset({Capability.UNIQUE_KEYS}),
    # ISO GQL graph types: node and edge types with typed properties. No
    # cardinality, key or deprecation constraints (see gql.py).
    Target.GQL: frozenset({Capability.CLASSES, Capability.PROPERTIES, Capability.DATATYPES}),
}


def target_capabilities(model: SemanticModel) -> dict[Target, frozenset[Capability]]:
    """Capabilities of the targets this model actually compiles to (GQL only
    when the model configures a GQL artifact), so models that do not opt in
    keep byte-identical artifacts."""
    return {t: c for t, c in TARGET_CAPABILITIES.items() if t != Target.GQL or model.artifacts.gql}


@dataclass(frozen=True)
class Diagnostic:
    severity: Literal["info", "warning", "error"]
    code: str
    target: Target
    model_path: str
    line: int | None
    element: str
    message: str
    fidelity: Literal["preserved", "approximated", "unenforceable"]
    runtime_control: str | None = None
    # Typed target outcome, derived from fidelity and the compensating control
    # (see `_outcome`); `fidelity` stays for existing consumers.
    outcome: str = ""

    def __post_init__(self) -> None:
        if not self.outcome:
            object.__setattr__(self, "outcome", _outcome(self))


OUTCOMES = ("enforced", "translated", "runtime-mitigated", "warning", "unsupported")


def _outcome(item: "Diagnostic") -> str:
    """What the target does with a rule it cannot enforce natively.

    * ``warning``: the semantics are carried but not enforced (open-world
      OWL cardinality); another target is the gate.
    * ``runtime-mitigated``: not enforced by the target; a named runtime
      control compensates.
    * ``translated``: converted to the closest construct (flattened, expanded
      or reified), no enforcement lost silently.
    * ``unsupported``: neither enforced nor compensated.
    ``enforced`` never appears on a diagnostic (a loss report); it is the
    positive outcome recorded in the capability matrix.
    """
    if item.fidelity == "preserved":
        return "warning"
    if item.runtime_control:
        return "runtime-mitigated"
    return "translated" if item.fidelity == "approximated" else "unsupported"


@dataclass(frozen=True)
class Compilation:
    owl: str
    shacl: str
    neo4j: str
    diagnostics: tuple[Diagnostic, ...]
    gql: str = ""


def _prefixes(model: SemanticModel, *, shacl: bool = False) -> list[str]:
    required = {"energy": model.namespace, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#", "xsd": "http://www.w3.org/2001/XMLSchema#"}
    required.update({"sh": "http://www.w3.org/ns/shacl#"} if shacl else {
        "dcterms": "http://purl.org/dc/terms/", "owl": "http://www.w3.org/2002/07/owl#",
    })
    prefixes = {**model.prefixes, **required}
    return [f"@prefix {name}: <{iri}> ." for name, iri in sorted(prefixes.items())]


def _cardinality_terms(spec: PropertySpec) -> list[tuple[str, int]]:
    terms: list[tuple[str, int]] = []
    if spec.cardinality.exact is not None:
        terms.append(("exact", spec.cardinality.exact))
    else:
        if spec.cardinality.min is not None:
            terms.append(("min", spec.cardinality.min))
        if spec.cardinality.max is not None:
            terms.append(("max", spec.cardinality.max))
    return terms


def _resource(value: str, model: SemanticModel) -> str:
    prefix, separator, _ = value.partition(":")
    known = set(model.prefixes) | {"energy", "rdf", "rdfs", "xsd", "owl", "dcterms"}
    return value if separator and prefix in known else f"<{value}>"


def _compile_owl(model: SemanticModel) -> str:
    source = model.source_path.name if model.source_path else "semantic model"
    lines = [_GENERATED.format(source=source).rstrip(), *_prefixes(model), "",
             f'energy: a owl:Ontology ; rdfs:label "{model.label}" ; owl:versionInfo "{model.version}" .', ""]
    for name in sorted(model.types):
        spec = model.types[name]
        parts = ["a owl:Class", f'rdfs:label "{spec.label or name}"']
        if spec.description:
            parts.append(f'rdfs:comment {json.dumps(spec.description)}')
        if spec.extends:
            parent = spec.extends if ":" in spec.extends else f"energy:{spec.extends}"
            parts.append(f"rdfs:subClassOf {parent}")
        if spec.deprecated:
            parts.append('owl:deprecated "true"^^xsd:boolean')
        if spec.replaced_by:
            parts.append(f"dcterms:isReplacedBy energy:{spec.replaced_by}")
        lines.append(f"energy:{name} " + " ;\n  ".join(parts) + " .")
    lines.append("")
    defining_properties: dict[str, tuple[str, PropertySpec]] = {}
    for type_name, type_spec in model.types.items():
        for prop_name, prop in type_spec.properties.items():
            defining_properties.setdefault(prop_name, (type_name, prop))
    for mixin_name, mixin in model.mixins.items():
        for prop_name, prop in mixin.properties.items():
            defining_properties.setdefault(prop_name, (mixin_name, prop))
    for prop_name, (owner, prop) in sorted(defining_properties.items()):
        domain = f"energy:{owner}" if owner in model.types else "owl:Thing"
        parts = ["a owl:DatatypeProperty", f"rdfs:domain {domain}", f"rdfs:range {_XSD[prop.datatype]}"]
        if prop.label:
            parts.append(f"rdfs:label {json.dumps(prop.label)}")
        if prop.description:
            parts.append(f"rdfs:comment {json.dumps(prop.description)}")
        if prop.vocabulary:
            parts.append(f"rdfs:seeAlso {_resource(prop.vocabulary, model)}")
        lines.append(f"energy:{prop_name} " + " ; ".join(parts) + " .")
    for name, relation in sorted(model.relations.items()):
        parts = ["a owl:ObjectProperty", f"rdfs:domain energy:{relation.source}",
                 f"rdfs:range energy:{relation.target}"]
        if relation.label:
            parts.append(f"rdfs:label {json.dumps(relation.label)}")
        if relation.description:
            parts.append(f"rdfs:comment {json.dumps(relation.description)}")
        if relation.deprecated:
            parts.append('owl:deprecated "true"^^xsd:boolean')
        if relation.replaced_by:
            parts.append(f"dcterms:isReplacedBy energy:{relation.replaced_by}")
        lines.append(f"energy:{name} " + " ; ".join(parts) + " .")
    # Relationship attributes: OWL has no edge properties, so each relation that
    # carries attributes gets a reification class holding them (rdf:Statement).
    for name, relation in sorted(model.relations.items()):
        if not relation.properties:
            continue
        cls = statement_class(name)
        lines.append(f'energy:{cls} a owl:Class ; rdfs:subClassOf rdf:Statement ; '
                     f'rdfs:label "{name} statement" .')
        for attribute, prop in sorted(relation.properties.items()):
            lines.append(f"energy:{attribute} a owl:DatatypeProperty ; rdfs:domain energy:{cls} ; "
                         f"rdfs:range {_XSD[prop.datatype]} .")
    lines.append("")
    for type_name in sorted(model.types):
        keys = [name for name, prop in sorted(model.effective_properties(type_name).items()) if prop.key]
        if keys:
            lines.append(f"energy:{type_name} owl:hasKey ( {' '.join(f'energy:{key}' for key in keys)} ) .")
        for members in model.composite_keys_for(type_name):
            lines.append(f"energy:{type_name} owl:hasKey ( {' '.join(f'energy:{key}' for key in members)} ) .")
        for prop_name, prop in sorted(model.effective_properties(type_name).items()):
            for kind, value in _cardinality_terms(prop):
                predicate = {"exact": "owl:cardinality", "min": "owl:minCardinality", "max": "owl:maxCardinality"}[kind]
                lines.append(f"energy:{type_name} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty energy:{prop_name} ; {predicate} \"{value}\"^^xsd:nonNegativeInteger ] .")
        for relation_name, relation in sorted(model.relations_for(type_name).items()):
            card = PropertySpec(datatype="string", cardinality=relation.cardinality)
            for kind, value in _cardinality_terms(card):
                predicate = {"exact": "owl:cardinality", "min": "owl:minCardinality", "max": "owl:maxCardinality"}[kind]
                lines.append(f"energy:{type_name} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty energy:{relation_name} ; {predicate} \"{value}\"^^xsd:nonNegativeInteger ] .")
            for qualified in relation.qualified:
                card = PropertySpec(datatype="string", cardinality=qualified.cardinality)
                for kind, value in _cardinality_terms(card):
                    predicate = {"exact": "owl:qualifiedCardinality", "min": "owl:minQualifiedCardinality",
                                 "max": "owl:maxQualifiedCardinality"}[kind]
                    lines.append(f"energy:{type_name} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty energy:{relation_name} ; owl:onClass energy:{qualified.on_class} ; {predicate} \"{value}\"^^xsd:nonNegativeInteger ] .")
    return "\n".join(lines).rstrip() + "\n"


def _compile_shacl(model: SemanticModel) -> str:
    source = model.source_path.name if model.source_path else "semantic model"
    lines = [_GENERATED.format(source=source).rstrip(), *_prefixes(model, shacl=True), ""]
    for type_name in sorted(model.types):
        constraints: list[str] = []
        for prop_name, prop in sorted(model.effective_properties(type_name).items()):
            terms = [f"sh:path energy:{prop_name}", f"sh:datatype {_XSD[prop.datatype]}"]
            if prop.cardinality.minimum is not None:
                terms.append(f"sh:minCount {prop.cardinality.minimum}")
            if prop.cardinality.maximum is not None:
                terms.append(f"sh:maxCount {prop.cardinality.maximum}")
            constraints.append("  sh:property [ " + " ; ".join(terms) + " ]")
        for relation_name, relation in sorted(model.relations_for(type_name).items()):
            terms = [f"sh:path energy:{relation_name}", "sh:nodeKind sh:IRI", f"sh:class energy:{relation.target}"]
            if relation.cardinality.minimum is not None:
                terms.append(f"sh:minCount {relation.cardinality.minimum}")
            if relation.cardinality.maximum is not None:
                terms.append(f"sh:maxCount {relation.cardinality.maximum}")
            constraints.append("  sh:property [ " + " ; ".join(terms) + " ]")
            for qualified in relation.qualified:
                terms = [f"sh:path energy:{relation_name}",
                         f"sh:qualifiedValueShape [ sh:class energy:{qualified.on_class} ]"]
                if qualified.cardinality.minimum is not None:
                    terms.append(f"sh:qualifiedMinCount {qualified.cardinality.minimum}")
                if qualified.cardinality.maximum is not None:
                    terms.append(f"sh:qualifiedMaxCount {qualified.cardinality.maximum}")
                constraints.append("  sh:property [ " + " ; ".join(terms) + " ]")
        if constraints:
            lines.append(f"energy:{type_name}Shape a sh:NodeShape ; sh:targetClass energy:{type_name} ;")
            lines.append(" ;\n".join(constraints) + " .")
    # Relationship attributes live on a reified statement: its endpoints keep
    # the declared source and target types, its attributes keep their datatypes.
    for relation_name, relation in sorted(model.relations.items()):
        if not relation.properties:
            continue
        cls = statement_class(relation_name)
        constraints = [
            f"  sh:property [ sh:path rdf:subject ; sh:nodeKind sh:IRI ; sh:class energy:{relation.source} ; sh:minCount 1 ; sh:maxCount 1 ]",
            f"  sh:property [ sh:path rdf:object ; sh:nodeKind sh:IRI ; sh:class energy:{relation.target} ; sh:minCount 1 ; sh:maxCount 1 ]",
        ]
        for attribute, prop in sorted(relation.properties.items()):
            terms = [f"sh:path energy:{attribute}", f"sh:datatype {_XSD[prop.datatype]}"]
            if prop.cardinality.minimum is not None:
                terms.append(f"sh:minCount {prop.cardinality.minimum}")
            if prop.cardinality.maximum is not None:
                terms.append(f"sh:maxCount {prop.cardinality.maximum}")
            constraints.append("  sh:property [ " + " ; ".join(terms) + " ]")
        lines.append(f"energy:{cls}Shape a sh:NodeShape ; sh:targetClass energy:{cls} ;")
        lines.append(" ;\n".join(constraints) + " .")
    return "\n".join(lines).rstrip() + "\n"


def _compile_neo4j(model: SemanticModel) -> str:
    source = model.source_path.name if model.source_path else "semantic model"
    lines = [_GENERATED.format(source=source).rstrip(), "// Neo4j schema can enforce keys/indexes; see diagnostics.json for runtime-only semantics."]
    for type_name in sorted(model.types):
        label = model.lpg_label(type_name, model.types[type_name])
        for prop_name, prop in sorted(model.effective_properties(type_name).items()):
            if prop.key:
                cname = f"energy_{label.lower()}_{prop_name.lower()}_unique"
                lines.append(f"CREATE CONSTRAINT {cname} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop_name} IS UNIQUE;")
        for members in model.composite_keys_for(type_name):
            cname = f"energy_{label.lower()}_{'_'.join(m.lower() for m in members)}_unique"
            columns = ", ".join(f"n.{m}" for m in members)
            lines.append(f"CREATE CONSTRAINT {cname} IF NOT EXISTS FOR (n:{label}) REQUIRE ({columns}) IS UNIQUE;")
    return "\n".join(lines).rstrip() + "\n"


def _diagnostics(model: SemanticModel) -> tuple[Diagnostic, ...]:
    # Keep committed diagnostics byte-identical across checkout locations.
    path = model.source_path.name if model.source_path else ""
    result: list[Diagnostic] = []
    for name, spec in sorted(model.types.items()):
        line = model.source_lines.get(name)
        if spec.abstract:
            result.append(Diagnostic("warning", "OWL_ABSTRACT_RUNTIME", Target.OWL, path, line, name,
                "OWL has no native abstract-class constraint that forbids direct instances.", "unenforceable",
                "Reject direct instances at the governed mutation or publication boundary."))
            result.append(Diagnostic("warning", "SHACL_ABSTRACT_RUNTIME", Target.SHACL, path, line, name,
                "The generated SHACL Core shape does not forbid direct instances of an abstract class.", "unenforceable",
                "Reject abstract classes in the governed mutation validator."))
            result.append(Diagnostic("warning", "NEO4J_ABSTRACT_RUNTIME", Target.NEO4J, path, line, name,
                "Neo4j labels do not prevent direct instances of an abstract semantic type.", "unenforceable",
                "Reject abstract labels in the mutation validator."))
        if spec.mixins:
            result.append(Diagnostic("info", "NEO4J_MIXIN_EXPANDED", Target.NEO4J, path, line, name,
                "Mixin properties are expanded onto the concrete label; mixin identity is not native Neo4j schema.", "approximated"))
        if spec.extends:
            # Distinct from the abstract-runtime and mixin diagnostics above:
            # this is plain single inheritance. OWL preserves it natively via
            # rdfs:subClassOf (no diagnostic needed there); SHACL builds one
            # flat NodeShape per type from its own effective properties with
            # no sh:node shape chaining, and Neo4j has no label-hierarchy
            # enforcement -- both lose the is-a relationship itself, even
            # though the inherited properties are still individually present.
            result.append(Diagnostic("info", "SHACL_INHERITANCE_FLATTENED", Target.SHACL, path, line, name,
                "The generated SHACL shape has no shape-inheritance link to the parent type; "
                "inherited properties are present but the is-a relationship itself is not represented.",
                "approximated", "Preserve the type hierarchy through the canonical OWL export for lineage checks."))
            result.append(Diagnostic("info", "NEO4J_INHERITANCE_FLATTENED", Target.NEO4J, path, line, name,
                "Neo4j has no native label-hierarchy constraint; inherited properties are expanded onto "
                "the concrete label with no structural link back to the parent type.",
                "approximated", "Preserve the type hierarchy through the canonical OWL export for lineage checks."))
        for prop_name, prop in sorted(model.effective_properties(name).items()):
            if prop.cardinality.minimum:
                result.append(Diagnostic("warning", "OWL_OPEN_WORLD_MINIMUM", Target.OWL, path, model.source_lines.get(prop_name), f"{name}.{prop_name}",
                    "OWL cardinality describes semantics under the open-world assumption; it does not reject missing input data.", "preserved",
                    "Use the generated SHACL shape as the closed-world publication gate."))
            if prop.required and not prop.key:
                result.append(Diagnostic("warning", "NEO4J_REQUIRED_PROPERTY_RUNTIME", Target.NEO4J, path, model.source_lines.get(prop_name), f"{name}.{prop_name}",
                    "Portable Neo4j community constraints cannot enforce this required property.", "unenforceable",
                    "Validate the property before the mutation and monitor out-of-band writes."))
            result.append(Diagnostic("warning", "NEO4J_DATATYPE_RUNTIME", Target.NEO4J, path,
                model.source_lines.get(prop_name), f"{name}.{prop_name}",
                f"Neo4j schema does not enforce the canonical {prop.datatype} datatype for this property.",
                "unenforceable", "Validate values in the shared mutation validator."))
            if prop.cardinality.maximum is not None:
                result.append(Diagnostic("warning", "NEO4J_PROPERTY_CARDINALITY_RUNTIME", Target.NEO4J, path,
                    model.source_lines.get(prop_name), f"{name}.{prop_name}",
                    "Neo4j property schema has no native value-count limit; a maximum-cardinality "
                    "property can silently accept more values than the model allows.",
                    "unenforceable", "Validate value count in the shared mutation validator."))
            if prop.key:
                result.append(Diagnostic("warning", "SHACL_KEY_RUNTIME", Target.SHACL, path,
                    model.source_lines.get(prop_name), f"{name}.{prop_name}",
                    "SHACL Core cardinality does not enforce graph-wide key uniqueness.", "unenforceable",
                    "Enforce uniqueness in the selected persistence transaction."))
    for name, relation in sorted(model.relations.items()):
        result.append(Diagnostic("warning", "NEO4J_RELATION_ENDPOINT_RUNTIME", Target.NEO4J, path,
            model.source_lines.get(name), name,
            "Neo4j schema constraints do not enforce inherited relation source and target types.",
            "unenforceable", "Validate relation endpoints in the shared mutation validator."))
        if relation.cardinality.minimum is not None or relation.cardinality.maximum is not None:
            result.append(Diagnostic("warning", "NEO4J_RELATION_CARDINALITY_RUNTIME", Target.NEO4J, path, model.source_lines.get(name), name,
                "Neo4j schema constraints do not enforce relationship cardinality.", "unenforceable",
                "Check inside the same write transaction; application-only check-then-write can race."))
        if relation.cardinality.minimum:
            # Same open-world caveat OWL_OPEN_WORLD_MINIMUM already reports
            # for property cardinality: a minimum on a *relation* is equally
            # descriptive, not enforcing, under OWL's open-world assumption.
            result.append(Diagnostic("warning", "OWL_OPEN_WORLD_MINIMUM_RELATION", Target.OWL, path,
                model.source_lines.get(name), name,
                "OWL cardinality describes semantics under the open-world assumption; it does not reject a relation with too few instances.",
                "preserved", "Use the generated SHACL shape as the closed-world publication gate."))
    for name, spec in sorted(model.types.items()):
        for members in spec.composite_keys:
            result.append(Diagnostic("warning", "SHACL_COMPOSITE_KEY_RUNTIME", Target.SHACL, path,
                model.source_lines.get(name), f"{name}({', '.join(members)})",
                "SHACL Core cardinality does not enforce graph-wide uniqueness of a property combination.",
                "unenforceable", "Enforce uniqueness in the selected persistence transaction."))
    for name, relation in sorted(model.relations.items()):
        line = model.source_lines.get(name)
        for qualified in relation.qualified:
            element = f"{name}[{qualified.on_class}]"
            result.append(Diagnostic("warning", "NEO4J_QUALIFIED_CARDINALITY_RUNTIME", Target.NEO4J, path, line,
                element, "Neo4j schema constraints cannot count relationships by target type.",
                "unenforceable", "Check inside the same write transaction; application-only check-then-write can race."))
            if qualified.cardinality.minimum:
                result.append(Diagnostic("warning", "OWL_OPEN_WORLD_MINIMUM_QUALIFIED", Target.OWL, path, line,
                    element, "OWL qualified cardinality describes semantics under the open-world assumption; "
                    "it does not reject too few matching relations.", "preserved",
                    "Use the generated SHACL shape as the closed-world publication gate."))
        if relation.properties:
            result.append(Diagnostic("info", "OWL_RELATION_PROPERTY_REIFIED", Target.OWL, path, line, name,
                "OWL has no edge properties; attributes sit on a reified rdf:Statement class.", "approximated"))
            result.append(Diagnostic("info", "SHACL_RELATION_PROPERTY_REIFIED", Target.SHACL, path, line, name,
                "SHACL validates relation attributes on the reified statement, not on the edge itself.",
                "approximated", "Materialise the statement node when exporting edge attributes to RDF."))
            result.append(Diagnostic("warning", "NEO4J_RELATION_PROPERTY_RUNTIME", Target.NEO4J, path, line, name,
                "Neo4j community schema constraints do not enforce relationship property presence or datatype.",
                "unenforceable", "Validate relationship properties in the shared mutation validator."))
    if model.unknown_property_policy == "reject":
        result.append(Diagnostic("warning", "SHACL_UNKNOWN_PROPERTY_RUNTIME", Target.SHACL, path, None,
            "unknown_property_policy",
            "Generated shapes remain open to preserve existing RDF metadata and do not reject every undeclared predicate.",
            "approximated", "Reject unknown LPG properties in the shared mutation validator; use curated closed SHACL shapes where required."))
        result.append(Diagnostic("warning", "NEO4J_UNKNOWN_PROPERTY_RUNTIME", Target.NEO4J, path, None,
            "unknown_property_policy",
            "Neo4j has no schema-level closed-world constraint; nothing rejects an undeclared property written onto a node.",
            "unenforceable", "Reject unknown LPG properties in the shared mutation validator."))
    return tuple(result)


def _gql_diagnostics(model: SemanticModel) -> tuple[Diagnostic, ...]:
    """Losses specific to the GQL graph-type rendering (only when configured)."""
    path = model.source_path.name if model.source_path else ""
    result: list[Diagnostic] = []
    for name, spec in sorted(model.types.items()):
        line = model.source_lines.get(name)
        if spec.extends:
            result.append(Diagnostic("info", "GQL_INHERITANCE_FLATTENED", Target.GQL, path, line, name,
                "The GQL rendering copies inherited properties onto each concrete node type; "
                "the is-a relationship itself is not represented.", "approximated"))
        if spec.mixins:
            result.append(Diagnostic("info", "GQL_MIXIN_EXPANDED", Target.GQL, path, line, name,
                "The GQL rendering copies mixin properties onto the concrete node type; "
                "mixin identity is not represented.", "approximated"))
        for prop_name, prop in sorted(model.effective_properties(name).items()):
            if prop.key:
                result.append(Diagnostic("warning", "GQL_KEY_UNSUPPORTED", Target.GQL, path,
                    model.source_lines.get(prop_name), f"{name}.{prop_name}",
                    "GQL graph types carry no key or uniqueness constraint.", "unenforceable",
                    "Enforce uniqueness in the selected persistence transaction."))
        for members in spec.composite_keys:
            result.append(Diagnostic("warning", "GQL_COMPOSITE_KEY_UNSUPPORTED", Target.GQL, path, line,
                f"{name}({', '.join(members)})",
                "GQL graph types carry no key or uniqueness constraint.", "unenforceable",
                "Enforce uniqueness in the selected persistence transaction."))
    for name, relation in sorted(model.relations.items()):
        bounded = (relation.cardinality.minimum is not None or relation.cardinality.maximum is not None)
        if bounded:
            result.append(Diagnostic("warning", "GQL_CARDINALITY_UNSUPPORTED", Target.GQL, path,
                model.source_lines.get(name), name,
                "GQL graph types carry no relationship cardinality.", "unenforceable",
                "Check inside the same write transaction; application-only check-then-write can race."))
    return tuple(result)


def compile_model(model: SemanticModel) -> Compilation:
    diagnostics = _diagnostics(model)
    gql = ""
    if model.artifacts.gql:
        gql = render_gql(model)
        diagnostics = (*diagnostics, *_gql_diagnostics(model))
    return Compilation(_compile_owl(model), _compile_shacl(model), _compile_neo4j(model), diagnostics, gql)


def _render_diagnostics_report(model: SemanticModel, compiled: Compilation) -> str:
    """Human-readable counterpart to the machine-readable `diagnostics.json`
    (roadmap "P0 — target capability matrix and loss diagnostics"): the same
    capability matrix and per-rule loss diagnostics, rendered as Markdown for
    a reviewer who isn't going to parse JSON by eye."""
    source = model.source_path.name if model.source_path else "semantic model"
    lines = [
        f"<!-- DO NOT EDIT — generated from {source}. -->",
        f"# {model.label} ({model.id} v{model.version}) — schema capability diagnostics",
        "",
        "For each canonical rule, whether OWL, SHACL, and Neo4j enforce it natively, "
        "how it degrades where they don't, and what compensates. The canonical "
        "semantic model remains authoritative; this report makes constraint loss "
        "visible rather than silently weakening governance.",
        "",
        "## Target capabilities",
        "",
        "| Target | Supported capabilities |",
        "| --- | --- |",
    ]
    for target, capabilities in target_capabilities(model).items():
        lines.append(f"| {target.value} | {', '.join(sorted(c.value for c in capabilities))} |")
    lines += [
        "",
        f"## Diagnostics ({len(compiled.diagnostics)})",
        "",
        "| Severity | Target | Code | Element | Fidelity | Outcome | Message | Mitigation | Location |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted(compiled.diagnostics, key=lambda d: (d.target.value, d.code, d.element)):
        location = f"{item.model_path}:{item.line}" if item.line else item.model_path
        mitigation = item.runtime_control or "—"
        lines.append(
            f"| {item.severity} | {item.target.value} | {item.code} | {item.element} | "
            f"{item.fidelity} | {item.outcome} | {item.message} | {mitigation} | {location} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _outputs(model: SemanticModel, output_dir: Path | None) -> dict[str, Path]:
    if output_dir is not None:
        return {"owl": output_dir / "ontology.ttl", "shacl": output_dir / "shapes.ttl",
                "neo4j": output_dir / "neo4j.cypher", "diagnostics": output_dir / "diagnostics.json",
                "diagnostics_report": output_dir / "diagnostics.md", "erd": output_dir / "erd.mmd",
                "capability_matrix": output_dir / "capability-matrix.json",
                **({"gql": output_dir / "graph-type.gql"} if model.artifacts.gql else {})}
    assert model.source_path is not None
    root = model.source_path.parents[2]
    return {name: root / value for name, value in model.artifacts.model_dump().items() if value is not None}


def compile_to_disk(
    model_path: str | Path, *, output_dir: str | Path | None = None,
    check: bool = False, fail_on_unenforceable: bool = False, fail_on_error: bool = False,
) -> dict[str, Path]:
    model = load_model(model_path)
    compiled = compile_model(model)
    if fail_on_unenforceable:
        forbidden = [item for item in compiled.diagnostics if item.fidelity == "unenforceable"]
        if forbidden:
            codes = ", ".join(sorted({item.code for item in forbidden}))
            raise SemanticModelError(f"forbidden unenforceable semantics: {codes}")
    if fail_on_error:
        # The CI-facing gate roadmap bullet 3 asks for: "fail on errors, but
        # allow reviewed warnings and informational downgrades". `severity`
        # (independent of `fidelity`) is the axis meant for exactly this —
        # no diagnostic emits "error" today, so this is a currently-dormant,
        # non-breaking gate that starts working the moment a future rule
        # needs to hard-fail compilation rather than just warn.
        errors = [item for item in compiled.diagnostics if item.severity == "error"]
        if errors:
            codes = ", ".join(sorted({item.code for item in errors}))
            raise SemanticModelError(f"error-severity diagnostics: {codes}")
    paths = _outputs(model, Path(output_dir) if output_dir else None)
    contents = {"owl": compiled.owl, "shacl": compiled.shacl, "neo4j": compiled.neo4j,
                "diagnostics": json.dumps({"model": model.id, "version": model.version,
                    "capabilities": {
                        target.value: sorted(capability.value for capability in capabilities)
                        for target, capabilities in target_capabilities(model).items()
                    },
                    "diagnostics": [asdict(item) for item in compiled.diagnostics]}, indent=2, sort_keys=True) + "\n"}
    if "diagnostics_report" in paths:
        contents["diagnostics_report"] = _render_diagnostics_report(model, compiled)
    if "erd" in paths:
        contents["erd"] = render_erd_mermaid(model)
    if "capability_matrix" in paths:
        from graphrag.semantic_model.matrix import build_capability_matrix

        contents["capability_matrix"] = json.dumps(build_capability_matrix(
            model, compiled.diagnostics, tuple(target_capabilities(model))), indent=2, sort_keys=True) + "\n"
    if "gql" in paths:
        contents["gql"] = compiled.gql
    drift = [name for name, path in paths.items() if not path.exists() or path.read_text(encoding="utf-8") != contents[name]]
    if check:
        if drift:
            raise SemanticModelError("generated artifacts are stale: " + ", ".join(drift))
        return paths
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(contents[name])
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
    return paths
