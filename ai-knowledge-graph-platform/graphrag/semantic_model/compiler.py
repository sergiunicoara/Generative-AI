"""Deterministic OWL, SHACL, and Neo4j compilation with loss diagnostics."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from graphrag.semantic_model.models import PropertySpec, SemanticModel, SemanticModelError, load_model

_XSD = {
    "string": "xsd:string", "integer": "xsd:integer", "decimal": "xsd:decimal",
    "boolean": "xsd:boolean", "date": "xsd:date", "dateTime": "xsd:dateTime",
}
_GENERATED = "# DO NOT EDIT — generated from {source}.\n"


class Target(str, Enum):
    OWL = "owl"
    SHACL = "shacl"
    NEO4J = "neo4j"


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
}


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


@dataclass(frozen=True)
class Compilation:
    owl: str
    shacl: str
    neo4j: str
    diagnostics: tuple[Diagnostic, ...]


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
    lines.append("")
    for type_name in sorted(model.types):
        keys = [name for name, prop in sorted(model.effective_properties(type_name).items()) if prop.key]
        if keys:
            lines.append(f"energy:{type_name} owl:hasKey ( {' '.join(f'energy:{key}' for key in keys)} ) .")
        for prop_name, prop in sorted(model.effective_properties(type_name).items()):
            for kind, value in _cardinality_terms(prop):
                predicate = {"exact": "owl:cardinality", "min": "owl:minCardinality", "max": "owl:maxCardinality"}[kind]
                lines.append(f"energy:{type_name} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty energy:{prop_name} ; {predicate} \"{value}\"^^xsd:nonNegativeInteger ] .")
        for relation_name, relation in sorted(model.relations_for(type_name).items()):
            card = PropertySpec(datatype="string", cardinality=relation.cardinality)
            for kind, value in _cardinality_terms(card):
                predicate = {"exact": "owl:cardinality", "min": "owl:minCardinality", "max": "owl:maxCardinality"}[kind]
                lines.append(f"energy:{type_name} rdfs:subClassOf [ a owl:Restriction ; owl:onProperty energy:{relation_name} ; {predicate} \"{value}\"^^xsd:nonNegativeInteger ] .")
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
        if constraints:
            lines.append(f"energy:{type_name}Shape a sh:NodeShape ; sh:targetClass energy:{type_name} ;")
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
    if model.unknown_property_policy == "reject":
        result.append(Diagnostic("warning", "SHACL_UNKNOWN_PROPERTY_RUNTIME", Target.SHACL, path, None,
            "unknown_property_policy",
            "Generated shapes remain open to preserve existing Energy RDF metadata and do not reject every undeclared predicate.",
            "approximated", "Reject unknown LPG properties in the shared mutation validator; use curated closed SHACL shapes where required."))
    return tuple(result)


def compile_model(model: SemanticModel) -> Compilation:
    return Compilation(_compile_owl(model), _compile_shacl(model), _compile_neo4j(model), _diagnostics(model))


def _outputs(model: SemanticModel, output_dir: Path | None) -> dict[str, Path]:
    if output_dir is not None:
        return {"owl": output_dir / "ontology.ttl", "shacl": output_dir / "shapes.ttl",
                "neo4j": output_dir / "neo4j.cypher", "diagnostics": output_dir / "diagnostics.json"}
    assert model.source_path is not None
    root = model.source_path.parents[2]
    return {name: root / value for name, value in model.artifacts.model_dump().items()}


def compile_to_disk(
    model_path: str | Path, *, output_dir: str | Path | None = None,
    check: bool = False, fail_on_unenforceable: bool = False,
) -> dict[str, Path]:
    model = load_model(model_path)
    compiled = compile_model(model)
    if fail_on_unenforceable:
        forbidden = [item for item in compiled.diagnostics if item.fidelity == "unenforceable"]
        if forbidden:
            codes = ", ".join(sorted({item.code for item in forbidden}))
            raise SemanticModelError(f"forbidden unenforceable semantics: {codes}")
    contents = {"owl": compiled.owl, "shacl": compiled.shacl, "neo4j": compiled.neo4j,
                "diagnostics": json.dumps({"model": model.id, "version": model.version,
                    "capabilities": {
                        target.value: sorted(capability.value for capability in capabilities)
                        for target, capabilities in TARGET_CAPABILITIES.items()
                    },
                    "diagnostics": [asdict(item) for item in compiled.diagnostics]}, indent=2, sort_keys=True) + "\n"}
    paths = _outputs(model, Path(output_dir) if output_dir else None)
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
