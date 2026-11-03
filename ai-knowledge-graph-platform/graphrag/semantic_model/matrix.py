"""Full rule x target capability matrix with a typed outcome per cell.

``diagnostics.json`` is a loss report: it lists only what a target cannot
enforce natively.  The matrix is its positive counterpart.  Every rule the
model declares is listed once per target, with one of five outcomes:

* ``enforced``: the target rejects violations natively;
* ``translated``: converted to the closest construct (flattened, expanded,
  reified), with no enforcement claimed;
* ``runtime-mitigated``: not enforced by the target; a named runtime control
  compensates (see the diagnostic's ``runtime_control``);
* ``warning``: the semantics are carried but not enforced by this target;
* ``unsupported``: neither enforced nor compensated.

A cell starts from the static capability of the target for that kind of rule
(``BASE``) and is overridden by the compiler's own diagnostic whenever one
exists for the same target and element, so the matrix can never claim more
than the diagnostics admit.  ``CODE_RULE`` maps every diagnostic code to the
rule kind it speaks about; a test fails when a code is unmapped.
"""

from __future__ import annotations

from graphrag.semantic_model.compiler import Diagnostic, Target
from graphrag.semantic_model.models import SemanticModel

RULES = (
    "class", "inheritance", "mixin", "abstract", "property", "datatype", "required", "max_cardinality",
    "key", "composite_key", "relation", "relation_cardinality", "qualified_cardinality",
    "relation_attribute", "deprecation", "closed_world",
)

# (OWL, SHACL, Neo4j, GQL)
_BASE: dict[str, tuple[str, str, str, str]] = {
    "class": ("enforced", "enforced", "translated", "enforced"),
    "inheritance": ("enforced", "translated", "translated", "translated"),
    "mixin": ("translated", "translated", "translated", "translated"),
    "abstract": ("runtime-mitigated", "runtime-mitigated", "runtime-mitigated", "translated"),
    "property": ("enforced", "enforced", "runtime-mitigated", "enforced"),
    "datatype": ("enforced", "enforced", "runtime-mitigated", "enforced"),
    "required": ("warning", "enforced", "runtime-mitigated", "enforced"),
    "max_cardinality": ("warning", "enforced", "runtime-mitigated", "unsupported"),
    "key": ("warning", "runtime-mitigated", "enforced", "unsupported"),
    "composite_key": ("warning", "runtime-mitigated", "enforced", "unsupported"),
    "relation": ("warning", "enforced", "runtime-mitigated", "enforced"),
    "relation_cardinality": ("warning", "enforced", "runtime-mitigated", "unsupported"),
    "qualified_cardinality": ("warning", "enforced", "runtime-mitigated", "unsupported"),
    "relation_attribute": ("translated", "translated", "runtime-mitigated", "enforced"),
    "deprecation": ("translated", "unsupported", "unsupported", "unsupported"),
    "closed_world": ("unsupported", "runtime-mitigated", "runtime-mitigated", "runtime-mitigated"),
}
_ORDER = (Target.OWL, Target.SHACL, Target.NEO4J, Target.GQL)

CODE_RULE = {
    "OWL_ABSTRACT_RUNTIME": "abstract", "SHACL_ABSTRACT_RUNTIME": "abstract", "NEO4J_ABSTRACT_RUNTIME": "abstract",
    "NEO4J_MIXIN_EXPANDED": "mixin", "GQL_MIXIN_EXPANDED": "mixin",
    "SHACL_INHERITANCE_FLATTENED": "inheritance", "NEO4J_INHERITANCE_FLATTENED": "inheritance",
    "GQL_INHERITANCE_FLATTENED": "inheritance",
    "OWL_OPEN_WORLD_MINIMUM": "required", "NEO4J_REQUIRED_PROPERTY_RUNTIME": "required",
    "NEO4J_DATATYPE_RUNTIME": "datatype", "NEO4J_PROPERTY_CARDINALITY_RUNTIME": "max_cardinality",
    "SHACL_KEY_RUNTIME": "key", "GQL_KEY_UNSUPPORTED": "key",
    "SHACL_COMPOSITE_KEY_RUNTIME": "composite_key", "GQL_COMPOSITE_KEY_UNSUPPORTED": "composite_key",
    "NEO4J_RELATION_ENDPOINT_RUNTIME": "relation",
    "NEO4J_RELATION_CARDINALITY_RUNTIME": "relation_cardinality",
    "OWL_OPEN_WORLD_MINIMUM_RELATION": "relation_cardinality",
    "GQL_CARDINALITY_UNSUPPORTED": "relation_cardinality",
    "NEO4J_QUALIFIED_CARDINALITY_RUNTIME": "qualified_cardinality",
    "OWL_OPEN_WORLD_MINIMUM_QUALIFIED": "qualified_cardinality",
    "OWL_RELATION_PROPERTY_REIFIED": "relation_attribute", "SHACL_RELATION_PROPERTY_REIFIED": "relation_attribute",
    "NEO4J_RELATION_PROPERTY_RUNTIME": "relation_attribute",
    "SHACL_UNKNOWN_PROPERTY_RUNTIME": "closed_world", "NEO4J_UNKNOWN_PROPERTY_RUNTIME": "closed_world",
}


def _rule_instances(model: SemanticModel) -> list[tuple[str, str, int | None]]:
    """Every (element, rule, line) the model declares, in a stable order."""
    rows: list[tuple[str, str, int | None]] = []
    line = model.source_lines.get
    for name in sorted(model.types):
        spec = model.types[name]
        rows.append((name, "class", line(name)))
        if spec.extends:
            rows.append((name, "inheritance", line(name)))
        if spec.mixins:
            rows.append((name, "mixin", line(name)))
        if spec.abstract:
            rows.append((name, "abstract", line(name)))
        if spec.deprecated:
            rows.append((name, "deprecation", line(name)))
        for prop_name, prop in sorted(model.effective_properties(name).items()):
            element = f"{name}.{prop_name}"
            rows.append((element, "property", line(prop_name)))
            rows.append((element, "datatype", line(prop_name)))
            if prop.cardinality.minimum:
                rows.append((element, "required", line(prop_name)))
            if prop.cardinality.maximum is not None:
                rows.append((element, "max_cardinality", line(prop_name)))
            if prop.key:
                rows.append((element, "key", line(prop_name)))
        for members in model.types[name].composite_keys:
            rows.append((f"{name}({', '.join(members)})", "composite_key", line(name)))
    for name, relation in sorted(model.relations.items()):
        rows.append((name, "relation", line(name)))
        if relation.cardinality.minimum is not None or relation.cardinality.maximum is not None:
            rows.append((name, "relation_cardinality", line(name)))
        for qualified in relation.qualified:
            rows.append((f"{name}[{qualified.on_class}]", "qualified_cardinality", line(name)))
        if relation.properties:
            rows.append((name, "relation_attribute", line(name)))
        if relation.deprecated:
            rows.append((name, "deprecation", line(name)))
    if model.unknown_property_policy == "reject":
        rows.append(("unknown_property_policy", "closed_world", None))
    return rows


def build_capability_matrix(
    model: SemanticModel, diagnostics: tuple[Diagnostic, ...], targets: tuple[Target, ...],
) -> dict:
    """The matrix as a JSON-ready dict (deterministic ordering)."""
    overrides: dict[tuple[str, str, str], str] = {}
    for item in diagnostics:
        rule = CODE_RULE.get(item.code)
        if rule is not None:
            overrides[(item.target.value, item.element, rule)] = item.outcome
    rules = []
    for element, rule, line in _rule_instances(model):
        cells = {}
        for target in targets:
            base = _BASE[rule][_ORDER.index(target)]
            cells[target.value] = overrides.get((target.value, element, rule), base)
        rules.append({"element": element, "rule": rule, "line": line, "outcomes": cells})
    summary = {t.value: {} for t in targets}
    for row in rules:
        for target, outcome in row["outcomes"].items():
            summary[target][outcome] = summary[target].get(outcome, 0) + 1
    return {
        "model": model.id, "version": model.version,
        "targets": [t.value for t in targets],
        "outcomes": ["enforced", "translated", "runtime-mitigated", "warning", "unsupported"],
        "summary": {t: dict(sorted(counts.items())) for t, counts in summary.items()},
        "rules": rules,
    }
