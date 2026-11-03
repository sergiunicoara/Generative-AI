"""ISO GQL graph-type rendering of the canonical semantic model.

An interoperability surface, never a source of truth: the YAML model stays
canonical (see ADR on targets as projections).  The output follows the
graph-type constructs of ISO/IEC 39075 (``CREATE GRAPH TYPE``, node types with
labels and typed properties, directed edge types) as that standard is
generally documented.  **No GQL engine is installed or run in this repository,
so the rendering is not validated against one**; treat it as a review and
hand-off artifact until a target engine has accepted it.

What GQL graph types do not express here is reported, not dropped: inheritance
is flattened (inherited and mixin properties are copied onto each concrete
type), and keys, cardinalities and deprecation are not part of the rendering
(see the ``GQL_*`` diagnostics emitted by the compiler).  Abstract types are
not instantiable and are omitted; an edge declared on an abstract endpoint is
expanded to its concrete subtypes.
"""

from __future__ import annotations

from graphrag.semantic_model.models import SemanticModel

_GQL_TYPES = {
    "string": "STRING", "integer": "INT64", "decimal": "DECIMAL",
    "boolean": "BOOLEAN", "date": "DATE", "dateTime": "ZONED DATETIME",
}


def _properties(props) -> str:
    parts = []
    for name, prop in sorted(props.items()):
        suffix = " NOT NULL" if prop.cardinality.minimum else ""
        parts.append(f"{name} {_GQL_TYPES[prop.datatype]}{suffix}")
    return ", ".join(parts)


def concrete_subtypes(model: SemanticModel, name: str) -> list[str]:
    return sorted(t for t, spec in model.types.items() if not spec.abstract and model.is_subtype(t, name))


def render_gql(model: SemanticModel) -> str:
    source = model.source_path.name if model.source_path else "semantic model"
    graph_type = model.id.replace("-", "_") + "_type"
    lines = [
        f"-- DO NOT EDIT — generated from {source}.",
        "-- ISO GQL graph-type rendering (interoperability surface). Not executed by any GQL engine here.",
        f"CREATE GRAPH TYPE {graph_type} AS {{",
    ]
    entries: list[str] = []
    for name in sorted(model.types):
        spec = model.types[name]
        if spec.abstract:
            continue
        label = model.lpg_label(name, spec)
        entries.append(f"  ({name} :{label} {{{_properties(model.effective_properties(name))}}})")
    for relation_name, relation in sorted(model.relations.items()):
        rel_label = model.lpg_relation(relation_name, relation)
        sources = concrete_subtypes(model, relation.source)
        targets = concrete_subtypes(model, relation.target)
        for src in sources:
            for tgt in targets:
                src_label = model.lpg_label(src, model.types[src])
                tgt_label = model.lpg_label(tgt, model.types[tgt])
                entries.append(
                    f"  (:{src_label})-[{relation_name}_{src}_{tgt} :{rel_label} "
                    f"{{{_properties(relation.properties)}}}]->(:{tgt_label})")
    lines.append(",\n".join(entries))
    lines.append("}")
    return "\n".join(lines) + "\n"
