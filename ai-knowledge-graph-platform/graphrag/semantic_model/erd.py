"""Reviewable ERD/frame view generated from the canonical semantic model.

Roadmap "P1 -- visual and tool-friendly semantic modelling", bullet 1:
"Generate a reviewable ERD/frame view from the canonical YAML, including
entities, slots/properties, inheritance, mixins, typed relations and
cardinality." YAML remains the canonical source; this is a review surface,
generated the same deterministic, atomically-replaced way as the OWL/SHACL/
Neo4j/diagnostics artifacts (see ``compiler.compile_to_disk``) -- never
hand-edited, never a second source of truth.

Mermaid (https://mermaid.js.org/syntax/classDiagram.html) was chosen over a
Graphviz/DOT export because GitHub, GitLab and most Markdown viewers render
```mermaid fenced blocks natively -- no separate rendering step, viewer, or
dependency is needed to review it.
"""

from __future__ import annotations

from graphrag.semantic_model.models import Cardinality, SemanticModel


def _cardinality_label(cardinality: Cardinality) -> str:
    if cardinality.exact is not None:
        return str(cardinality.exact)
    lower = cardinality.min if cardinality.min is not None else 0
    upper = str(cardinality.max) if cardinality.max is not None else "*"
    return f"{lower}..{upper}"


def render_erd_mermaid(model: SemanticModel) -> str:
    """Render ``model`` as a Mermaid ``classDiagram``.

    - Every type becomes a class, with its own (non-inherited) properties as
      typed attributes -- inherited/mixin properties are not repeated on
      every subtype, the same way the canonical YAML itself declares them
      once; ``<<abstract>>`` is stamped on abstract types.
    - ``extends`` becomes a UML inheritance arrow (``<|--``).
    - A mixin becomes its own ``<<mixin>>``-stamped class with a realization
      arrow (``..|>``) from every type that uses it -- Mermaid has no native
      "trait/mixin" arrow, so this is the closest standard notation that
      still reads as "not inheritance" at a glance.
    - Every relation becomes an association arrow labelled with its name and
      both ends' cardinality (``min..max``, or ``exact`` when set).
    """
    lines = ["classDiagram"]
    for mixin_name, mixin in sorted(model.mixins.items()):
        lines.append(f"    class {mixin_name} {{")
        lines.append("        <<mixin>>")
        for prop_name, prop in sorted(mixin.properties.items()):
            lines.append(f"        +{prop.datatype} {prop_name}")
        lines.append("    }")
    for type_name, spec in sorted(model.types.items()):
        lines.append(f"    class {type_name} {{")
        if spec.abstract:
            lines.append("        <<abstract>>")
        for prop_name, prop in sorted(spec.properties.items()):
            marker = "*" if prop.key else ("!" if prop.required else "")
            lines.append(f"        +{prop.datatype} {prop_name}{marker}")
        lines.append("    }")
        if spec.extends:
            if ":" in spec.extends:
                # An external/foreign parent (e.g. `prov:Entity`) rather
                # than a type declared in this model. A Mermaid class name
                # can't contain a colon, and drawing an inheritance arrow to
                # a class that's never declared would be a dangling
                # reference either way -- note it instead of breaking the
                # diagram or fabricating a box for a type this model
                # doesn't own.
                lines.append(f"    %% {type_name} extends external type {spec.extends}")
            else:
                lines.append(f"    {spec.extends} <|-- {type_name}")
        for mixin_name in spec.mixins:
            lines.append(f"    {type_name} ..|> {mixin_name} : mixin")
    for relation_name, relation in sorted(model.relations.items()):
        label = _cardinality_label(relation.cardinality)
        lines.append(
            f'    {relation.source} "1" --> "{label}" {relation.target} : {relation_name}',
        )
    return "\n".join(lines) + "\n"
