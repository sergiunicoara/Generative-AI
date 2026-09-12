"""Execute an R2RML mapping into real RDF triples.

Companion to ``graphrag/ingestion/r2rml.py``'s ``r2rml_to_mapping()``, which
parses the same ``.r2rml.ttl`` files but produces a Neo4j-shaped
``RelationalGraphMapping`` (a single ``rdfs:label`` column plus
parent-triples-map joins) for the property-graph ingestion path. That model
cannot represent an arbitrary plain-column predicate (e.g. ``energy:status``
in ``ontology/mappings/energy-assets.r2rml.ttl``) -- ``r2rml_to_mapping()``
silently ignores any ``rr:predicateObjectMap`` that is neither an
``rdfs:label`` column nor a join.

This module is a direct execution of the R2RML mapping graph itself against
a relational source, producing an ``rdflib.Graph`` of real triples: no Neo4j
intermediary, no lossy narrowing. It is the actual "run the mapping" half of
R2RML that nothing in this codebase did before -- R2RML was previously only
ever parsed and validated (``r2rml_to_mapping()``) or hand-approximated
(``scripts/export_rdf.py`` serializes Neo4j, not R2RML output).

Deliberately narrow, same philosophy as ``r2rml.py``: unsupported R2RML
constructs raise ``R2RMLMappingError`` rather than being silently dropped or
guessed at. Supported per ``rr:TriplesMap``:

- ``rr:subjectMap`` with a single-column ``rr:template`` and one or more
  ``rr:class`` declarations (subject IRI + explicit ``rdf:type`` triples).
- ``rr:predicateObjectMap`` with ``rr:objectMap/rr:column`` -- a literal
  triple straight from the row.
- ``rr:predicateObjectMap`` with ``rr:objectMap/rr:parentTriplesMap`` +
  ``rr:joinCondition`` -- a triple whose object is the *parent* row's
  subject IRI. The parent ``rr:TriplesMap`` must already have been
  materialized (TriplesMaps are processed in file order; both shipped
  mappings already satisfy this) -- an unresolved forward reference raises
  rather than silently reordering or guessing.
- ``rr:predicateObjectMap`` with ``rr:objectMap/rr:constant`` -- a fixed
  value (typically an IRI, e.g. a provenance source tag) attached to every
  row unconditionally, independent of any column.

Anything else -- a multi-column template, ``rr:datatype``, ``rr:language``,
an object map that is not exactly one of ``rr:column``,
``rr:parentTriplesMap``, or ``rr:constant`` -- raises ``R2RMLMappingError``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF

from graphrag.ingestion.r2rml import RR, R2RMLMappingError, _one
from graphrag.ingestion.relational import TabularSourceConnector

# _identifier_from_template() in r2rml.py uses a greedy regex that, given a
# template with two or more {column} placeholders, silently matches the
# LAST one rather than rejecting the template -- fine for r2rml.py's own
# narrower callers where a multi-column template is already excluded by
# construction, but not safe to reuse here where materialize_r2rml() must
# itself reject that shape explicitly. This module does its own check.
_TEMPLATE_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _single_template_column(template: str, label: str) -> str:
    columns = _TEMPLATE_PLACEHOLDER.findall(template)
    if len(columns) != 1:
        raise R2RMLMappingError(f"{label} must contain exactly one identifier template {{column}}")
    return columns[0]


async def materialize_r2rml(
    mapping_path: str | Path,
    connector: TabularSourceConnector,
) -> Graph:
    """Read every ``rr:TriplesMap``'s source table via `connector`, apply its
    subject map and each predicate-object map, and return a populated
    ``rdflib.Graph`` of real triples.

    Raises ``R2RMLMappingError`` for any construct outside the supported
    subset documented on this module -- fail closed, never approximate.
    """
    mapping_graph = Graph()
    mapping_graph.parse(str(mapping_path), format="turtle")

    triples_maps = list(mapping_graph.subjects(RDF.type, RR.TriplesMap))
    if not triples_maps:
        raise R2RMLMappingError("no rr:TriplesMap found")

    out = Graph()
    # Maps a TriplesMap subject -> {row-identifier-value: subject IRI} for
    # every row already materialized under that map, so a join predicate can
    # resolve its object without re-reading the parent table.
    subjects_by_map: dict[Any, dict[str, URIRef]] = {}

    for triples_map in triples_maps:
        logical_table = _one(mapping_graph, triples_map, RR.logicalTable, "rr:logicalTable")
        table = str(_one(mapping_graph, logical_table, RR.tableName, "rr:tableName"))
        subject_map = _one(mapping_graph, triples_map, RR.subjectMap, "rr:subjectMap")
        template = str(_one(mapping_graph, subject_map, RR.template, "rr:subjectMap/rr:template"))
        id_column = _single_template_column(template, "rr:subjectMap/rr:template")
        rdf_classes = list(mapping_graph.objects(subject_map, RR["class"]))
        if not rdf_classes:
            raise R2RMLMappingError("rr:subjectMap requires at least one rr:class")

        pred_obj_maps = list(mapping_graph.objects(triples_map, RR.predicateObjectMap))
        # Validate every predicate-object map's shape up front, before
        # reading a single row -- a malformed mapping must fail before any
        # partial output is produced, matching r2rml.py's own philosophy.
        resolved_poms: list[tuple[URIRef, str, Any, Any]] = []
        for pom in pred_obj_maps:
            predicate = _one(mapping_graph, pom, RR.predicate, "rr:predicateObjectMap/rr:predicate")
            object_map = _one(mapping_graph, pom, RR.objectMap, "rr:predicateObjectMap/rr:objectMap")
            column = mapping_graph.value(object_map, RR.column)
            parent_map = mapping_graph.value(object_map, RR.parentTriplesMap)
            constant = mapping_graph.value(object_map, RR.constant)
            if mapping_graph.value(object_map, RR.datatype) is not None:
                raise R2RMLMappingError("rr:datatype is not supported")
            if mapping_graph.value(object_map, RR.language) is not None:
                raise R2RMLMappingError("rr:language is not supported")
            shapes = [v for v in (column, parent_map, constant) if v is not None]
            if len(shapes) != 1:
                raise R2RMLMappingError(
                    "a predicateObjectMap must be exactly one of rr:column, "
                    "rr:parentTriplesMap, or rr:constant"
                )
            if column is not None:
                resolved_poms.append((predicate, "column", str(column), None))
            elif constant is not None:
                # A fixed value attached to every row -- e.g. a provenance
                # IRI naming the source system, not derived from any column.
                resolved_poms.append((predicate, "constant", constant, None))
            elif parent_map is not None:
                if parent_map not in subjects_by_map:
                    raise R2RMLMappingError(
                        f"{table}: rr:parentTriplesMap references a TriplesMap not yet "
                        "materialized -- parent maps must precede children in file order"
                    )
                join = _one(mapping_graph, object_map, RR.joinCondition, "rr:joinCondition")
                child_column = str(_one(mapping_graph, join, RR.child, "rr:child"))
                parent_column = str(_one(mapping_graph, join, RR.parent, "rr:parent"))
                resolved_poms.append((predicate, "join", child_column, (parent_map, parent_column)))
            else:
                raise R2RMLMappingError(
                    "a predicateObjectMap's rr:objectMap must have rr:column, "
                    "rr:parentTriplesMap, or rr:constant"
                )

        rows = await connector.read_table(table)
        subjects_by_map[triples_map] = {}
        for row in rows:
            id_value = row.get(id_column)
            if id_value is None:
                raise R2RMLMappingError(f"{table}: row missing identifier column {id_column!r}")
            subject = URIRef(template.format(**{id_column: id_value}))
            subjects_by_map[triples_map][str(id_value)] = subject
            # R2RML permits more than one rr:class on a subject map.  Keeping
            # every declared type is important for a domain model where a
            # concrete WindTurbine is also an Asset, and avoids relying on a
            # triplestore's optional RDFS inference at query time.
            for rdf_class in rdf_classes:
                out.add((subject, RDF.type, rdf_class))

            for predicate, kind, key, extra in resolved_poms:
                if kind == "column":
                    value = row.get(key)
                    if value is None:
                        continue
                    out.add((subject, predicate, Literal(value)))
                elif kind == "constant":
                    out.add((subject, predicate, key))
                else:
                    child_value = row.get(key)
                    if child_value is None:
                        continue
                    parent_map, parent_column = extra
                    parent_subject = subjects_by_map[parent_map].get(str(child_value))
                    if parent_subject is None:
                        raise R2RMLMappingError(
                            f"{table}: join value {child_value!r} for {key!r} has no matching "
                            f"row in the parent table (joined on {parent_column!r})"
                        )
                    out.add((subject, predicate, parent_subject))

    return out


__all__ = ["materialize_r2rml"]
