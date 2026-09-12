"""Execute an RML mapping (JSON source) into real RDF triples.

Companion to ``graphrag/ingestion/r2rml_rdf.py``'s ``materialize_r2rml()``
for tabular (SQL) sources -- same idea, "run the mapping for real, no
Neo4j intermediary, no hand-written duplicate," applied to RML's JSON-source
shape instead of R2RML's table shape.

Deliberately narrow, same philosophy as ``materialize_r2rml()``: a small,
self-built executor for exactly the construct shapes this repo's one
shipped RML mapping (``ontology/mappings/energy-observations.rml.ttl``)
needs, not a general-purpose engine or a new third-party dependency to run
one file. No RML processor or JSONPath library is installed anywhere in
this repo (confirmed while scoping this) -- this executor needs neither:
the only reference formulation supported is a top-level JSON array
iterator (``$[*]``) with direct-key ``rml:reference`` lookups, which is
ordinary dict/list indexing, not real JSONPath evaluation. A mapping that
declares anything beyond that shape is rejected, not silently
approximated.

Supported per ``rr:TriplesMap``:

- ``rml:logicalSource`` with ``rml:source`` (a JSON file, resolved against
  ``base_dir``), ``rml:referenceFormulation ql:JSONPath``,
  ``rml:iterator "$[*]"`` -- the source must parse to a JSON array; each
  element is iterated as one row-equivalent "item" (a JSON object).
- ``rr:subjectMap`` with a single-column ``rr:template`` and ``rr:class``.
- ``rr:predicateObjectMap``/``rr:objectMap`` in exactly one of three forms:
  ``rml:reference "key"`` (a literal straight from ``item[key]``, with an
  optional ``rr:datatype``), ``rr:template "..."`` (an IRI built by
  substituting item fields into the template -- e.g. a reference to
  another mapping's subject-IRI shape, without a real cross-map join), or
  ``rr:constant <iri>`` (a fixed IRI, unconditionally attached to every
  row).

Anything else -- a different ``rml:referenceFormulation``/``rml:iterator``,
``rr:column`` (a tabular/R2RML-only construct), ``rr:parentTriplesMap``,
a multi-column template, ``rr:language`` -- raises ``RMLMappingError``.

Note: this module reuses ``graphrag/ingestion/r2rml.py``'s ``_one()``
helper for basic "this predicate must occur exactly once" RDF-shape
checks, shared low-level parsing plumbing rather than RML-specific logic
(the same cross-module reuse ``r2rml_rdf.py`` already does). Its errors
surface as ``R2RMLMappingError`` even from this module; every
RML-construct-specific check here raises ``RMLMappingError``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from graphrag.ingestion.r2rml import RR, _one

RML = Namespace("http://semweb.mmlab.be/ns/rml#")
QL = Namespace("http://semweb.mmlab.be/ns/ql#")

_SUPPORTED_ITERATOR = "$[*]"

# Mirrors r2rml_rdf.py's _TEMPLATE_PLACEHOLDER -- duplicated rather than
# imported across the module boundary for this small piece of logic,
# matching this repo's own established convention (see
# scripts/run_retrieval_quality_eval.py's _rank_key comment for the
# precedent: duplicate a small private helper rather than reach into
# another module's own).
_TEMPLATE_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class RMLMappingError(RuntimeError):
    """The mapping uses an RML/R2RML construct outside this executor's
    deliberately narrow supported subset -- see the module docstring."""


def _single_template_column(template: str, label: str) -> str:
    columns = _TEMPLATE_PLACEHOLDER.findall(template)
    if len(columns) != 1:
        raise RMLMappingError(f"{label} must contain exactly one identifier template {{column}}")
    return columns[0]


def _substitute_template(template: str, item: dict[str, Any]) -> str:
    def _replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in item:
            raise RMLMappingError(f"template {template!r} references missing key {key!r}")
        return str(item[key])

    return _TEMPLATE_PLACEHOLDER.sub(_replace, template)


async def materialize_rml(mapping_path: str | Path, base_dir: str | Path) -> Graph:
    """Read every ``rr:TriplesMap``'s JSON source (resolved against
    ``base_dir``), apply its subject map and each predicate-object map, and
    return a populated ``rdflib.Graph`` of real triples.

    Async to match ``materialize_r2rml()``'s call shape, even though this
    executor's own work is synchronous (a JSON file read, no I/O worth
    awaiting) -- keeps both mapping-execution entry points swappable at a
    call site without the caller needing to know which kind of mapping it is.

    Raises ``RMLMappingError`` for any construct outside the supported
    subset documented on this module -- fail closed, never approximate.
    """
    base = Path(base_dir)
    mapping_graph = Graph()
    mapping_graph.parse(str(mapping_path), format="turtle")

    triples_maps = list(mapping_graph.subjects(RDF.type, RR.TriplesMap))
    if not triples_maps:
        raise RMLMappingError("no rr:TriplesMap found")

    out = Graph()

    for triples_map in triples_maps:
        logical_source = _one(mapping_graph, triples_map, RML.logicalSource, "rml:logicalSource")
        source = str(_one(mapping_graph, logical_source, RML.source, "rml:source"))
        formulation = mapping_graph.value(logical_source, RML.referenceFormulation)
        if formulation != QL.JSONPath:
            raise RMLMappingError(f"{source}: only ql:JSONPath is supported as rml:referenceFormulation")
        iterator = str(mapping_graph.value(logical_source, RML.iterator) or "")
        if iterator != _SUPPORTED_ITERATOR:
            raise RMLMappingError(
                f"{source}: only the top-level array iterator {_SUPPORTED_ITERATOR!r} is "
                f"supported, got {iterator!r}"
            )

        source_path = base / source
        try:
            items = json.loads(source_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RMLMappingError(f"{source}: JSON source not found under {base}") from exc
        if not isinstance(items, list):
            raise RMLMappingError(f"{source}: expected a top-level JSON array for {_SUPPORTED_ITERATOR!r}")

        subject_map = _one(mapping_graph, triples_map, RR.subjectMap, "rr:subjectMap")
        template = str(_one(mapping_graph, subject_map, RR.template, "rr:subjectMap/rr:template"))
        id_key = _single_template_column(template, "rr:subjectMap/rr:template")
        rdf_class = _one(mapping_graph, subject_map, RR["class"], "rr:subjectMap/rr:class")

        # Validate every predicate-object map's shape up front, before
        # reading a single item -- same "fail before any partial output"
        # discipline as materialize_r2rml().
        resolved_poms: list[tuple[Any, str, Any]] = []
        for pom in mapping_graph.objects(triples_map, RR.predicateObjectMap):
            predicate = _one(mapping_graph, pom, RR.predicate, "rr:predicateObjectMap/rr:predicate")
            object_map = _one(mapping_graph, pom, RR.objectMap, "rr:predicateObjectMap/rr:objectMap")
            reference = mapping_graph.value(object_map, RML.reference)
            obj_template = mapping_graph.value(object_map, RR.template)
            constant = mapping_graph.value(object_map, RR.constant)
            if mapping_graph.value(object_map, RR.column) is not None:
                raise RMLMappingError(
                    "rr:column is a tabular (R2RML) construct; use rml:reference for a JSON source"
                )
            if mapping_graph.value(object_map, RR.parentTriplesMap) is not None:
                raise RMLMappingError("rr:parentTriplesMap joins are not supported by this executor")
            if mapping_graph.value(object_map, RR.language) is not None:
                raise RMLMappingError("rr:language is not supported")
            shapes = [v for v in (reference, obj_template, constant) if v is not None]
            if len(shapes) != 1:
                raise RMLMappingError(
                    "a predicateObjectMap's rr:objectMap must have exactly one of "
                    "rml:reference, rr:template, or rr:constant"
                )
            if reference is not None:
                datatype = mapping_graph.value(object_map, RR.datatype)
                resolved_poms.append((predicate, "reference", (str(reference), datatype)))
            elif obj_template is not None:
                resolved_poms.append((predicate, "template", str(obj_template)))
            else:
                resolved_poms.append((predicate, "constant", constant))

        for item in items:
            if not isinstance(item, dict):
                raise RMLMappingError(f"{source}: every array element must be a JSON object")
            if id_key not in item:
                raise RMLMappingError(f"{source}: item missing identifier key {id_key!r}")
            subject = URIRef(_substitute_template(template, item))
            out.add((subject, RDF.type, rdf_class))

            for predicate, kind, payload in resolved_poms:
                if kind == "reference":
                    key, datatype = payload
                    if key not in item or item[key] is None:
                        continue
                    literal = Literal(item[key], datatype=datatype) if datatype is not None else Literal(item[key])
                    out.add((subject, predicate, literal))
                elif kind == "template":
                    out.add((subject, predicate, URIRef(_substitute_template(payload, item))))
                else:  # constant
                    out.add((subject, predicate, payload))

    return out


__all__ = ["RMLMappingError", "materialize_rml"]
