"""Read-only Cypher checks compiled from the canonical model (SHACL-to-Cypher subset).

The generated SHACL shapes validate RDF exports; this compiles the *same*
canonical model into Cypher queries that find violations in the property-graph
projection, so one set of rules is checked on both stores.  It covers exactly
what the generated shapes express:

* required properties (``sh:minCount`` on a datatype property);
* datatypes (``sh:datatype``), via Neo4j 5 ``valueType()``;
* relation endpoints (``sh:class``) and relation min/max counts;
* qualified relation counts (``sh:qualifiedValueShape``);
* key and composite-key uniqueness (which SHACL Core itself cannot check).

Every query is read-only, takes no parameters and returns ``rule``,
``element`` and ``node`` rows, one per violation.  Each check carries the
SHA-256 of the generated shapes it mirrors, so a run is tied to a model
version.  This in-house subset was chosen over adopting an external
SHACL-to-Cypher compiler (the evaluation is in
``docs/interoperability-and-curation.md``).  Parity with the RDF validator has
to be shown against a live Neo4j (``tests/e2e/test_live_cypher_checks.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from graphrag.semantic_model.compiler import compile_model
from graphrag.semantic_model.models import SemanticModel

_VALUE_TYPE = {
    "string": ("STRING",), "integer": ("INTEGER",), "decimal": ("FLOAT", "INTEGER"),
    "boolean": ("BOOLEAN",), "date": ("DATE",), "dateTime": ("ZONED DATETIME", "LOCAL DATETIME"),
}
_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH|DROP|LOAD|CALL)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CypherCheck:
    rule: str          # required | datatype | relation_endpoint | relation_min | relation_max | qualified_min |
                       # qualified_max | key | composite_key
    element: str
    query: str
    shapes_sha256: str


def _labels_of(model: SemanticModel, type_name: str) -> list[str]:
    return sorted(model.lpg_label(t, model.types[t]) for t in model.types
                  if not model.types[t].abstract and model.is_subtype(t, type_name))


def compile_cypher_checks(model: SemanticModel) -> list[CypherCheck]:
    shapes = hashlib.sha256(compile_model(model).shacl.encode("utf-8")).hexdigest()
    checks: list[CypherCheck] = []

    def add(rule: str, element: str, query: str) -> None:
        query = " ".join(query.split())
        if _WRITE.search(query):
            raise AssertionError(f"generated check for {element} is not read-only")
        checks.append(CypherCheck(rule, element, query, shapes))

    for name in sorted(model.types):
        spec = model.types[name]
        if spec.abstract:
            continue
        label = model.lpg_label(name, spec)
        for prop, prop_spec in sorted(model.effective_properties(name).items()):
            element = f"{name}.{prop}"
            if prop_spec.cardinality.minimum:
                add("required", element, f"MATCH (n:{label}) WHERE n.{prop} IS NULL "
                                          f"RETURN '{element}' AS element, 'required' AS rule, elementId(n) AS node")
            allowed = " OR ".join(f"valueType(n.{prop}) STARTS WITH '{t}'" for t in _VALUE_TYPE[prop_spec.datatype])
            add("datatype", element, f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL AND NOT ({allowed}) "
                                      f"RETURN '{element}' AS element, 'datatype' AS rule, elementId(n) AS node")
            if prop_spec.key:
                add("key", element, f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
                                     f"WITH n.{prop} AS value, collect(elementId(n)) AS nodes WHERE size(nodes) > 1 "
                                     f"UNWIND nodes AS node RETURN '{element}' AS element, 'key' AS rule, node")
        for members in model.composite_keys_for(name):
            element = f"{name}({', '.join(members)})"
            values = ", ".join(f"n.{m}" for m in members)
            present = " AND ".join(f"n.{m} IS NOT NULL" for m in members)
            add("composite_key", element, f"MATCH (n:{label}) WHERE {present} "
                                           f"WITH [{values}] AS value, collect(elementId(n)) AS nodes "
                                           f"WHERE size(nodes) > 1 UNWIND nodes AS node "
                                           f"RETURN '{element}' AS element, 'composite_key' AS rule, node")
        for relation_name, relation in sorted(model.relations_for(name).items()):
            rel = model.lpg_relation(relation_name, relation)
            element = f"{name}.{relation_name}"
            targets = _labels_of(model, relation.target)
            target_test = " OR ".join(f"m:{t}" for t in targets) or "false"
            add("relation_endpoint", element, f"MATCH (n:{label})-[:{rel}]->(m) WHERE NOT ({target_test}) "
                                               f"RETURN '{element}' AS element, 'relation_endpoint' AS rule, "
                                               f"elementId(n) AS node")
            low, high = relation.cardinality.minimum, relation.cardinality.maximum
            if low:
                add("relation_min", element, f"MATCH (n:{label}) WHERE COUNT {{ (n)-[:{rel}]->() }} < {low} "
                                              f"RETURN '{element}' AS element, 'relation_min' AS rule, "
                                              f"elementId(n) AS node")
            if high is not None:
                add("relation_max", element, f"MATCH (n:{label}) WHERE COUNT {{ (n)-[:{rel}]->() }} > {high} "
                                              f"RETURN '{element}' AS element, 'relation_max' AS rule, "
                                              f"elementId(n) AS node")
            for qualified in relation.qualified:
                q_labels = _labels_of(model, qualified.on_class)
                q_test = " OR ".join(f"m:{t}" for t in q_labels) or "false"
                q_element = f"{element}[{qualified.on_class}]"
                count = f"COUNT {{ (n)-[:{rel}]->(m) WHERE {q_test} }}"
                if qualified.cardinality.minimum:
                    add("qualified_min", q_element, f"MATCH (n:{label}) WHERE {count} < {qualified.cardinality.minimum} "
                                                     f"RETURN '{q_element}' AS element, 'qualified_min' AS rule, "
                                                     f"elementId(n) AS node")
                if qualified.cardinality.maximum is not None:
                    add("qualified_max", q_element, f"MATCH (n:{label}) WHERE {count} > {qualified.cardinality.maximum} "
                                                     f"RETURN '{q_element}' AS element, 'qualified_max' AS rule, "
                                                     f"elementId(n) AS node")
    return checks


__all__ = ["CypherCheck", "compile_cypher_checks"]
