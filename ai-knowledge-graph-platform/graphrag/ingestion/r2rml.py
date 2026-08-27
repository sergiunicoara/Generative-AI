"""Small, validated R2RML-to-local-mapping adapter and federated OBDA runner.

R2RML remains the portable mapping source of truth.  The adapter supports the
safe, operational subset needed by this platform: ``rr:tableName``, subject
templates with one identifier column, ``rr:class``, ``rdfs:label`` object
columns, and parent-triples-map joins.  Unsupported constructs are rejected
before a source is read; silently approximating a virtual mapping would make
lineage claims incorrect.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from graphrag.ingestion.relational import (
    EntityTableMapping, RelationTableMapping, RelationalGraphIngestor,
    RelationalGraphMapping,
)

RR = Namespace("http://www.w3.org/ns/r2rml#")
_TEMPLATE_COLUMN = re.compile(r"^.*\{([A-Za-z_][A-Za-z0-9_]*)\}.*$")


class R2RMLMappingError(ValueError):
    """The mapping uses semantics the materialized ingestion path cannot prove."""


def _one(graph: Graph, subject, predicate, label: str):
    values = list(graph.objects(subject, predicate))
    if len(values) != 1:
        raise R2RMLMappingError(f"{label} must occur exactly once")
    return values[0]


def _local_name(value: URIRef) -> str:
    name = str(value).rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()
    if not normalized:
        raise R2RMLMappingError(f"cannot derive a safe local name from {value}")
    return normalized


def _identifier_from_template(value: object, label: str) -> str:
    matched = _TEMPLATE_COLUMN.match(str(value))
    if not matched:
        raise R2RMLMappingError(f"{label} must contain exactly one identifier template {{column}}")
    return matched.group(1)


def r2rml_to_mapping(
    source: str | Path,
    *,
    mapping_id: str,
    version: str,
    source_id: str,
    tenant: str,
    ontology_version: str = "r2rml/v1",
) -> RelationalGraphMapping:
    graph = Graph()
    graph.parse(str(source), format="turtle")
    triples_maps = list(graph.subjects(RDF.type, RR.TriplesMap))
    if not triples_maps:
        raise R2RMLMappingError("no rr:TriplesMap found")

    entities: list[EntityTableMapping] = []
    entity_by_map: dict[Any, EntityTableMapping] = {}
    for triples_map in triples_maps:
        logical_table = _one(graph, triples_map, RR.logicalTable, "rr:logicalTable")
        table = str(_one(graph, logical_table, RR.tableName, "rr:tableName"))
        subject_map = _one(graph, triples_map, RR.subjectMap, "rr:subjectMap")
        identifier = _identifier_from_template(
            _one(graph, subject_map, RR.template, "rr:subjectMap/rr:template"),
            "rr:subjectMap/rr:template",
        )
        entity_type = _local_name(_one(graph, subject_map, RR["class"], "rr:subjectMap/rr:class"))
        label_columns = []
        for pom in graph.objects(triples_map, RR.predicateObjectMap):
            if RDFS.label in graph.objects(pom, RR.predicate):
                object_map = _one(graph, pom, RR.objectMap, "rdfs:label rr:objectMap")
                label_columns.append(str(_one(graph, object_map, RR.column, "rdfs:label rr:column")))
        if len(label_columns) != 1:
            raise R2RMLMappingError(f"{table}: exactly one rdfs:label rr:column is required")
        entity = EntityTableMapping(
            table=table, entity_type=entity_type, id_column=identifier, name_column=label_columns[0],
        )
        entities.append(entity)
        entity_by_map[triples_map] = entity

    relations: list[RelationTableMapping] = []
    for child_map, child_entity in entity_by_map.items():
        for pom in graph.objects(child_map, RR.predicateObjectMap):
            predicates = list(graph.objects(pom, RR.predicate))
            object_maps = list(graph.objects(pom, RR.objectMap))
            if not object_maps or not any(graph.value(obj, RR.parentTriplesMap) is not None for obj in object_maps):
                continue
            if len(predicates) != 1 or len(object_maps) != 1:
                raise R2RMLMappingError("a parent-triples-map relation needs one predicate and one object map")
            object_map = object_maps[0]
            parent_map = _one(graph, object_map, RR.parentTriplesMap, "rr:parentTriplesMap")
            parent_entity = entity_by_map.get(parent_map)
            if parent_entity is None:
                raise R2RMLMappingError("rr:parentTriplesMap must reference a local rr:TriplesMap")
            join = _one(graph, object_map, RR.joinCondition, "rr:joinCondition")
            child_column = str(_one(graph, join, RR.child, "rr:child"))
            parent_column = str(_one(graph, join, RR.parent, "rr:parent"))
            if parent_column != parent_entity.id_column:
                raise R2RMLMappingError(
                    f"join parent {parent_column!r} must be the parent subject identifier {parent_entity.id_column!r}"
                )
            relations.append(RelationTableMapping(
                table=child_entity.table,
                source_table=child_entity.table,
                target_table=parent_entity.table,
                source_column=child_entity.id_column,
                target_column=child_column,
                relation=_local_name(predicates[0]),
            ))
    return RelationalGraphMapping(
        id=mapping_id, version=version, source_id=source_id, tenant=tenant,
        entities=entities, relations=relations, ontology_version=ontology_version,
    )


@dataclass(frozen=True)
class FederatedOBDASource:
    """One independently authorized source participating in a tenant federation."""

    name: str
    ingestor: RelationalGraphIngestor
    mapping: RelationalGraphMapping


class FederatedOBDAIngestor:
    """Validate every source before materializing a federated tenant snapshot."""

    def __init__(self, sources: Iterable[FederatedOBDASource]):
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("at least one federated OBDA source is required")

    async def validate(self):
        tenants = {source.mapping.tenant for source in self.sources}
        source_ids = [source.mapping.source_id for source in self.sources]
        if len(tenants) != 1:
            raise ValueError("all federated OBDA sources must target one tenant")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("federated OBDA source_id values must be unique")
        reports = await asyncio.gather(*(source.ingestor.validate(source.mapping) for source in self.sources))
        failed = [report for report in reports if not report.valid]
        if failed:
            details = "; ".join(error for report in failed for error in report.errors)
            raise ValueError("federated OBDA validation failed: " + details)
        return reports

    async def ingest(self):
        # Preflight avoids starting a partial cross-source materialization when
        # any source mapping is invalid. Individual source writes retain their
        # established, SHACL-gated provenance contract.
        await self.validate()
        return await asyncio.gather(*(source.ingestor.ingest(source.mapping) for source in self.sources))


__all__ = [
    "FederatedOBDAIngestor", "FederatedOBDASource", "R2RMLMappingError", "r2rml_to_mapping",
]
