"""Governed one-way projection of published Energy RDF into Neo4j.

RDF remains the authoritative Energy evidence graph.  This module creates a
rebuildable Neo4j read model for traversal and GraphRAG: it never reads from
Neo4j to mutate RDF, and it rejects RDF it cannot represent faithfully enough
for the property-graph contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from graphrag.core.models import Entity
from graphrag.domains.energy.demo import ENERGY, PROV


class RDFProjectionError(ValueError):
    """Raised when an RDF construct cannot be projected without guessing."""


class Neo4jProjectionTarget(Protocol):
    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> list[dict]: ...

    async def merge_relations_batch(self, rows: list[dict], tenant: str = "default") -> None: ...


@dataclass(frozen=True)
class ProjectionReport:
    tenant: str
    node_count: int
    relationship_count: int
    source_graph_triples: int


_BASE = "https://example.energy.demo/"
_TYPE_PRIORITY = (
    "WindTurbine", "DocumentRevision", "WorkOrder", "Observation",
    "Component", "Site", "Asset",
)
_TYPE_TO_LPG = {
    "Asset": "ASSET", "WindTurbine": "WIND_TURBINE", "Component": "COMPONENT",
    "Site": "SITE", "Observation": "OBSERVATION", "WorkOrder": "WORK_ORDER",
    "DocumentRevision": "DOCUMENT_REVISION",
}


def _local_name(value: URIRef) -> str:
    text = str(value)
    if not text.startswith(str(ENERGY)):
        raise RDFProjectionError(f"unsupported vocabulary term {text!r}")
    return text.removeprefix(str(ENERGY))


def _lpg_relation(value: str) -> str:
    """Convert the canonical camelCase predicate to Neo4j's UPPER_SNAKE form."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).upper()


def _literal_value(value: Literal) -> Any:
    """Use Neo4j-compatible scalar values, retaining the XSD type separately."""
    parsed = value.toPython()
    if isinstance(parsed, Decimal):
        return float(parsed)
    if isinstance(parsed, (str, bool, int, float, datetime, date)):
        return parsed
    return str(value)


def _type_for(graph: Graph, subject: URIRef) -> tuple[str, str]:
    types = {_local_name(value) for value in graph.objects(subject, RDF.type) if isinstance(value, URIRef) and str(value).startswith(str(ENERGY))}
    for type_name in _TYPE_PRIORITY:
        if type_name in types:
            return type_name, _TYPE_TO_LPG[type_name]
    raise RDFProjectionError(f"{subject} has no supported Energy rdf:type")


def _label(graph: Graph, subject: URIRef) -> str:
    label = graph.value(subject, RDFS.label)
    return str(label) if isinstance(label, Literal) else str(subject)


def _subject_properties(graph: Graph, subject: URIRef) -> tuple[dict[str, Any], dict[str, str], str]:
    properties: dict[str, Any] = {}
    datatypes: dict[str, str] = {}
    source_doc_id = "rdf-projection"
    for predicate, value in graph.predicate_objects(subject):
        if predicate in {RDF.type, RDFS.label}:
            continue
        if predicate == PROV.wasDerivedFrom:
            if not isinstance(value, URIRef):
                raise RDFProjectionError(f"{subject}: prov:wasDerivedFrom must be an IRI")
            source_doc_id = str(value)
            continue
        if not isinstance(predicate, URIRef) or not str(predicate).startswith(str(ENERGY)):
            raise RDFProjectionError(f"{subject}: unsupported predicate {predicate!r}")
        if isinstance(value, BNode):
            raise RDFProjectionError(f"{subject}: blank-node value for {predicate} cannot be projected")
        if isinstance(value, URIRef):
            # Object-property triples are emitted as relationships below.
            continue
        if not isinstance(value, Literal):
            raise RDFProjectionError(f"{subject}: unsupported value {value!r}")
        key = _local_name(predicate)
        if key in properties:
            raise RDFProjectionError(f"{subject}: repeated literal {key!r} requires an explicit LPG collection mapping")
        properties[key] = _literal_value(value)
        if value.datatype:
            datatypes[key] = str(value.datatype)
    return properties, datatypes, source_doc_id


def build_projection(graph: Graph, *, tenant: str) -> tuple[list[Entity], list[dict[str, Any]]]:
    """Build Neo4j mutation rows from a published, concrete Energy RDF graph."""
    if not tenant.strip():
        raise RDFProjectionError("a trusted tenant is required")

    typed_subjects = {subject for subject in graph.subjects(RDF.type, None) if isinstance(subject, URIRef) and str(subject).startswith(_BASE)}
    entities: list[Entity] = []
    identity: dict[URIRef, tuple[str, str, str]] = {}

    for subject in sorted(typed_subjects, key=str):
        model_type, lpg_type = _type_for(graph, subject)
        properties, datatypes, source_doc_id = _subject_properties(graph, subject)
        identity[subject] = (str(subject), lpg_type, source_doc_id)
        entities.append(Entity(
            name=str(subject),
            type=lpg_type,
            description=_label(graph, subject),
            tenant=tenant,
            source_doc_id=source_doc_id,
            source_type="document",
            prompt_version="rdf-energy-projection/v1",
            semantic_properties={
                **properties,
                "rdf_iri": str(subject),
                "rdf_type": model_type,
                "rdf_datatypes": json.dumps(datatypes, sort_keys=True),
            },
        ))

    relationships: list[dict[str, Any]] = []
    for subject, predicate, target in graph:
        if not (isinstance(subject, URIRef) and isinstance(predicate, URIRef) and isinstance(target, URIRef)):
            continue
        if subject not in identity or target not in identity or not str(predicate).startswith(str(ENERGY)):
            continue
        relation = _local_name(predicate)
        src_name, src_type, source_doc_id = identity[subject]
        tgt_name, tgt_type, _ = identity[target]
        relationships.append({
            "src_name": src_name, "src_type": src_type,
            "tgt_name": tgt_name, "tgt_type": tgt_type,
            "relation": _lpg_relation(relation), "weight": 1.0, "confidence": 1.0,
            "confidence_state": "ASSERTED", "extracted_at": None,
            "source_doc_id": source_doc_id, "source_type": "DOCUMENT",
            "constraint_type": "SOFT", "valid_from": None, "valid_to": None,
            "span_start": None, "span_end": None,
            "extraction_model": "rdf-energy-projection", "prompt_version": "v1",
        })
    return entities, relationships


async def project_to_neo4j(graph: Graph, target: Neo4jProjectionTarget, *, tenant: str) -> ProjectionReport:
    """Upsert the RDF-derived read model using the real Neo4j client contract."""
    entities, relationships = build_projection(graph, tenant=tenant)
    await target.merge_entities_batch(entities, tenant=tenant)
    await target.merge_relations_batch(relationships, tenant=tenant)
    return ProjectionReport(tenant, len(entities), len(relationships), len(graph))


__all__ = ["Neo4jProjectionTarget", "ProjectionReport", "RDFProjectionError", "build_projection", "project_to_neo4j"]
