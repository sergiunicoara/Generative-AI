"""Small PROV-O vocabulary and URI layer used by RDF exports.

The operational graph remains a Neo4j property graph.  This module provides a
stable, tenant-scoped RDF projection for provenance without forcing PROV-O
classes into every operational node or duplicating the graph's domain model.
"""

from __future__ import annotations

from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

PROV = Namespace("http://www.w3.org/ns/prov#")
INST = Namespace("https://graphrag.example.com/entity/")
ANNOT = Namespace("https://graphrag.example.com/annotation#")


def _part(value: object) -> str:
    return quote(str(value or ""), safe="")


def _uri(kind: str, identifier: object, tenant: str) -> URIRef:
    return INST[f"{kind}/{_part(tenant)}/{_part(identifier)}"]


def document_uri(identifier: object, tenant: str) -> URIRef:
    return _uri("document", identifier, tenant)


def chunk_uri(identifier: object, tenant: str) -> URIRef:
    return _uri("chunk", identifier, tenant)


def entity_uri(identifier: object, tenant: str) -> URIRef:
    return _uri("prov-entity", identifier, tenant)


def query_uri(identifier: object, tenant: str) -> URIRef:
    return _uri("query", identifier, tenant)


def answer_uri(identifier: object, tenant: str) -> URIRef:
    return _uri("answer", identifier, tenant)


def activity_uri(kind: str, identifier: object, tenant: str) -> URIRef:
    return _uri(f"activity/{_part(kind)}", identifier, tenant)


def agent_uri(kind: str, identifier: object, tenant: str) -> URIRef:
    return _uri(f"agent/{_part(kind)}", identifier, tenant)


def add_entity(
    graph: Graph,
    uri: URIRef,
    *,
    label: str = "",
    entity_type: URIRef | None = None,
    tenant: str = "",
) -> URIRef:
    graph.add((uri, RDF.type, PROV.Entity))
    if label:
        graph.add((uri, RDFS.label, Literal(label)))
    if entity_type is not None:
        graph.add((uri, RDF.type, entity_type))
    if tenant:
        graph.add((uri, ANNOT.tenant, Literal(tenant)))
    return uri


def add_agent(
    graph: Graph,
    uri: URIRef,
    *,
    label: str,
    software: bool = True,
    tenant: str = "",
) -> URIRef:
    graph.add((uri, RDF.type, PROV.SoftwareAgent if software else PROV.Agent))
    # PROV-O declares SoftwareAgent as a subclass of Agent, but exported
    # graphs must remain valid when consumed without an ontology reasoner.
    if software:
        graph.add((uri, RDF.type, PROV.Agent))
    graph.add((uri, RDFS.label, Literal(label)))
    if tenant:
        graph.add((uri, ANNOT.tenant, Literal(tenant)))
    return uri


def add_activity(
    graph: Graph,
    uri: URIRef,
    *,
    label: str,
    tenant: str,
    started_at: object = None,
    ended_at: object = None,
    status: str = "",
) -> URIRef:
    graph.add((uri, RDF.type, PROV.Activity))
    graph.add((uri, RDFS.label, Literal(label)))
    graph.add((uri, ANNOT.tenant, Literal(tenant)))
    if started_at:
        graph.add((uri, PROV.startedAtTime, _datetime_literal(started_at)))
    if ended_at:
        graph.add((uri, PROV.endedAtTime, _datetime_literal(ended_at)))
    if status:
        graph.add((uri, ANNOT.status, Literal(status)))
    return uri


def add_used(graph: Graph, activity: URIRef, entity: URIRef) -> None:
    graph.add((activity, PROV.used, entity))


def add_generated(graph: Graph, entity: URIRef, activity: URIRef) -> None:
    graph.add((entity, PROV.wasGeneratedBy, activity))
    graph.add((activity, PROV.generated, entity))


def add_derived(graph: Graph, entity: URIRef, source: URIRef) -> None:
    graph.add((entity, PROV.wasDerivedFrom, source))


def add_association(graph: Graph, activity: URIRef, agent: URIRef) -> None:
    graph.add((activity, PROV.wasAssociatedWith, agent))


def _datetime_literal(value: object) -> Literal:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return Literal(text, datatype=XSD.dateTime)


__all__ = [
    "ANNOT",
    "INST",
    "PROV",
    "activity_uri",
    "add_activity",
    "add_agent",
    "add_association",
    "add_derived",
    "add_entity",
    "add_generated",
    "add_used",
    "agent_uri",
    "answer_uri",
    "chunk_uri",
    "document_uri",
    "entity_uri",
    "query_uri",
]
