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
from graphrag.domains.energy.graph_blobs import content_hash


class RDFProjectionError(ValueError):
    """Raised when an RDF construct cannot be projected without guessing."""


class ProjectionConflictError(RDFProjectionError):
    """The active Neo4j projection pointer changed before its CAS flip."""


class Neo4jProjectionTarget(Protocol):
    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> list[dict]: ...

    async def merge_relations_batch(self, rows: list[dict], tenant: str = "default") -> None: ...


@dataclass(frozen=True)
class ProjectionReport:
    tenant: str
    node_count: int
    relationship_count: int
    source_graph_triples: int
    projection_version: str = ""
    projected_triples: int = 0
    excluded_triples: int = 0
    rejected_triples: int = 0


@dataclass(frozen=True)
class TripleLedger:
    """Every input triple has exactly one named projection disposition."""
    projected: int
    excluded: int
    rejected: int
    total: int

    def assert_complete(self) -> None:
        if self.projected + self.excluded + self.rejected != self.total:
            raise RDFProjectionError("triple ledger does not account for every input triple")


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


def _subject_properties(graph: Graph, subject: URIRef) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    properties: dict[str, Any] = {}
    literal_forms: dict[str, Any] = {}
    provenance_sources: list[str] = []
    for predicate, value in graph.predicate_objects(subject):
        if predicate in {RDF.type, RDFS.label}:
            continue
        if predicate == PROV.wasDerivedFrom:
            if not isinstance(value, URIRef):
                raise RDFProjectionError(f"{subject}: prov:wasDerivedFrom must be an IRI")
            provenance_sources.append(str(value))
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
        literal_forms[key] = {"lexical": str(value), "datatype": str(value.datatype or ""), "language": value.language or ""}
    return properties, literal_forms, sorted(set(provenance_sources))


def build_projection(graph: Graph, *, tenant: str) -> tuple[list[Entity], list[dict[str, Any]]]:
    """Build Neo4j mutation rows from a published, concrete Energy RDF graph."""
    if not tenant.strip():
        raise RDFProjectionError("a trusted tenant is required")

    typed_subjects = {subject for subject in graph.subjects(RDF.type, None) if isinstance(subject, URIRef) and str(subject).startswith(_BASE)}
    entities: list[Entity] = []
    identity: dict[URIRef, tuple[str, str, str]] = {}

    for subject in sorted(typed_subjects, key=str):
        model_type, lpg_type = _type_for(graph, subject)
        properties, literal_forms, provenance_sources = _subject_properties(graph, subject)
        all_types = sorted({_local_name(value) for value in graph.objects(subject, RDF.type)
                            if isinstance(value, URIRef) and str(value).startswith(str(ENERGY))})
        source_doc_id = provenance_sources[0] if provenance_sources else "rdf-projection"
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
                "rdf_types": json.dumps(all_types, sort_keys=True),
                "rdf_literal_forms": json.dumps(literal_forms, sort_keys=True),
                "rdf_provenance_sources": provenance_sources,
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


def triple_ledger(graph: Graph) -> TripleLedger:
    """Classify the source graph once, refusing unrepresentable triples."""
    typed = {subject for subject in graph.subjects(RDF.type, None)
             if isinstance(subject, URIRef) and str(subject).startswith(_BASE)}
    projected = excluded = rejected = 0
    for subject, predicate, obj in graph:
        if subject not in typed:
            rejected += 1
        elif predicate in {RDF.type, RDFS.label, PROV.wasDerivedFrom}:
            projected += 1
        elif isinstance(predicate, URIRef) and str(predicate).startswith(str(ENERGY)):
            if isinstance(obj, BNode) or (isinstance(obj, URIRef) and obj not in typed):
                rejected += 1
            else:
                projected += 1
        else:
            # A named policy: non-domain metadata is intentionally not read
            # by the LPG model, but is never silently dropped from the report.
            excluded += 1
    ledger = TripleLedger(projected, excluded, rejected, len(graph))
    ledger.assert_complete()
    return ledger


async def project_to_neo4j(
    graph: Graph, target: Neo4jProjectionTarget, *, tenant: str,
    expected_active_version: str | None = None,
) -> ProjectionReport:
    """Stage a governed, versioned Neo4j projection when the target supports it.

    The legacy batch contract remains a small-test fallback; the production
    ``Neo4jClient`` takes the transactional path below.
    """
    entities, relationships = build_projection(graph, tenant=tenant)
    ledger = triple_ledger(graph)
    if ledger.rejected:
        raise RDFProjectionError(f"projection rejected {ledger.rejected} RDF triple(s)")
    version = content_hash(graph)
    if hasattr(target, "run_in_transaction"):
        expected = expected_active_version
        if expected is None and hasattr(target, "run"):
            rows = await target.run(
                "MATCH (p:EnergyProjectionPointer {tenant: $tenant, name: 'energy'}) "
                "RETURN p.active_version AS version", tenant=tenant,
            )
            expected = rows[0].get("version") if rows else None
        if expected == version:
            return ProjectionReport(tenant, len(entities), len(relationships), len(graph), version,
                                    ledger.projected, ledger.excluded, ledger.rejected)
        await _write_governed_projection(target, entities, relationships, tenant, version, expected)
        return ProjectionReport(tenant, len(entities), len(relationships), len(graph), version,
                                ledger.projected, ledger.excluded, ledger.rejected)
    await target.merge_entities_batch(entities, tenant=tenant)
    await target.merge_relations_batch(relationships, tenant=tenant)
    return ProjectionReport(tenant, len(entities), len(relationships), len(graph), version,
                            ledger.projected, ledger.excluded, ledger.rejected)


async def _write_governed_projection(target: Any, entities: list[Entity], relationships: list[dict[str, Any]],
                                     tenant: str, version: str, expected: str | None) -> None:
    """Write candidate rows then CAS-flip the active pointer in one transaction."""
    entity_rows = []
    for entity in entities:
        types = json.loads(entity.semantic_properties["rdf_types"])
        labels = ["EnergyProjectionEntity"] + [_TYPE_TO_LPG[item] for item in types if item in _TYPE_TO_LPG]
        entity_rows.append({"iri": entity.name, "labels": labels, "properties": {
            "name": entity.name, "type": entity.type, "description": entity.description,
            "source_doc_id": entity.source_doc_id, "source_type": str(entity.source_type),
            **entity.semantic_properties,
        }})

    async def work(tx: Any) -> None:
        # Candidate records are invisible to readers until the pointer flip.
        result = await tx.run("""
            MERGE (v:EnergyProjectionVersion {tenant: $tenant, version: $version})
            ON CREATE SET v.created_at = datetime(), v.marker = 'energy-rdf-projection/v2'
            RETURN v.version AS version
        """, tenant=tenant, version=version)
        await result.consume()
        for labels in sorted({tuple(row["labels"]) for row in entity_rows}):
            safe_labels = ":".join(labels)
            rows = [row for row in entity_rows if tuple(row["labels"]) == labels]
            result = await tx.run(f"""
                UNWIND $rows AS row
                MERGE (e:{safe_labels} {{tenant: $tenant, projection_version: $version, rdf_iri: row.iri}})
                SET e += row.properties
                WITH e MATCH (v:EnergyProjectionVersion {{tenant: $tenant, version: $version}})
                MERGE (v)-[:CONTAINS]->(e)
            """, rows=rows, tenant=tenant, version=version)
            await result.consume()
        for relation in sorted({row["relation"] for row in relationships}):
            rows = [row for row in relationships if row["relation"] == relation]
            result = await tx.run(f"""
                UNWIND $rows AS row
                MATCH (s:EnergyProjectionEntity {{tenant: $tenant, projection_version: $version, rdf_iri: row.src_name}})
                MATCH (t:EnergyProjectionEntity {{tenant: $tenant, projection_version: $version, rdf_iri: row.tgt_name}})
                MERGE (s)-[r:RELATES_TO {{tenant: $tenant, projection_version: $version, relation: row.relation}}]->(t)
                SET r += row
                MERGE (s)-[typed:{relation} {{tenant: $tenant, projection_version: $version}}]->(t)
                SET typed.rdf_predicate = row.relation, typed.marker = 'energy-rdf-projection/v2'
            """, rows=rows, tenant=tenant, version=version)
            await result.consume()
        # A zero-row CAS rolls the entire candidate write back with this transaction.
        result = await tx.run("""
            MERGE (p:EnergyProjectionPointer {tenant: $tenant, name: 'energy'})
            ON CREATE SET p.active_version = null
            WITH p, p.active_version AS actual
            WHERE (actual IS NULL AND $expected IS NULL) OR actual = $expected OR actual = $version
            OPTIONAL MATCH (p)-[old:ACTIVE_ENERGY_PROJECTION]->()
            DELETE old
            WITH p
            MATCH (v:EnergyProjectionVersion {tenant: $tenant, version: $version})
            MERGE (p)-[:ACTIVE_ENERGY_PROJECTION]->(v)
            SET p.active_version = $version, p.updated_at = datetime()
            RETURN p.active_version AS active_version
        """, tenant=tenant, version=version, expected=expected)
        record = await result.single()
        if record is None:
            raise ProjectionConflictError("active Energy projection changed; reload its version and retry")
        result = await tx.run("""
            CALL { MATCH (e:EnergyProjectionEntity {tenant: $tenant, projection_version: $version}) RETURN count(e) AS nodes }
            CALL { MATCH (:EnergyProjectionEntity {tenant: $tenant, projection_version: $version})-[r:RELATES_TO {tenant: $tenant, projection_version: $version}]->() RETURN count(r) AS generic_edges }
            CALL { MATCH (:EnergyProjectionEntity {tenant: $tenant, projection_version: $version})-[r]->() WHERE type(r) <> 'RELATES_TO' AND r.marker = 'energy-rdf-projection/v2' RETURN count(r) AS typed_edges }
            RETURN nodes, generic_edges, typed_edges
        """, tenant=tenant, version=version)
        counts = await result.single()
        if counts is None or int(counts["nodes"]) != len(entities) or int(counts["generic_edges"]) != len(relationships) or int(counts["typed_edges"]) != len(relationships):
            raise RDFProjectionError("Neo4j written-side count assertion failed")
        # Keep active + predecessor only; every delete is tenant and marker scoped.
        result = await tx.run("""
            MATCH (v:EnergyProjectionVersion {tenant: $tenant, marker: 'energy-rdf-projection/v2'})
            WITH v ORDER BY v.created_at DESC, v.version DESC SKIP 2
            OPTIONAL MATCH (e:EnergyProjectionEntity {tenant: $tenant, projection_version: v.version})
            DETACH DELETE e
            WITH DISTINCT v DETACH DELETE v
        """, tenant=tenant)
        await result.consume()
    await target.run_in_transaction(work)


__all__ = ["Neo4jProjectionTarget", "ProjectionConflictError", "ProjectionReport", "RDFProjectionError", "TripleLedger", "build_projection", "project_to_neo4j", "triple_ledger"]
