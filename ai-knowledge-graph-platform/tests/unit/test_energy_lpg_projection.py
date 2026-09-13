from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rdflib import BNode, Graph, Literal
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ASSET, ENERGY, EnergyDemoService
from graphrag.domains.energy.lpg_projection import RDFProjectionError, build_projection, project_to_neo4j, triple_ledger
from graphrag.core.models import Entity


class _Target:
    def __init__(self) -> None:
        self.entities: list[Entity] = []
        self.relationships: list[dict[str, Any]] = []
        self.entity_tenant = ""
        self.relationship_tenant = ""

    async def merge_entities_batch(self, entities: list[Entity], tenant: str = "default") -> list[dict[str, Any]]:
        self.entities = entities
        self.entity_tenant = tenant
        return []

    async def merge_relations_batch(self, rows: list[dict[str, Any]], tenant: str = "default") -> None:
        self.relationships = rows
        self.relationship_tenant = tenant


class _Result:
    def __init__(self, row=None):
        self.row = row

    async def consume(self):
        return None

    async def single(self):
        return self.row


class _GovernedTransaction:
    def __init__(self, entities: int, relations: int):
        self.entities, self.relations, self.cypher = entities, relations, []
    async def run(self, cypher, **_params):
        self.cypher.append(cypher)
        if "RETURN p.active_version" in cypher:
            return _Result({"active_version": "version"})
        if "RETURN nodes, generic_edges, typed_edges" in cypher:
            return _Result({"nodes": self.entities, "generic_edges": self.relations, "typed_edges": self.relations})
        return _Result()


class _GovernedTarget:
    def __init__(self, entities: int, relations: int):
        self.tx = _GovernedTransaction(entities, relations)

    async def run_in_transaction(self, work):
        await work(self.tx)


def test_projection_preserves_turbine_identity_tenant_literals_and_relationships():
    entities, relationships = build_projection(EnergyDemoService().graph, tenant="energy-demo")
    turbine = next(item for item in entities if item.name == str(ASSET["WT-01"]))

    assert turbine.type == "WIND_TURBINE"
    assert turbine.tenant == "energy-demo"
    assert turbine.semantic_properties["assetId"] == "WT-01"
    assert turbine.semantic_properties["rdf_iri"] == str(ASSET["WT-01"])
    assert any(item["relation"] == "HAS_COMPONENT" for item in relationships)


def test_projection_uses_the_real_neo4j_batch_client_contract():
    target = _Target()
    report = asyncio.run(project_to_neo4j(EnergyDemoService().graph, target, tenant="energy-demo"))

    assert report.node_count == len(target.entities)
    assert report.relationship_count == len(target.relationships)
    assert target.entity_tenant == target.relationship_tenant == "energy-demo"


def test_projection_rejects_blank_nodes_instead_of_silently_losing_them():
    graph = Graph()
    graph.add((ASSET["WT-X"], RDF.type, ENERGY.Asset))
    graph.add((ASSET["WT-X"], ENERGY.assetId, Literal("WT-X")))
    graph.add((ASSET["WT-X"], ENERGY.hasComponent, BNode()))

    with pytest.raises(RDFProjectionError, match="blank-node"):
        build_projection(graph, tenant="energy-demo")


def test_projection_ledger_accounts_for_every_rdf_triple_and_preserves_literal_fidelity():
    graph = EnergyDemoService().graph
    entities, _ = build_projection(graph, tenant="energy-demo")
    ledger = triple_ledger(graph)
    observation = next(item for item in entities if item.type == "OBSERVATION")

    assert ledger.projected + ledger.excluded + ledger.rejected == len(graph)
    assert ledger.rejected == 0
    literal_forms = observation.semantic_properties["rdf_literal_forms"]
    assert '"lexical": "96.0"' in literal_forms
    assert "rdf_provenance_sources" in observation.semantic_properties


def test_governed_projection_dual_writes_typed_edges_and_asserts_written_counts():
    graph = EnergyDemoService().graph
    entities, relations = build_projection(graph, tenant="energy-demo")
    target = _GovernedTarget(len(entities), len(relations))
    asyncio.run(project_to_neo4j(graph, target, tenant="energy-demo"))

    rendered = "\n".join(target.tx.cypher)
    assert "EnergyProjectionPointer" in rendered
    assert "RELATES_TO" in rendered
    assert "HAS_COMPONENT" in rendered
    assert "typed_edges" in rendered
