from __future__ import annotations

import asyncio
from typing import Any

import pytest
from rdflib import BNode, Graph, Literal
from rdflib.namespace import RDF

from graphrag.domains.energy.demo import ASSET, ENERGY, EnergyDemoService
from graphrag.domains.energy.lpg_projection import RDFProjectionError, build_projection, project_to_neo4j
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
