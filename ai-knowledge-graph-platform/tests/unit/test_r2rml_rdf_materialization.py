"""Regression tests proving R2RML mappings execute for real: relational row
-> materialized RDF triples -> SPARQL finding.

Before graphrag/ingestion/r2rml_rdf.py, R2RML mappings were parsed and
validated (r2rml_to_mapping(), tests/unit/test_r2rml_obda.py) or wired into
a Neo4j-only ingestion CLI (scripts/ingest_r2rml.py,
tests/unit/test_ingest_r2rml.py) -- no test anywhere produced a single RDF
triple from an R2RML mapping. These tests close that gap end to end, without
needing a live triplestore: the materialized rdflib.Graph is queried
directly via SPARQLBridge, the same query engine POST /kg/sparql uses
locally.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, RDFS

# Allow importing from scripts/ — same convention as test_ingest_r2rml.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from create_energy_demo_sqlite import create as create_energy_demo_sqlite  # noqa: E402

from graphrag.graph.sparql_bridge import SPARQLBridge  # noqa: E402
from graphrag.ingestion.r2rml import R2RMLMappingError  # noqa: E402
from graphrag.ingestion.r2rml_rdf import materialize_r2rml  # noqa: E402
from graphrag.ingestion.relational import SQLiteSourceConnector  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ENERGY_MAPPING = ROOT / "ontology" / "mappings" / "energy-assets.r2rml.ttl"
SUPPLY_CHAIN_MAPPING = ROOT / "ontology" / "mappings" / "supply-chain.r2rml.ttl"

ENERGY_NS = "https://example.energy.demo/ontology#"
ASSET = URIRef("https://example.energy.demo/asset/WT-01")
WORK_ORDER = URIRef("https://example.energy.demo/record/WO-9001")


@pytest.fixture
def energy_sqlite(tmp_path) -> Path:
    db_path = tmp_path / "energy-demo-sap.sqlite"
    create_energy_demo_sqlite(db_path)
    return db_path


class TestMaterializeEnergyAssets:
    async def test_asset_and_work_order_triples_are_materialized(self, energy_sqlite):
        graph = await materialize_r2rml(ENERGY_MAPPING, SQLiteSourceConnector(energy_sqlite))

        assert (ASSET, RDF.type, URIRef(ENERGY_NS + "Asset")) in graph
        assert (WORK_ORDER, RDF.type, URIRef(ENERGY_NS + "WorkOrder")) in graph
        assert (WORK_ORDER, URIRef(ENERGY_NS + "status"), Literal("open")) in graph
        # The join predicate: the plain-column predicate the old
        # r2rml_to_mapping() parser cannot represent at all, and the join
        # relation it can represent only as a Neo4j edge, never a real triple.
        assert (WORK_ORDER, URIRef(ENERGY_NS + "concernsAsset"), ASSET) in graph

    async def test_closed_work_order_is_still_materialized_with_its_own_status(self, energy_sqlite):
        graph = await materialize_r2rml(ENERGY_MAPPING, SQLiteSourceConnector(energy_sqlite))

        closed = URIRef("https://example.energy.demo/record/WO-9003")
        assert (closed, URIRef(ENERGY_NS + "status"), Literal("closed")) in graph

    async def test_row_to_triples_to_sparql_finding(self, energy_sqlite):
        """The maintenance query this pipeline exists to answer: which
        assets currently have open work orders. Proves the full chain --
        SQLite row -> R2RML-materialized triple -> SPARQL result -- with no
        live triplestore, using the same SPARQLBridge query engine
        POST /kg/sparql uses against a local Turtle snapshot."""
        graph = await materialize_r2rml(ENERGY_MAPPING, SQLiteSourceConnector(energy_sqlite))
        bridge = SPARQLBridge(graph)

        rows = bridge.query(
            """
            PREFIX energy: <https://example.energy.demo/ontology#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?asset ?assetLabel ?workOrder ?status WHERE {
              ?workOrder a energy:WorkOrder ;
                         energy:status ?status ;
                         energy:concernsAsset ?asset .
              ?asset rdfs:label ?assetLabel .
              FILTER(?status = "open")
            }
            ORDER BY ?asset
            """
        )

        asset_labels = {row["assetLabel"] for row in rows}
        assert asset_labels == {"Wind turbine WT-01", "Wind turbine WT-02"}
        # WO-9003 (status "closed", asset WT-03) must not appear.
        assert all(row["status"] == "open" for row in rows)
        assert len(rows) == 2


class TestUnsupportedConstructsFailClosed:
    async def test_rr_constant_object_map_is_rejected(self, tmp_path, energy_sqlite):
        bad_mapping = tmp_path / "bad.r2rml.ttl"
        bad_mapping.write_text(
            """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix energy: <https://example.energy.demo/ontology#> .
            energy:AssetMap a rr:TriplesMap;
              rr:logicalTable [ rr:tableName "sap_assets" ];
              rr:subjectMap [ rr:template "https://example.energy.demo/asset/{asset_id}"; rr:class energy:Asset ];
              rr:predicateObjectMap [ rr:predicate energy:region; rr:objectMap [ rr:constant "eu-west" ] ].
            """,
            encoding="utf-8",
        )
        with pytest.raises(R2RMLMappingError):
            await materialize_r2rml(bad_mapping, SQLiteSourceConnector(energy_sqlite))

    async def test_multi_column_template_is_rejected(self, tmp_path, energy_sqlite):
        bad_mapping = tmp_path / "bad.r2rml.ttl"
        bad_mapping.write_text(
            """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix energy: <https://example.energy.demo/ontology#> .
            energy:AssetMap a rr:TriplesMap;
              rr:logicalTable [ rr:tableName "sap_assets" ];
              rr:subjectMap [ rr:template "https://example.energy.demo/asset/{asset_id}/{asset_name}"; rr:class energy:Asset ].
            """,
            encoding="utf-8",
        )
        with pytest.raises(R2RMLMappingError):
            await materialize_r2rml(bad_mapping, SQLiteSourceConnector(energy_sqlite))

    async def test_forward_referenced_parent_map_is_rejected(self, tmp_path, energy_sqlite):
        # WorkOrderMap listed before AssetMap -- the join target hasn't been
        # materialized yet when WorkOrderMap is processed.
        bad_mapping = tmp_path / "bad.r2rml.ttl"
        bad_mapping.write_text(
            """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix energy: <https://example.energy.demo/ontology#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            energy:WorkOrderMap a rr:TriplesMap;
              rr:logicalTable [ rr:tableName "sap_work_orders" ];
              rr:subjectMap [ rr:template "https://example.energy.demo/record/{work_order_id}"; rr:class energy:WorkOrder ];
              rr:predicateObjectMap [
                rr:predicate energy:concernsAsset;
                rr:objectMap [
                  rr:parentTriplesMap energy:AssetMap;
                  rr:joinCondition [ rr:child "asset_id"; rr:parent "asset_id" ]
                ]
              ].
            energy:AssetMap a rr:TriplesMap;
              rr:logicalTable [ rr:tableName "sap_assets" ];
              rr:subjectMap [ rr:template "https://example.energy.demo/asset/{asset_id}"; rr:class energy:Asset ].
            """,
            encoding="utf-8",
        )
        with pytest.raises(R2RMLMappingError):
            await materialize_r2rml(bad_mapping, SQLiteSourceConnector(energy_sqlite))


class TestMaterializeSupplyChain:
    """Cheap extra coverage on the simpler, existing mapping (label + join
    only, no plain-column predicate) -- guards against regressing the case
    r2rml_to_mapping() already handled."""

    @pytest.fixture
    def supply_chain_sqlite(self, tmp_path) -> Path:
        db_path = tmp_path / "supply-chain.sqlite"
        with sqlite3.connect(db_path) as db:
            db.executescript(
                """
                CREATE TABLE suppliers (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE materials (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE supplies (supplier_id TEXT NOT NULL, material_id TEXT NOT NULL);
                INSERT INTO suppliers VALUES ('s1', 'Supplier One');
                INSERT INTO materials VALUES ('m1', 'Material One');
                INSERT INTO supplies VALUES ('s1', 'm1');
                """
            )
        return db_path

    async def test_supplier_material_join_is_materialized(self, supply_chain_sqlite):
        # MaterialMap precedes SupplyMap in the shipped mapping file, so the
        # join in SupplyMap can resolve against already-materialized Material
        # subjects -- exercises the same file-order dependency the energy
        # mapping does, on a mapping r2rml_to_mapping() already parses today.
        graph = await materialize_r2rml(SUPPLY_CHAIN_MAPPING, SQLiteSourceConnector(supply_chain_sqlite))

        supplier = URIRef("https://example.org/suppliers/s1")
        material = URIRef("https://example.org/materials/m1")
        supply = URIRef("https://example.org/supplies/s1")  # SupplyMap templates on supplier_id
        ex_supply = "https://example.org/supply/"

        assert (supplier, RDF.type, URIRef(ex_supply + "Supplier")) in graph
        assert (supplier, RDFS.label, Literal("Supplier One")) in graph
        assert (material, RDF.type, URIRef(ex_supply + "Material")) in graph
        assert (supply, RDF.type, URIRef(ex_supply + "Supply")) in graph
        # The join triple: supply -> ex:supplies -> the joined Material subject.
        assert (supply, URIRef(ex_supply + "supplies"), material) in graph
