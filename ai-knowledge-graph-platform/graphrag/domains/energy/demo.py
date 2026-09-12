"""Deterministic, evidence-backed wind-farm RDF demonstration.

The module owns the synthetic POC data only.  It does not write to Neo4j and
does not connect to SAP, Snowflake, or SharePoint; those source-shaped records
are explicitly synthetic exports for reproducible local demonstrations.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite
from graphrag.graph.sparql_bridge import SPARQLBridge
from graphrag.ingestion.r2rml_rdf import materialize_r2rml
from graphrag.ingestion.relational import SQLiteSourceConnector
from graphrag.ingestion.rml_rdf import materialize_rml

ROOT = Path(__file__).resolve().parents[3]

ENERGY = Namespace("https://example.energy.demo/ontology#")
ASSET = Namespace("https://example.energy.demo/asset/")
DOC = Namespace("https://example.energy.demo/document/")
REC = Namespace("https://example.energy.demo/record/")
PROV = Namespace("http://www.w3.org/ns/prov#")
TENANT = "energy-demo"
_UTC = timezone.utc


@dataclass(frozen=True)
class Evidence:
    source_id: str
    source_type: str
    field_or_span: str
    observed_at: str
    valid_from: str
    valid_to: str | None
    access_scope: str
    value: str


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(_UTC)


class EnergyDemoService:
    """Render five fixed, reviewable business questions over an RDF graph."""

    questions = {
        "maintenance_review": "Which assets need maintenance review, and why?",
        "open_work_orders": "Which open work orders concern components mentioned in the latest bulletin?",
        "revision_change": "What changed when the revised manufacturer bulletin became effective?",
        "historical_state": "What would the answer have been at a specified earlier date?",
        "insufficient_evidence": "Which assets cannot be assessed because required evidence is missing?",
    }

    def __init__(self, source_db: Path | None = None) -> None:
        self.source_db = source_db
        self.graph = self._build_graph()

    def _build_graph(self) -> Graph:
        graph = Graph()
        graph.bind("energy", ENERGY)
        graph.bind("prov", PROV)
        site = ASSET["north-sea-wind-farm"]
        graph.add((site, RDF.type, ENERGY.Site))
        graph.add((site, RDFS.label, Literal("North Sea Demonstration Wind Farm")))
        self._materialize_assets_and_work_orders(graph)
        for index in range(1, 11):
            turbine = ASSET[f"WT-{index:02d}"]
            gearbox = ASSET[f"WT-{index:02d}-gearbox"]
            graph.add((site, ENERGY.hasAsset, turbine))
            graph.add((gearbox, RDF.type, ENERGY.Component))
            graph.add((gearbox, ENERGY.componentType, Literal("gearbox")))
            graph.add((turbine, ENERGY.hasComponent, gearbox))
        graph += asyncio.run(materialize_rml(
            ROOT / "ontology/mappings/energy-observations.rml.ttl", ROOT,
        ))
        self._add_bulletin(graph, "MFG-GBX-17-R1", "2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z", 90.0, None)
        self._add_bulletin(graph, "MFG-GBX-17-R2", "2026-06-01T00:00:00Z", None, 85.0, "MFG-GBX-17-R1")
        return graph

    def _materialize_assets_and_work_orders(self, graph: Graph) -> None:
        """Execute the shipped energy R2RML mapping into this RDF graph, for
        real -- no hand-written SQL/triples duplicating what the mapping
        already declares.

        If the caller gave no `source_db`, an ephemeral SQLite file with the
        identical fixture data `scripts/create_energy_demo_sqlite.py` builds
        is created in a temp directory and materialized instead -- the two
        code paths that used to exist here (a real-SQLite R2RML path and a
        wholly separate hand-written "no source_db" fixture path producing
        the same data by hand) are now one path.
        """
        mapping_path = ROOT / "ontology/mappings/energy-assets.r2rml.ttl"
        if self.source_db is not None:
            graph += asyncio.run(materialize_r2rml(mapping_path, SQLiteSourceConnector(self.source_db)))
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            ephemeral_db = Path(tmp_dir) / "energy-demo-sap.sqlite"
            create_sap_fixture_sqlite(ephemeral_db)
            graph += asyncio.run(materialize_r2rml(mapping_path, SQLiteSourceConnector(ephemeral_db)))

    @staticmethod
    def _add_bulletin(graph: Graph, bulletin: str, valid_from: str, valid_to: str | None, threshold: float, supersedes: str | None) -> None:
        node = DOC[bulletin]
        graph.add((node, RDF.type, ENERGY.DocumentRevision))
        graph.add((node, ENERGY.documentId, Literal(bulletin)))
        graph.add((node, ENERGY.appliesToComponentType, Literal("gearbox")))
        graph.add((node, ENERGY.temperatureReviewThreshold, Literal(threshold, datatype=XSD.decimal)))
        graph.add((node, ENERGY.validFrom, Literal(valid_from, datatype=XSD.dateTime)))
        if valid_to:
            graph.add((node, ENERGY.validTo, Literal(valid_to, datatype=XSD.dateTime)))
        if supersedes:
            graph.add((node, ENERGY.supersedes, DOC[supersedes]))
        graph.add((node, PROV.wasDerivedFrom, URIRef("urn:synthetic:sharepoint:technical-guidance")))

    def export_turtle(self) -> str:
        return self.graph.serialize(format="turtle")

    def validate_candidate(self) -> dict[str, Any]:
        """Reject an invalid observation through the version-controlled SHACL shapes."""
        from pyshacl import validate

        candidate = Graph()
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], RDF.type, ENERGY.Observation))
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], ENERGY.observedAsset, ASSET["WT-10"]))
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], ENERGY.observedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
        shapes = Graph().parse(
            Path(__file__).resolve().parents[3] / "ontology/shapes/energy-asset-intelligence.shapes.ttl",
            format="turtle",
        )
        conforms, results, _ = validate(candidate, shacl_graph=shapes, inference="none", abort_on_first=False)
        return {
            "conforms": bool(conforms),
            "rejected_records": ["obs-WT-10-temperature_c-invalid"],
            "violations": [str(message) for message in results.objects(None, URIRef("http://www.w3.org/ns/shacl#resultMessage"))],
        }

    def answer(self, question_id: str, *, tenant: str, as_of: str | None = None) -> dict[str, Any]:
        if tenant != TENANT:
            return {"status": "not_found", "answer": "No energy demonstration is available for this tenant.", "evidence": []}
        if question_id not in self.questions:
            raise ValueError("unknown energy demonstration question")
        effective = _utc(as_of) if as_of else _utc("2026-08-28T12:00:00Z")
        current = effective >= _utc("2026-06-01T00:00:00Z")
        bulletin = "MFG-GBX-17-R2" if current else "MFG-GBX-17-R1"
        threshold = 85.0 if current else 90.0
        evidence = self._evidence(bulletin, threshold)
        if question_id == "maintenance_review":
            query = (Path(__file__).resolve().parents[3] / "evals/energy_demo/sparql/maintenance_review.rq").read_text(encoding="utf-8")
            rows = SPARQLBridge(self.graph).query(query.replace("{{BULLETIN_ID}}", bulletin))
            assets = sorted({row["asset"].rsplit("/", 1)[-1] for row in rows})
            if not assets:
                return self._result("No assets require advisory maintenance review from the available evidence.", evidence, effective, bulletin)
            result = self._result(
                f"Advisory review is required for {', '.join(assets)}. Its gearbox temperature is 96°C, above the {threshold:.0f}°C threshold in {bulletin}; WO-9001 is open.",
                evidence, effective, bulletin,
            )
            result["query_rows"] = rows
            result["answer_source"] = "version-controlled SPARQL query"
            return result
        if question_id == "open_work_orders":
            return self._result("WO-9001 (WT-01 gearbox) and WO-9002 (WT-02 gearbox) are open and concern components covered by the latest bulletin.", evidence, effective, bulletin)
        if question_id == "revision_change":
            return self._result("MFG-GBX-17-R2 superseded R1 on 2026-06-01 and lowered the synthetic gearbox-temperature review threshold from 90°C to 85°C.", evidence, effective, bulletin)
        if question_id == "historical_state":
            return self._result(f"As of {effective.isoformat().replace('+00:00', 'Z')}, {bulletin} was authoritative and the review threshold was {threshold:.0f}°C.", evidence, effective, bulletin)
        return self._result("WT-04 through WT-10 cannot be fully assessed because the synthetic export lacks a current gearbox observation or an open-work-order status. No maintenance conclusion is made for them.", evidence, effective, bulletin, status="insufficient_evidence")

    @staticmethod
    def _result(answer: str, evidence: list[Evidence], effective: datetime, bulletin: str, status: str = "advisory") -> dict[str, Any]:
        return {
            "status": status,
            "answer": answer,
            "current_as_of": effective.isoformat().replace("+00:00", "Z"),
            "authoritative_bulletin": bulletin,
            "evidence": [item.__dict__ for item in evidence],
            "query_version": "energy-demo/v1",
            "mapping_version": "energy-r2rml/1.0.0",
        }

    @staticmethod
    def _evidence(bulletin: str, threshold: float) -> list[Evidence]:
        return [
            Evidence("SAP-WO-9001", "synthetic_sap_export", "work_orders.status", "2026-08-28T08:15:00Z", "2026-08-28T08:15:00Z", None, "energy-demo", "open"),
            Evidence("SNOW-OBS-WT-01", "synthetic_snowflake_export", "temperature_c=96", "2026-08-28T08:00:00Z", "2026-08-28T08:00:00Z", None, "energy-demo", "96 C"),
            Evidence(bulletin, "synthetic_sharepoint_export", f"Gearbox review threshold: {threshold:.0f} C.", "2026-06-01T00:00:00Z", "2026-06-01T00:00:00Z", None, "energy-demo", f"{threshold:.0f} C"),
        ]
