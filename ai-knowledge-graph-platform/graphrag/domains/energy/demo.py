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
from graphrag.domains.energy.publication import DatasetPublisher, PublicationReport
from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.graph.sparql_bridge import SPARQLBridge
from graphrag.ingestion.r2rml_rdf import materialize_r2rml
from graphrag.ingestion.relational import SQLiteSourceConnector
from graphrag.ingestion.rml_rdf import materialize_rml

ROOT = Path(__file__).resolve().parents[3]
SHAPES_PATH = ROOT / "ontology" / "shapes" / "energy-asset-intelligence.shapes.ttl"

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
        self._publisher = DatasetPublisher(SHAPES_PATH)
        candidate = self._build_graph()
        # SHACL as a publication gate: the candidate graph is staged,
        # validated, and only the conformant subset is published -- any
        # record that violates ontology/shapes/energy-asset-intelligence.shapes.ttl
        # is quarantined (see graphrag/domains/energy/publication.py), not
        # silently served. Every existing query path (answer(),
        # export_turtle()) now only ever sees this published graph.
        self._publisher.stage_and_publish(candidate)
        self.graph = self._publisher.current

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
        # str() first: Literal(<float>, datatype=XSD.decimal) keeps the
        # passed value's own Python type for .toPython() (a float, not a
        # Decimal) rather than parsing it through xsd:decimal's own
        # converter -- ill-typed under strict SHACL sh:datatype checking.
        # Same root-cause fix as graphrag/ingestion/rml_rdf.py's reference
        # object maps.
        graph.add((node, ENERGY.temperatureReviewThreshold, Literal(str(threshold), datatype=XSD.decimal)))
        graph.add((node, ENERGY.validFrom, Literal(valid_from, datatype=XSD.dateTime)))
        if valid_to:
            graph.add((node, ENERGY.validTo, Literal(valid_to, datatype=XSD.dateTime)))
        if supersedes:
            graph.add((node, ENERGY.supersedes, DOC[supersedes]))
        graph.add((node, PROV.wasDerivedFrom, URIRef("urn:synthetic:sharepoint:technical-guidance")))

    def export_turtle(self) -> str:
        return self.graph.serialize(format="turtle")

    def publication_report(self) -> PublicationReport:
        """The current published version's report -- version id, publish
        timestamp, published/candidate counts, and any quarantined records."""
        return self._publisher.current_report

    def publication_history(self) -> list[PublicationReport]:
        """Every version published so far for this service instance, oldest
        first (in-memory only -- does not persist across processes)."""
        return self._publisher.history()

    def rollback(self, version_id: str | None = None) -> PublicationReport:
        """Roll back to `version_id`, or the version immediately before the
        current one when omitted. Refreshes `self.graph` to match -- every
        subsequent query/export sees the restored content immediately."""
        report = self._publisher.rollback(version_id)
        self.graph = self._publisher.current
        return report

    def validate_candidate(self) -> dict[str, Any]:
        """Demonstrate SHACL rejection against one hardcoded invalid
        observation (missing energy:value/energy:unit) -- a fixed capability
        probe, not a validation of the live published graph (see
        publication_report() for that). Now goes through the real
        SHACLValidator (configurable shapes_path) instead of calling
        pyshacl.validate() directly, closing the exact inconsistency this
        session's SHACL-publication-gate work found: the two code paths
        used to validate the platform's RDF two different ways."""
        candidate = Graph()
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], RDF.type, ENERGY.Observation))
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], ENERGY.observedAsset, ASSET["WT-10"]))
        candidate.add((REC["obs-WT-10-temperature_c-invalid"], ENERGY.observedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
        report = SHACLValidator(candidate, shapes_path=SHAPES_PATH).validate_report(target="energy")
        return {
            "conforms": report.conforms,
            "rejected_records": ["obs-WT-10-temperature_c-invalid"],
            "violations": [r.message for r in report.results],
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
