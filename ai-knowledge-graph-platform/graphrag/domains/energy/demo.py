"""Deterministic, evidence-backed wind-farm RDF demonstration.

The module owns the synthetic POC data only.  It does not write to Neo4j and
does not connect to SAP, Snowflake, or SharePoint; those source-shaped records
are explicitly synthetic exports for reproducible local demonstrations.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from graphrag.domains.energy import answers as energy_answers
from graphrag.domains.energy.answers import Evidence
from graphrag.domains.energy.fixtures import create_sap_fixture_sqlite
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.domains.energy.publication import DatasetPublisher, PublicationReport
from graphrag.domains.energy.vocabulary import (
    ASSET, DOC, ENERGY, PROV, REC, TENANT, parse_instant,
)
from graphrag.graph.shacl_validator import SHACLValidator
from graphrag.graph.sparql_bridge import SPARQLBridge
from graphrag.ingestion.r2rml_rdf import materialize_r2rml
from graphrag.ingestion.relational import SQLiteSourceConnector
from graphrag.ingestion.rml_rdf import materialize_rml

ROOT = Path(__file__).resolve().parents[3]
SHAPES_PATH = ROOT / "ontology" / "shapes" / "energy-asset-intelligence.shapes.ttl"

# Namespaces, TENANT and Evidence are re-exported (they now live in
# vocabulary.py / answers.py) so existing importers -- workflow.py,
# lpg_projection.py, scripts and tests doing
# `from graphrag.domains.energy.demo import ENERGY, REC, TENANT` -- keep
# working unchanged.
_utc = parse_instant

__all__ = [
    "ASSET", "DOC", "ENERGY", "Evidence", "EnergyDemoService", "PROV", "REC",
    "SHAPES_PATH", "TENANT",
]


class EnergyDemoService:
    """Answer five fixed, reviewable business questions over an RDF graph.

    The question *ids* are fixed; the answers are not. Every answer and every
    citation is derived from the published graph by
    ``graphrag/domains/energy/answers.py`` through version-controlled SPARQL.
    """

    questions = energy_answers.QUESTIONS
    tenant = TENANT

    def __init__(self, source_db: Path | None = None, *, include_invalid_fixture: bool = False) -> None:
        # _build_graph() awaits materialize_r2rml()/materialize_rml() directly
        # rather than each nesting its own asyncio.run() call. That makes this
        # constructor the ONLY place that starts an event loop for the sync
        # path -- safe as long as no loop is already running in this thread.
        # If one is (an async caller: FastAPI startup under a loop, an async
        # script, an async test under pytest-asyncio), asyncio.run() below
        # would raise its own cryptic "cannot be called from a running event
        # loop" RuntimeError; fail with a clear pointer to the real fix
        # instead of leaving that to surface from deep inside construction.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no loop running in this thread -- safe to construct synchronously
        else:
            raise RuntimeError(
                "EnergyDemoService() cannot be constructed synchronously from "
                "inside a running event loop. Use "
                "`await EnergyDemoService.create(source_db=...)` instead."
            )
        self.source_db = source_db
        self._publisher = DatasetPublisher(SHAPES_PATH)
        self._governance_store = None
        self._include_invalid_fixture = include_invalid_fixture
        self._durable_report: PublicationReport | None = None
        candidate = asyncio.run(self._build_graph())
        self._publish(candidate)

    @classmethod
    async def create(
        cls, source_db: Path | None = None, *, governance_store: GovernanceStore | None = None,
        include_invalid_fixture: bool = False,
    ) -> "EnergyDemoService":
        """Async canonical constructor: awaits graph construction directly
        instead of nesting asyncio.run(). Use this from any caller that
        already owns a running event loop -- an async CLI entry point,
        FastAPI startup, an async test -- and use the synchronous constructor
        everywhere else."""
        self = cls.__new__(cls)
        self.source_db = source_db
        self._publisher = DatasetPublisher(SHAPES_PATH)
        self._governance_store = governance_store
        self._include_invalid_fixture = include_invalid_fixture
        candidate = await self._build_graph()
        self._publish(candidate)
        self._durable_report: PublicationReport | None = None
        if governance_store is not None:
            report = await governance_store.publish(self.tenant, self._publisher.current_report, self.graph)
            _stored_report, self.graph = await governance_store.current(self.tenant)
            self._durable_report = report
        return self

    def _publish(self, candidate: Graph) -> None:
        # SHACL as a publication gate: the candidate graph is staged,
        # validated, and only the conformant subset is published -- any
        # record that violates ontology/shapes/energy-asset-intelligence.shapes.ttl
        # is quarantined (see graphrag/domains/energy/publication.py), not
        # silently served. Every existing query path (answer(),
        # export_turtle()) now only ever sees this published graph.
        self._publisher.stage_and_publish(candidate)
        self.graph = self._publisher.current

    async def _build_graph(self) -> Graph:
        graph = Graph()
        graph.bind("energy", ENERGY)
        graph.bind("prov", PROV)
        site = ASSET["north-sea-wind-farm"]
        graph.add((site, RDF.type, ENERGY.Site))
        graph.add((site, RDFS.label, Literal("North Sea Demonstration Wind Farm")))
        await self._materialize_assets_and_work_orders(graph)
        for index in range(1, 11):
            turbine = ASSET[f"WT-{index:02d}"]
            gearbox = ASSET[f"WT-{index:02d}-gearbox"]
            graph.add((site, ENERGY.hasAsset, turbine))
            graph.add((gearbox, RDF.type, ENERGY.Component))
            graph.add((gearbox, ENERGY.componentType, Literal("gearbox")))
            graph.add((turbine, ENERGY.hasComponent, gearbox))
        graph += await materialize_rml(
            ROOT / "ontology/mappings/energy-observations.rml.ttl", ROOT,
        )
        self._add_bulletin(
            graph, "MFG-GBX-17-R1", "2026-01-01T00:00:00Z", "2026-06-01T00:00:00Z", 90.0, None,
            recorded_at="2026-01-01T00:00:00Z",
        )
        self._add_bulletin(
            graph, "MFG-GBX-17-R2", "2026-06-01T00:00:00Z", None, 85.0, "MFG-GBX-17-R1",
            recorded_at="2026-06-01T00:00:00Z",
        )
        if self._include_invalid_fixture:
            self._add_invalid_observation(graph)
        return graph

    @staticmethod
    def _add_invalid_observation(graph: Graph) -> None:
        """One deliberately-invalid Observation -- opt-in only, never part of
        the default candidate graph.

        Same shape `validate_candidate()` already uses as its fixed SHACL
        capability probe (missing `energy:value`/`energy:unit`, a known-safe,
        non-cascading violation), but fed through the *real* publication
        pipeline this time, plus a `prov:wasDerivedFrom` triple. Exists so the
        quarantine audit UI and its tests have one genuine quarantined record
        to render, without changing what the standard (flag-off) demo
        publishes -- every existing zero-quarantine assertion elsewhere
        keeps holding because nothing calls this by default.
        """
        subject = REC["obs-WT-10-temperature_c-invalid"]
        graph.add((subject, RDF.type, ENERGY.Observation))
        graph.add((subject, ENERGY.observedAsset, ASSET["WT-10"]))
        graph.add((subject, ENERGY.observedAt, Literal("2026-08-28T08:00:00Z", datatype=XSD.dateTime)))
        graph.add((subject, PROV.wasDerivedFrom, URIRef("urn:synthetic:snowflake:telemetry-invalid-sample")))

    async def _materialize_assets_and_work_orders(self, graph: Graph) -> None:
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
            graph += await materialize_r2rml(mapping_path, SQLiteSourceConnector(self.source_db))
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            ephemeral_db = Path(tmp_dir) / "energy-demo-sap.sqlite"
            create_sap_fixture_sqlite(ephemeral_db)
            graph += await materialize_r2rml(mapping_path, SQLiteSourceConnector(ephemeral_db))

    @staticmethod
    def _add_bulletin(
        graph: Graph, bulletin: str, valid_from: str, valid_to: str | None, threshold: float,
        supersedes: str | None, *, recorded_at: str,
    ) -> None:
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
        # Bitemporal recorded-time axis: when the platform learned this
        # revision existed, distinct from validFrom/validTo (valid-time: when
        # its guidance applies). Equal to valid_from here -- no publication
        # lag in this baseline; see the plan's Phase 4 for a genuine
        # late-arriving bulletin scenario.
        graph.add((node, ENERGY.recordedAt, Literal(recorded_at, datatype=XSD.dateTime)))
        if supersedes:
            graph.add((node, ENERGY.supersedes, DOC[supersedes]))
        graph.add((node, PROV.wasDerivedFrom, URIRef("urn:synthetic:sharepoint:technical-guidance")))

    def export_turtle(self) -> str:
        return self.graph.serialize(format="turtle")

    def publication_report(self) -> PublicationReport:
        """The current published version's report -- version id, publish
        timestamp, published/candidate counts, and any quarantined records."""
        return self._durable_report or self._publisher.current_report

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

    async def rollback_durable(self, version_id: str | None = None) -> PublicationReport:
        """Append a durable rollback version and refresh the serving graph."""
        if self._governance_store is None:
            return self.rollback(version_id)
        report = await self._governance_store.rollback(self.tenant, version_id)
        _stored_report, self.graph = await self._governance_store.current(self.tenant)
        self._durable_report = report
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

    def answer(
        self, question_id: str, *, tenant: str,
        as_of: str | None = None, known_as: str | None = None,
    ) -> dict[str, Any]:
        """Derive the answer and its citations from the published graph.

        Every value in the rendered answer and in every evidence row comes
        from a version-controlled SPARQL query over `self.graph` -- see
        graphrag/domains/energy/answers.py for what this replaced and why.

        `as_of` selects the valid-time instant and `known_as` the
        recorded-time instant; omitting `known_as` alongside an `as_of`
        answers "as we knew it then", which is what keeps a historical answer
        from citing evidence recorded after the instant it asks about.
        """
        try:
            return energy_answers.answer(
                self.graph, question_id, tenant=tenant, dataset_tenant=self.tenant,
                as_of=as_of, known_as=known_as,
            )
        except energy_answers.EnergyAnswerError as exc:
            # Preserve the historical contract: an unknown question id is a
            # plain ValueError, which api/routes/energy_demo.py turns into a
            # 404. EnergyAnswerError already subclasses ValueError; this
            # re-raise keeps the message stable for existing callers.
            raise ValueError(str(exc)) from exc

    def summary(self, *, tenant: str, as_of: str | None = None) -> dict[str, Any]:
        """Derived headline counts for the operations dashboard.

        Exists so the UI's summary tiles come from the same queries the
        answers do, instead of the hard-coded "1 / 2 / 3 of 10 / Checked"
        figures they used to display regardless of the data.
        """
        review = self.answer("maintenance_review", tenant=tenant, as_of=as_of)
        work_orders = self.answer("open_work_orders", tenant=tenant, as_of=as_of)
        incomplete = self.answer("insufficient_evidence", tenant=tenant, as_of=as_of)
        assets = set(self.graph.subjects(RDF.type, ENERGY.WindTurbine))
        observed = {
            observation
            for observation in self.graph.subjects(RDF.type, ENERGY.Observation)
        }
        assets_with_telemetry = {
            self.graph.value(observation, ENERGY.observedAsset) for observation in observed
        }
        return {
            "assets_under_review": len(review.get("query_rows", [])),
            "open_work_orders": len(work_orders.get("query_rows", [])),
            "assets_with_telemetry": len(assets_with_telemetry - {None}),
            "assets_total": len(assets),
            "assets_blocked_on_evidence": len(incomplete.get("query_rows", [])),
            "authoritative_bulletin": review.get("authoritative_bulletin", ""),
        }

    def bulletin_history(self) -> list[dict[str, Any]]:
        """Every guidance revision with its validity window and threshold.

        Replaces the dashboard's hard-coded "Before 1 Jun: R1 threshold 90°C
        / Current: R2 threshold 85°C" panel with the graph's own revisions.
        """
        rows = SPARQLBridge(self.graph).query(
            """
            PREFIX energy: <https://example.energy.demo/ontology#>
            SELECT ?bulletinId ?threshold ?validFrom ?validTo ?componentType WHERE {
              ?bulletin a energy:DocumentRevision ;
                        energy:documentId ?bulletinId ;
                        energy:temperatureReviewThreshold ?threshold ;
                        energy:appliesToComponentType ?componentType ;
                        energy:validFrom ?validFrom .
              OPTIONAL { ?bulletin energy:validTo ?validTo }
            }
            ORDER BY ?validFrom
            """
        )
        return rows
