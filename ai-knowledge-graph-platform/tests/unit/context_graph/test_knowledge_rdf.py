"""RDF projection of published knowledge and SHACL validation with shapes
compiled from the canonical model ``ontology/models/contextual-knowledge.yaml``."""

from __future__ import annotations

from pathlib import Path

import pytest
from rdflib import Literal, URIRef
from rdflib.namespace import RDF

from graphrag.context_graph import knowledge_rdf
from graphrag.context_graph.knowledge_rdf import (
    KNOWLEDGE_MODEL_PATH, knowledge_model, project_release, validate_projection,
)
from graphrag.context_graph.knowledge_release import (
    KnowledgePublicationError, check_pack, load_cases, load_pack, publish_pack,
)
from graphrag.semantic_model import compile_to_disk

from .test_contextual_knowledge_audit import CASES, PACK, item

NS = knowledge_model().namespace


@pytest.fixture
def pack():
    return load_pack(PACK)


def _projection(pack):
    return project_release(pack.published_items(), pack.published_exceptions())


def _iri(kind: str, entry_id: str, tenant: str = "packaging_demo") -> URIRef:
    return URIRef(f"{NS}entry/{tenant}/{kind}/{entry_id}")


class TestProjection:
    def test_published_pack_conforms(self, pack):
        projection = _projection(pack)
        report = validate_projection(projection)
        assert report.conforms, report.violations
        graph = projection.graph
        assert len(set(graph.subjects(RDF.type, URIRef(NS + "KnowledgeItem")))) == len(pack.published_items())
        assert len(set(graph.subjects(RDF.type, URIRef(NS + "KnowledgeException")))) == \
            len(pack.published_exceptions())

    def test_only_published_entries_are_projected(self, pack):
        graph = _projection(pack).graph
        assert (_iri("item", "PKG-013"), None, None) not in graph  # draft
        assert (_iri("item", "PKG-014"), None, None) not in graph  # withdrawn

    def test_projection_uses_only_terms_declared_in_the_canonical_model(self, pack):
        model = knowledge_model()
        declared = set(model.relations)
        for name in model.types:
            declared |= set(model.effective_properties(name))
        graph = _projection(pack).graph
        used = {str(p)[len(NS):] for p in graph.predicates() if str(p).startswith(NS)}
        assert used and used <= declared, used - declared
        classes = {str(o)[len(NS):] for o in graph.objects(None, RDF.type)}
        assert classes <= set(model.types)
        assert not any(model.types[name].abstract for name in classes)

    def test_references_become_edges_between_released_items(self, pack):
        graph = _projection(pack).graph
        assert (_iri("item", "PKG-011"), URIRef(NS + "supersedes"), _iri("item", "PKG-002")) in graph
        assert (_iri("exception", "EXC-002"), URIRef(NS + "precedenceWinner"), _iri("item", "PKG-010")) in graph

    def test_reference_outside_the_release_is_reported_not_projected(self):
        newer = item("NEW", supersedes=("RETIRED",))
        projection = project_release([newer])
        assert projection.omitted_references == ("NEW@v1 supersedes RETIRED",)
        assert validate_projection(projection).conforms

    def test_unknown_applicability_asserts_no_scope(self, pack):
        graph = _projection(pack).graph
        partial = _iri("item", "PKG-016")  # declares no brand scope
        dimensions = {str(graph.value(scope, URIRef(NS + "dimension")))
                      for scope in graph.objects(partial, URIRef(NS + "hasScope"))}
        assert "brand" not in dimensions and "market" in dimensions


class TestShapes:
    def test_committed_artifacts_match_the_canonical_model(self):
        compile_to_disk(KNOWLEDGE_MODEL_PATH, check=True, fail_on_error=True)

    @pytest.mark.parametrize("predicate", ["citesSource", "reviewedBy", "modality"])
    def test_missing_required_structure_is_a_violation(self, pack, predicate):
        projection = _projection(pack)
        projection.graph.remove((_iri("item", "PKG-001"), URIRef(NS + predicate), None))
        report = validate_projection(projection)
        assert not report.conforms
        assert any("PKG-001" in v and predicate in v for v in report.violations)

    def test_wrong_datatype_is_a_violation(self, pack):
        projection = _projection(pack)
        subject = _iri("item", "PKG-001")
        projection.graph.set((subject, URIRef(NS + "version"), Literal("one")))
        assert not validate_projection(projection).conforms

    def test_edge_to_a_non_item_is_a_violation(self, pack):
        projection = _projection(pack)
        projection.graph.add((_iri("exception", "EXC-001"), URIRef(NS + "exceptionTarget"),
                              _iri("exception", "EXC-003")))
        assert not validate_projection(projection).conforms


class TestReleaseGate:
    @pytest.fixture
    def stricter_model(self, tmp_path, monkeypatch) -> Path:
        """The canonical model with ``series`` made mandatory: most synthetic
        items have none, so the gate must refuse them through SHACL alone."""
        text = KNOWLEDGE_MODEL_PATH.read_text(encoding="utf-8").replace(
            "      series:\n        datatype: string\n",
            "      series:\n        datatype: string\n        required: true\n")
        assert text != KNOWLEDGE_MODEL_PATH.read_text(encoding="utf-8")
        path = tmp_path / "contextual-knowledge.yaml"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(knowledge_rdf, "KNOWLEDGE_MODEL_PATH", path)
        return path

    def test_gate_reports_shape_violations(self, pack, stricter_model):
        failures = check_pack(pack, load_cases(CASES))
        assert failures and all(f.startswith("shacl: ") for f in failures)
        assert any("PKG-001" in f and "series" in f for f in failures)

    async def test_publication_is_refused_and_checks_record_shacl(self, pack, stricter_model, tmp_path):
        from graphrag.domains.energy.governance_store import GovernanceStore

        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                                blob_root=tmp_path / "blobs")
        await store.open()
        try:
            with pytest.raises(KnowledgePublicationError, match="shacl"):
                await publish_pack(store, pack, published_by="user:p", cases=load_cases(CASES))
            assert await store.active_knowledge_release("packaging_demo", "packaging_labeling") is None
        finally:
            await store.close()

    async def test_conforming_release_records_the_shacl_check(self, pack, tmp_path):
        from graphrag.domains.energy.governance_store import GovernanceStore

        store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}",
                                blob_root=tmp_path / "blobs")
        await store.open()
        try:
            release = await publish_pack(store, pack, published_by="user:p", cases=load_cases(CASES))
        finally:
            await store.close()
        assert release["checks"]["shacl"] == "passed"
