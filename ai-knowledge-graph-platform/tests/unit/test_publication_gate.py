"""Phase 1: deterministic validation as a publication gate.

Proves invalid records are withheld from the published graph and quarantined
with rule ids, valid records are published, validation queries are read-only,
and the batch lifecycle is recorded on the run manifest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from graphrag.core.models import Chunk, Document, Entity, IngestionRunManifest, Relation
from graphrag.graph.validation import (
    RULES,
    PublicationGate,
    PublicationRejected,
    Severity,
    validate_batch,
    validate_document,
)
from graphrag.graph.validation import graph_checks
from graphrag.graph.validation.graph_checks import WriteQueryRejected, assert_read_only
from graphrag.graph.validation.quarantine_store import _UPSERT

T = "acme"


class FakeNeo4j:
    """Records writes (run) separately from reads (run_read)."""

    def __init__(self, read_rows: dict[str, list[dict]] | None = None):
        self.writes: list[tuple[str, dict]] = []
        self.reads: list[tuple[str, dict]] = []
        self._read_rows = read_rows or {}
        self.begin_corpus_update = AsyncMock()
        self.complete_corpus_update = AsyncMock(return_value=7)

    async def run(self, cypher, **params):
        self.writes.append((cypher, params))
        return []

    async def run_read(self, cypher, **params):
        self.reads.append((cypher, params))
        for marker, rows in self._read_rows.items():
            if marker in cypher:
                return rows
        return []

    def quarantine_rows(self) -> list[dict]:
        return [row for q, p in self.writes if q == _UPSERT for row in p["rows"]]


def ent(name="Acme", type_="ORG", **kw) -> Entity:
    return Entity(name=name, type=type_, tenant=T, **kw)


def rel(src: Entity, tgt: Entity, relation="OWNS", **kw) -> Relation:
    return Relation(source_entity_id=src.id, target_entity_id=tgt.id, relation=relation, **kw)


def rules_of(result) -> set[str]:
    return {rid for r in result.rejected for rid in r.rule_ids}


# ── pure batch validation ─────────────────────────────────────────────────────

def test_valid_batch_is_published_untouched():
    a, b = ent("Acme"), ent("Bolt")
    r = rel(a, b)
    result = validate_batch([a, b], [r], tenant=T, source="test")
    assert result.entities == [a, b] and result.relations == [r]
    assert result.rejected == []
    assert result.report.conforms


@pytest.mark.parametrize("mutate,rule_id", [
    (lambda e: setattr(e, "name", "  "), "ENT-REQ-001"),
    (lambda e: setattr(e, "type", ""), "ENT-REQ-002"),
    (lambda e: setattr(e, "tenant", "other-tenant"), "ENT-TENANT-001"),
    (lambda e: setattr(e, "confidence", 1.7), "ENT-CONF-001"),
    (lambda e: setattr(e, "confidence", float("nan")), "ENT-CONF-001"),
    (lambda e: setattr(e, "resolution_status", "merged_by_llm"), "ENT-ER-001"),
    (lambda e: setattr(e, "canonical_name", "Acme Corp"), "ENT-ER-001"),  # type missing
])
def test_blocking_entity_rules_quarantine_the_entity(mutate, rule_id):
    bad, ok = ent("Acme"), ent("Bolt")
    mutate(bad)
    result = validate_batch([bad, ok], [], tenant=T, source="test")
    assert bad not in result.entities and ok in result.entities
    assert rule_id in rules_of(result)
    q = result.rejected[0]
    assert q.record_kind == "entity" and q.payload["id"] == bad.id
    assert RULES[rule_id].severity is Severity.BLOCKING


def test_default_tenant_is_treated_as_unset_not_as_a_mismatch():
    e = Entity(name="Acme", type="ORG")  # tenant defaults to "default"
    assert validate_batch([e], [], tenant=T, source="test").rejected == []


def test_conflicting_identity_for_one_id_is_rejected():
    a = ent("Acme")
    clash = ent("Other")
    clash.id = a.id
    result = validate_batch([a, clash], [], tenant=T, source="test")
    assert "ENT-ID-001" in rules_of(result)
    assert result.entities == []


def test_relation_to_rejected_endpoint_is_cascaded_not_dangling():
    bad, ok = ent(""), ent("Bolt")
    r = rel(bad, ok)
    result = validate_batch([bad, ok], [r], tenant=T, source="test")
    assert result.relations == []
    rel_rec = next(x for x in result.rejected if x.record_kind == "relation")
    assert rel_rec.rule_ids == ["REL-REF-002"]


@pytest.mark.parametrize("build,rule_id", [
    (lambda a, b: Relation(source_entity_id=a.id, target_entity_id="ghost", relation="OWNS"), "REL-REF-001"),
    (lambda a, b: rel(a, a), "REL-SELF-001"),
    (lambda a, b: rel(a, b, relation=" "), "REL-REQ-001"),
    (lambda a, b: rel(a, b, confidence=-0.1), "REL-CONF-001"),
    (lambda a, b: rel(a, b, weight=float("inf")), "REL-CONF-001"),
    (lambda a, b: rel(a, b, valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                      valid_to=datetime(2024, 1, 1, tzinfo=timezone.utc)), "REL-TEMPORAL-001"),
    (lambda a, b: rel(a, b, confidence_state="MAYBE"), "REL-STATE-001"),
])
def test_blocking_relation_rules(build, rule_id):
    a, b = ent("Acme"), ent("Bolt")
    r = build(a, b)
    result = validate_batch([a, b], [r], tenant=T, source="test")
    assert r not in result.relations
    assert rule_id in rules_of(result)
    rec = next(x for x in result.rejected if x.record_kind == "relation")
    assert "source_endpoint" in rec.payload and "target_endpoint" in rec.payload


def test_same_name_and_type_endpoints_is_a_self_loop():
    a, b = ent("Acme"), ent("Acme")
    assert "REL-SELF-001" in rules_of(validate_batch([a, b], [rel(a, b)], tenant=T, source="t"))


def test_warnings_are_reported_but_published():
    a, b = ent("Acme", "WIDGET"), ent("Bolt")
    r = rel(a, b, relation="owns")  # writer normalises case
    result = validate_batch(
        [a, b], [r], tenant=T, source="test",
        allowed_entity_types={"ORG"},
        triplet_validator=lambda s, rel_, t: (False, rel_),
    )
    assert result.entities == [a, b] and result.relations == [r]
    found = {v.rule_id for v in result.report.violations}
    assert {"ENT-TYPE-001", "REL-REQ-002", "REL-DOMAIN-001"} <= found
    assert result.report.conforms


def test_property_rules_are_warnings():
    a = ent("Acme")
    checker = lambda name, typ, props: [{"type": "invalid_value", "constraint": "bad status"}]  # noqa: E731
    result = validate_batch([a], [], tenant=T, source="t", property_checker=checker)
    assert result.entities == [a]
    assert "ENT-PROP-002" in {v.rule_id for v in result.report.violations}


def test_semantic_validator_violations_block():
    a, b = ent("Acme"), ent("Bolt")
    sem = MagicMock()
    sem.validate_node.return_value = []
    sem.validate_relation.return_value = [MagicMock(code="MAX_CARDINALITY", message="too many")]
    result = validate_batch([a, b], [rel(a, b)], tenant=T, source="t", semantic_validator=sem)
    assert result.relations == []
    assert "SEM-VIOLATION-001" in rules_of(result)


@pytest.mark.parametrize("kw,chunk_tenant,rule_id", [
    ({"tenant": ""}, "", "DOC-TENANT-001"),
    ({"filename": "", "source_path": "", "content_hash": ""}, T, "DOC-PROV-001"),
    ({"valid_from": datetime(2025, 1, 1, tzinfo=timezone.utc),
      "valid_to": datetime(2020, 1, 1, tzinfo=timezone.utc)}, T, "DOC-TEMPORAL-001"),
    ({}, "intruder", "DOC-TENANT-002"),
])
def test_document_rules(kw, chunk_tenant, rule_id):
    doc = Document(**{"filename": "a.txt", "source_path": "a.txt", "raw_text": "x", "tenant": T, **kw})
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant=chunk_tenant)
    report = validate_document(doc, [chunk], source="t")
    assert rule_id in {v.rule_id for v in report.blocking}


# ── report ────────────────────────────────────────────────────────────────────

def test_report_counts_and_junit():
    bad, ok = ent(""), ent("Bolt")
    report = validate_batch([bad, ok], [rel(ok, ok)], tenant=T, source="pdf").report
    counts = report.counts()
    assert counts["by_rule"]["ENT-REQ-001"] == 1
    assert counts["by_rule"]["REL-SELF-001"] == 1
    assert counts["by_tenant"] == {T: len(report.violations)}
    assert counts["by_source"] == {"pdf": len(report.violations)}
    suite = ET.fromstring(report.to_junit_xml())
    failed = {c.get("name") for c in suite.iter("testcase") if c.find("failure") is not None}
    assert failed == {"ENT-REQ-001", "REL-SELF-001"}
    assert int(suite.get("tests")) == len(RULES)
    d = report.to_dict()
    assert d["conforms"] is False
    assert json.dumps(d)  # serialisable


def test_shacl_mapping_points_at_real_shapes():
    ttl = open("ontology/shapes/ingestion.shapes.ttl", encoding="utf-8").read()
    for r in RULES.values():
        if r.shacl_ref:
            assert r.shacl_ref.split("/")[0].split(":")[1] in ttl


# ── read-only guarantee ───────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "MATCH (n) SET n.x = 1",
    "MATCH (n) DETACH DELETE n",
    "MERGE (n:Entity {id: 1})",
    "CREATE (n)",
    "MATCH (n) REMOVE n.x",
    "CALL { MATCH (n) DELETE n }",
    "CALL apoc.periodic.iterate('x','y',{})",
    "MATCH (n) FOREACH (_ IN [1] | CREATE (m))",
    "LOAD CSV FROM 'f' AS row RETURN row",
])
def test_assert_read_only_rejects_writes(q):
    with pytest.raises(WriteQueryRejected):
        assert_read_only(q)


def test_assert_read_only_ignores_keywords_inside_strings_and_comments():
    assert_read_only("MATCH (d:Document {name: 'CREATE SET'}) // DELETE later\nRETURN d.dataset")


@pytest.mark.asyncio
async def test_every_validation_and_store_read_is_read_only():
    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    await graph_checks.check_supersedes(neo, tenant=T, document_key="k", supersedes=["x"], source="s")
    await gate.store.list(tenant=T)
    await gate.store.get(tenant=T, record_id="r")
    await gate.store.summary(tenant=T)
    assert neo.writes == []
    assert len(neo.reads) >= 4
    for q, _ in neo.reads:
        assert_read_only(q)
        assert "$tenant" in q  # tenant-scoped


# ── gate lifecycle ────────────────────────────────────────────────────────────

def _doc(**kw) -> Document:
    return Document(**{"filename": "report.pdf", "source_path": "report.pdf", "raw_text": "x", "tenant": T, **kw})


def _manifest(doc) -> IngestionRunManifest:
    return IngestionRunManifest(job_id="j", tenant=doc.tenant, filename=doc.filename,
                                content_hash="h", model_provider="p", model_version="v")


@pytest.mark.asyncio
async def test_stage_filters_invalid_records_and_records_lifecycle():
    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    doc = _doc()
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant=T)
    a, b, bad = ent("Acme"), ent("Bolt"), ent("")
    good_rel, bad_rel = rel(a, b), rel(bad, b)
    manifest = _manifest(doc)

    staged = await gate.stage(doc, [chunk], [([a, b, bad], [good_rel, bad_rel])], manifest)

    assert staged.extraction_results == [([a, b], [good_rel])]
    assert {r.record_key for r in staged.rejected} == {"ORG:", "-OWNS->Bolt"}
    assert manifest.stage_metrics["publication"]["state"] == "VALIDATED"
    assert manifest.stage_metrics["publication"]["quarantined"] == 2
    assert neo.writes == []  # nothing written before the caller publishes

    ids = await gate.quarantine(staged.rejected, doc=doc, document_id="doc-1", manifest=manifest)
    rows = neo.quarantine_rows()
    assert len(ids) == 2 and {r["id"] for r in rows} == set(ids)
    assert all(r["chunk_id"] == chunk.id for r in rows)
    assert {tuple(r["rule_ids"]) for r in rows} == {("ENT-REQ-001",), ("REL-REF-002",)}
    _, params = neo.writes[0]
    assert params["tenant"] == T and params["document_id"] == "doc-1"
    assert params["manifest_id"] == manifest.id

    PublicationGate.published(manifest, 2)
    assert manifest.stage_metrics["publication"]["state"] == "PUBLISHED"


@pytest.mark.asyncio
async def test_quarantine_ids_are_idempotent_across_reingest():
    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    doc = _doc()
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant=T)
    first = await gate.stage(doc, [chunk], [([ent("")], [])])
    second = await gate.stage(doc, [chunk], [([ent("")], [])])
    ids1 = await gate.quarantine(first.rejected, doc=doc, document_id="d")
    ids2 = await gate.quarantine(second.rejected, doc=doc, document_id="d")
    assert ids1 == ids2


@pytest.mark.asyncio
async def test_document_level_blocking_rejects_batch_and_quarantines_document():
    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    doc = _doc(valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
               valid_to=datetime(2020, 1, 1, tzinfo=timezone.utc))
    manifest = _manifest(doc)
    with pytest.raises(PublicationRejected) as exc:
        await gate.stage(doc, [], [], manifest)
    assert "DOC-TEMPORAL-001" in str(exc.value)
    assert manifest.stage_metrics["publication"]["state"] == "REJECTED"
    rows = neo.quarantine_rows()
    assert len(rows) == 1 and rows[0]["record_kind"] == "document"


@pytest.mark.asyncio
async def test_dangling_supersedes_is_a_warning_queried_in_tenant_only():
    neo = FakeNeo4j(read_rows={"MATCH (d:Document": [{"id": "known"}]})
    gate = PublicationGate(neo)
    doc = _doc(supersedes=["known", "missing"])
    staged = await gate.stage(doc, [], [])
    warn = [v for v in staged.report.violations if v.rule_id == "DOC-SUPERSEDES-001"]
    assert [v.message for v in warn] == ["supersedes missing"]
    q, params = neo.reads[0]
    assert params["tenant"] == T


def test_gate_disabled_by_setting():
    with patch("graphrag.graph.validation.gate.get_settings") as s:
        s.return_value.ingestion = {"publication_gate_enabled": False}
        assert PublicationGate.from_settings(FakeNeo4j()) is None
        s.return_value.ingestion = {}
        assert isinstance(PublicationGate.from_settings(FakeNeo4j()), PublicationGate)


# ── retry ─────────────────────────────────────────────────────────────────────

class _NoopMutation:
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _stored(record_kind: str, payload: dict, status="QUARANTINED") -> dict:
    return {"id": "rid", "record_kind": record_kind, "record_key": "k", "rule_ids": ["ENT-REQ-001"],
            "status": status, "chunk_id": "c1", "document_id": "d1", "payload": payload}


@pytest.mark.asyncio
async def test_retry_with_correction_publishes_through_writer():
    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    bad = ent("")
    gate.store.get = AsyncMock(return_value=_stored("entity", bad.model_dump(mode="json")))
    gate.store.mark_retry = AsyncMock()
    writer = MagicMock()
    writer.write_entities = AsyncMock(return_value=[])
    corrected = {**bad.model_dump(mode="json"), "name": "Acme"}

    with patch("graphrag.graph.corpus_revision.CorpusMutation", _NoopMutation):
        out = await gate.retry(tenant=T, record_id="rid", writer=writer, corrected=corrected)

    assert out["status"] == "RESOLVED"
    (entities, chunk), _ = writer.write_entities.call_args
    assert entities[0].name == "Acme" and chunk.id == "c1" and chunk.tenant == T
    assert gate.store.mark_retry.call_args.kwargs["resolved"] is True


@pytest.mark.asyncio
async def test_retry_still_invalid_stays_quarantined_and_writes_nothing():
    gate = PublicationGate(FakeNeo4j())
    bad = ent("")
    gate.store.get = AsyncMock(return_value=_stored("entity", bad.model_dump(mode="json")))
    gate.store.mark_retry = AsyncMock()
    writer = MagicMock()
    writer.write_entities = AsyncMock()
    out = await gate.retry(tenant=T, record_id="rid", writer=writer)
    assert out == {"id": "rid", "status": "QUARANTINED", "rule_ids": ["ENT-REQ-001"]}
    writer.write_entities.assert_not_called()
    assert gate.store.mark_retry.call_args.kwargs["resolved"] is False


@pytest.mark.asyncio
async def test_retry_cannot_move_a_record_into_another_tenant():
    gate = PublicationGate(FakeNeo4j())
    e = ent("Acme")
    gate.store.get = AsyncMock(return_value=_stored("entity", e.model_dump(mode="json")))
    gate.store.mark_retry = AsyncMock()
    writer = MagicMock()
    writer.write_entities = AsyncMock()
    out = await gate.retry(tenant=T, record_id="rid", writer=writer,
                           corrected={**e.model_dump(mode="json"), "tenant": "victim"})
    assert "ENT-TENANT-001" in out["rule_ids"]
    writer.write_entities.assert_not_called()


@pytest.mark.asyncio
async def test_retry_relation_requires_published_endpoints():
    neo = FakeNeo4j(read_rows={"OPTIONAL MATCH (e:Entity": [
        {"name": "Acme", "type": "ORG", "present": True},
        {"name": "Bolt", "type": "ORG", "present": False},
    ]})
    gate = PublicationGate(neo)
    a, b = ent("Acme"), ent("Bolt")
    r = rel(a, b)
    payload = {**r.model_dump(mode="json"),
               "source_endpoint": {"name": "Acme", "type": "ORG"},
               "target_endpoint": {"name": "Bolt", "type": "ORG"}}
    gate.store.get = AsyncMock(return_value=_stored("relation", payload))
    gate.store.mark_retry = AsyncMock()
    writer = MagicMock()
    writer.write_relations = AsyncMock()
    out = await gate.retry(tenant=T, record_id="rid", writer=writer)
    assert out["rule_ids"] == ["REL-REF-001"]
    writer.write_relations.assert_not_called()


@pytest.mark.asyncio
async def test_documents_and_resolved_records_are_not_retryable():
    gate = PublicationGate(FakeNeo4j())
    gate.store.get = AsyncMock(return_value=_stored("document", {}))
    with pytest.raises(ValueError, match="re-ingesting"):
        await gate.retry(tenant=T, record_id="rid", writer=MagicMock())
    gate.store.get = AsyncMock(return_value=_stored("entity", {}, status="RESOLVED"))
    with pytest.raises(ValueError, match="RESOLVED"):
        await gate.retry(tenant=T, record_id="rid", writer=MagicMock())
    gate.store.get = AsyncMock(return_value=None)
    with pytest.raises(LookupError):
        await gate.retry(tenant=T, record_id="rid", writer=MagicMock())


# ── ingestion agent integration ───────────────────────────────────────────────

def _agent_with_writer(neo):
    from graphrag.agents.ingestion_agent import IngestionAgent

    agent = IngestionAgent.__new__(IngestionAgent)
    writer = MagicMock()

    async def fake_write_document(doc):
        doc.id = "canonical-id"
        return "canonical-id"

    writer.write_document = AsyncMock(side_effect=fake_write_document)
    writer.neo4j_client = neo
    writer.document_has_evidence = AsyncMock(return_value=False)
    writer.write_chunks = AsyncMock()
    writer.write_entities = AsyncMock(return_value=[])
    writer.write_relations = AsyncMock()
    writer.write_ingestion_manifest = AsyncMock()
    writer.validate_and_check_cycles = AsyncMock(return_value={
        "validation": {"total_issues": 0}, "new_conflicts": 0})
    writer.mark_document_ingest_complete = AsyncMock()
    writer._ontology = None
    writer.drain_rejections = MagicMock(return_value=[])
    agent._writer = writer
    agent._publication_gate = PublicationGate(neo)
    return agent, writer


@pytest.mark.asyncio
async def test_agent_never_writes_blocking_records_and_quarantines_them():
    neo = FakeNeo4j()
    agent, writer = _agent_with_writer(neo)
    doc = _doc()
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant=T)
    a, b, bad = ent("Acme"), ent("Bolt"), ent("Acme")
    bad.confidence = 2.0
    good_rel, bad_rel = rel(a, b), rel(a, b, valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                        valid_to=datetime(2025, 1, 1, tzinfo=timezone.utc))
    manifest = _manifest(doc)

    with patch("graphrag.agents.ingestion_agent.get_settings") as s:
        s.return_value.wikidata_linking_enabled = False
        out = await agent.write({"job_id": "j", "doc": doc, "chunks": [chunk], "manifest": manifest,
                                 "extraction_results": [([a, b, bad], [good_rel, bad_rel])]})

    written_entities = writer.write_entities.call_args.args[0]
    written_relations = writer.write_relations.call_args.args[0]
    assert bad not in written_entities and written_entities == [a, b]
    assert written_relations == [good_rel]
    rows = neo.quarantine_rows()
    assert {tuple(r["rule_ids"]) for r in rows} == {("ENT-CONF-001",), ("REL-TEMPORAL-001",)}
    assert out["quarantined"] == 2
    assert manifest.stage_metrics["publication"]["state"] == "PUBLISHED"
    upsert_params = next(p for q, p in neo.writes if q == _UPSERT)
    assert upsert_params["document_id"] == "canonical-id"


@pytest.mark.asyncio
async def test_agent_document_rejection_writes_nothing_to_the_graph():
    neo = FakeNeo4j()
    agent, writer = _agent_with_writer(neo)
    doc = _doc()
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant="intruder")
    manifest = _manifest(doc)
    with pytest.raises(PublicationRejected):
        await agent.write({"job_id": "j", "doc": doc, "chunks": [chunk], "manifest": manifest,
                           "extraction_results": [([ent("Acme")], [])]})
    writer.write_document.assert_not_called()
    writer.write_chunks.assert_not_called()
    writer.write_entities.assert_not_called()
    neo.begin_corpus_update.assert_not_called()
    assert manifest.status == "failed"
    assert manifest.stage_metrics["publication"]["state"] == "REJECTED"


@pytest.mark.asyncio
async def test_agent_quarantines_post_resolution_writer_rejections():
    from graphrag.graph.validation import RejectedRecord

    neo = FakeNeo4j()
    agent, writer = _agent_with_writer(neo)
    writer.drain_rejections = MagicMock(return_value=[RejectedRecord(
        "relation", "Acme-OWNS->Bolt", ["REL-DOMAIN-002"], ["PERSON-OWNS->CITY"], {"x": 1}, "c")])
    doc = _doc()
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant=T)
    with patch("graphrag.agents.ingestion_agent.get_settings") as s:
        s.return_value.wikidata_linking_enabled = False
        out = await agent.write({"job_id": "j", "doc": doc, "chunks": [chunk],
                                 "extraction_results": [([], [])]})
    assert out["quarantined"] == 1
    assert [r["rule_ids"] for r in neo.quarantine_rows()] == [["REL-DOMAIN-002"]]


# ── graph writer records post-resolution refusals ─────────────────────────────

@pytest.mark.asyncio
async def test_graph_writer_records_dangling_and_domain_rejections():
    from graphrag.ingestion.graph_writer import GraphWriter

    writer = GraphWriter.__new__(GraphWriter)
    writer._neo4j = MagicMock()
    writer._neo4j.merge_relations_batch = AsyncMock()
    writer._audit = MagicMock()
    writer._audit.log_relations_batch = AsyncMock()
    writer._ensure_registry = AsyncMock()
    registry = MagicMock()
    registry.resolve.return_value = None
    writer._get_registry = MagicMock(return_value=registry)
    writer._semantic_validator = None
    writer._changed_by = "t"
    writer._ontology = MagicMock()
    writer._ontology.validate_relation_triplet.return_value = (False, "OWNS")
    writer._ontology.record_schema_event = AsyncMock()

    a, b = ent("Acme", "PERSON"), ent("Paris", "CITY")
    dangling = Relation(source_entity_id=a.id, target_entity_id="ghost", relation="OWNS")
    await writer.write_relations([rel(a, b), dangling], {a.id: a, b.id: b}, doc_id="d", tenant=T)

    rejected = writer.drain_rejections()
    assert sorted(r.rule_ids[0] for r in rejected) == ["REL-DOMAIN-002", "REL-REF-001"]
    assert writer.drain_rejections() == []
    rows = writer._neo4j.merge_relations_batch.call_args.args[0]
    assert rows == []


# ── relational path ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_relational_batch_is_rejected_and_quarantined_on_blocking_rule():
    from graphrag.ingestion.relational import _publication_gate_check

    neo = FakeNeo4j()
    writer = MagicMock()
    writer.neo4j_client = neo
    a = ent("Acme")
    with pytest.raises(ValueError, match="REL-SELF-001"):
        await _publication_gate_check(writer, [a], [rel(a, a)], tenant=T, source_id="erp")
    rows = neo.quarantine_rows()
    assert rows and rows[0]["rule_ids"] == ["REL-SELF-001"]


@pytest.mark.asyncio
async def test_relational_valid_batch_passes_without_writes():
    from graphrag.ingestion.relational import _publication_gate_check

    neo = FakeNeo4j()
    writer = MagicMock()
    writer.neo4j_client = neo
    a, b = ent("Acme"), ent("Bolt")
    await _publication_gate_check(writer, [a, b], [rel(a, b)], tenant=T, source_id="erp")
    assert neo.writes == []
