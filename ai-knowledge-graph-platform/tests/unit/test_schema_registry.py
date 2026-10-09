"""Phase 2: versioned schema registry, drift detection, schema provenance.

Cypher semantics (single active version, CONFORMS_TO repointing, retained
history) are proven against a live Neo4j in tests/e2e/test_live_schema_registry.py.
These tests cover the deterministic hash, the drift/enforce policy, the
registry's tenant scoping, and the provenance plumbing, with scripted fakes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph import schema_registry as sr
from graphrag.graph.ontology_registry import OntologyRegistry, _RELATION_RULES
from graphrag.graph.schema_registry import (
    SchemaDriftError,
    SchemaIdentity,
    SchemaRegistry,
    UnknownSchemaError,
    compute_content_hash,
    dataset_id_for,
    profile_hash,
)


def _hash(**over) -> str:
    base = dict(
        allowed_types={"ORG", "PERSON"},
        domain_rules={"OWNS": {("ORG", "ORG")}},
        builtin_rules=_RELATION_RULES,
        vocabulary={"ORG": {"definition": "an organisation"}},
        migration_map={"IS_CEO": "CEO_OF"},
        ontology_doc={"ontology": {"id": "x", "version": "1.0.0"}, "relation_rules": {"OWNS": {}}},
        profile_hash_="p1",
    )
    base.update(over)
    return compute_content_hash(**base)


def _identity(tenant="acme", h="a" * 64, dataset=None) -> SchemaIdentity:
    return SchemaIdentity(tenant=tenant, dataset_id=dataset or tenant, name="acme-onto", version="1.2.0",
                          content_hash=h, profile_hash="p", types=("ORG",))


class ScriptedNeo4j:
    """Returns canned version rows for the lookup query; records every call."""

    def __init__(self, versions_by_tenant: dict[str, list[dict]] | None = None, upsert_rows=None):
        self.versions_by_tenant = versions_by_tenant or {}
        self.upsert_rows = upsert_rows if upsert_rows is not None else [
            {"version_id": "v-new", "created": True, "was_active": False, "prior_hashes": []}]
        self.calls: list[tuple[str, dict]] = []

    async def run(self, cypher, **params):
        self.calls.append((cypher, params))
        if cypher is sr.UPSERT_AND_ACTIVATE:
            return self.upsert_rows
        if "RETURN o.id AS id, coalesce(o.content_hash" in cypher:
            return self.versions_by_tenant.get(params["tenant"], [])
        return []


def v(id_, h, active=False, deactivated_at=None, activated_at=None) -> dict:
    return {"id": id_, "content_hash": h, "name": "n", "version": "1", "active": active,
            "source_uri": "", "created_at": "2026-01-01T00:00:00Z",
            "activated_at": activated_at, "deactivated_at": deactivated_at}


@pytest.fixture(autouse=True)
def _reset():
    SchemaRegistry.invalidate_labels()
    yield
    SchemaRegistry.invalidate_labels()


# ── deterministic content hash ────────────────────────────────────────────────

def test_hash_is_deterministic_and_order_independent():
    a = _hash(allowed_types={"ORG", "PERSON"}, migration_map={"A": "B", "C": "D"})
    b = _hash(allowed_types={"PERSON", "ORG"}, migration_map={"C": "D", "A": "B"})
    assert a == b and len(a) == 64


@pytest.mark.parametrize("change", [
    {"allowed_types": {"ORG", "PERSON", "EVENT"}},
    {"domain_rules": {"OWNS": {("ORG", "PERSON")}}},
    {"builtin_rules": {**_RELATION_RULES, "FOUNDED": set()}},
    {"vocabulary": {"ORG": {"definition": "changed"}}},
    {"migration_map": {"IS_CEO": "CEO_OF", "X": "Y"}},
    {"ontology_doc": {"ontology": {"id": "x", "version": "1.0.1"}, "relation_rules": {"OWNS": {}}}},
    {"ontology_doc": {"ontology": {"id": "x", "version": "1.0.0"}, "relation_rules": {"OWNS": {"note": "n"}}}},
    {"ontology_doc": {"ontology": {"id": "x", "version": "1.0.0"}, "relation_rules": {"OWNS": {}},
                      "inference_rules": [{"r": 1}]}},
    {"profile_hash_": "p2"},
])
def test_every_contract_component_changes_the_hash(change):
    assert _hash(**change) != _hash()


def test_ignored_yaml_keys_do_not_change_the_hash():
    doc = {"ontology": {"id": "x", "version": "1.0.0"}, "relation_rules": {"OWNS": {}}}
    assert _hash(ontology_doc={**doc, "authority_levels": {"a": 1}}) == _hash(ontology_doc=doc)


def test_profile_hash_tracks_shape_file_content(tmp_path):
    for n in ("ingestion.shapes.ttl", "export.shapes.ttl"):
        (tmp_path / n).write_text("a", encoding="utf-8")
    h1 = profile_hash(tmp_path)
    (tmp_path / "export.shapes.ttl").write_text("b", encoding="utf-8")
    assert profile_hash(tmp_path) != h1
    assert profile_hash(tmp_path / "missing") == profile_hash(tmp_path / "also-missing")


def test_dataset_defaults_to_tenant_and_honours_mapping():
    assert dataset_id_for("acme") == "acme"
    assert dataset_id_for("acme", {"schema_registry": {"datasets": {"acme": "finance"}}}) == "finance"
    assert dataset_id_for("other", {"schema_registry": {"datasets": {"acme": "finance"}}}) == "other"


# ── check / enforce ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_statuses():
    ident = _identity(h="b" * 64)
    reg = SchemaRegistry(ScriptedNeo4j({"acme": []}))
    assert (await reg.check(ident))["status"] == "unregistered"

    reg = SchemaRegistry(ScriptedNeo4j({"acme": [v("1", "b" * 64, active=True)]}))
    assert (await reg.check(ident))["status"] == "match"

    reg = SchemaRegistry(ScriptedNeo4j({"acme": [v("1", "a" * 64, active=True)]}))
    res = await reg.check(ident)
    assert res["status"] == "drift" and res["known"] is False

    reg = SchemaRegistry(ScriptedNeo4j({"acme": [v("1", "a" * 64, active=True), v("0", "b" * 64)]}))
    res = await reg.check(ident)
    assert res["status"] == "drift" and res["known"] is True


@pytest.mark.asyncio
async def test_enforce_blocks_unknown_and_inactive_schemas_but_not_match_or_bootstrap():
    ident = _identity(h="b" * 64)
    active_other = [v("1", "a" * 64, active=True)]

    with pytest.raises(UnknownSchemaError):
        await SchemaRegistry(ScriptedNeo4j({"acme": active_other})).enforce(ident, mode=sr.ENFORCE)
    with pytest.raises(SchemaDriftError):
        await SchemaRegistry(ScriptedNeo4j({"acme": active_other + [v("0", "b" * 64)]})).enforce(
            ident, mode=sr.ENFORCE)
    # match and first-ever registration (nothing to drift from) are allowed
    await SchemaRegistry(ScriptedNeo4j({"acme": [v("1", "b" * 64, active=True)]})).enforce(ident, mode=sr.ENFORCE)
    await SchemaRegistry(ScriptedNeo4j({"acme": []})).enforce(ident, mode=sr.ENFORCE)
    # auto mode reports but never raises
    res = await SchemaRegistry(ScriptedNeo4j({"acme": active_other})).enforce(ident, mode=sr.AUTO)
    assert res["status"] == "drift"


# ── activation, deactivation, rollback, tenant isolation ─────────────────────

@pytest.mark.asyncio
async def test_register_and_activate_is_one_tenant_scoped_statement():
    neo = ScriptedNeo4j()
    out = await SchemaRegistry(neo).register_and_activate(_identity())
    assert out == {"version_id": "v-new", "created": True, "was_active": False, "prior_hashes": []}
    assert len(neo.calls) == 1
    cypher, p = neo.calls[0]
    assert p["tenant"] == "acme" and p["dataset_id"] == "acme" and p["content_hash"] == "a" * 64
    assert "$tenant" in cypher and "Dataset" in cypher and "CONFORMS_TO" in cypher
    assert "USES_VALIDATION_PROFILE" in cypher and "SchemaVersion" in cypher


@pytest.mark.asyncio
async def test_activate_unknown_or_foreign_tenant_version_is_refused():
    neo = ScriptedNeo4j({"acme": [v("mine", "a" * 64)], "victim": [v("theirs", "c" * 64)]})
    reg = SchemaRegistry(neo)
    with pytest.raises(UnknownSchemaError):
        await reg.activate("acme", "acme", "theirs")  # exists, but under another tenant
    assert all(p["tenant"] == "acme" for _, p in neo.calls)
    assert not any("SET o.active = true" in c for c, _ in neo.calls)  # nothing was written


@pytest.mark.asyncio
async def test_activate_already_active_is_a_noop():
    neo = ScriptedNeo4j({"acme": [v("1", "a" * 64, active=True)]})
    assert await SchemaRegistry(neo).activate("acme", "acme", "1") == {"version_id": "1", "changed": False}
    assert not any("SET o.active = true" in c for c, _ in neo.calls)


@pytest.mark.asyncio
async def test_rollback_reactivates_most_recently_deactivated_version():
    neo = ScriptedNeo4j({"acme": [
        v("3", "c" * 64, active=True),
        v("2", "b" * 64, deactivated_at="2026-03-01T00:00:00Z"),
        v("1", "a" * 64, deactivated_at="2026-02-01T00:00:00Z"),
    ]})
    out = await SchemaRegistry(neo).rollback("acme", "acme")
    assert out["version_id"] == "2"
    write = next(p for c, p in neo.calls if "SET o.active = true" in c)
    assert write["id"] == "2" and write["tenant"] == "acme"


@pytest.mark.asyncio
async def test_rollback_without_history_fails_cleanly():
    with pytest.raises(UnknownSchemaError):
        await SchemaRegistry(ScriptedNeo4j({"acme": [v("1", "a" * 64, active=True)]})).rollback("acme", "acme")


@pytest.mark.asyncio
async def test_deactivate_is_tenant_and_dataset_scoped():
    neo = ScriptedNeo4j()
    neo.run = AsyncMock(return_value=[{"n": 1}])
    assert await SchemaRegistry(neo).deactivate("acme", "finance") == 1
    cypher, = neo.run.await_args.args
    kw = neo.run.await_args.kwargs
    assert kw == {"tenant": "acme", "dataset_id": "finance"}
    assert "o.active = false" in cypher and "$tenant" in cypher


@pytest.mark.asyncio
async def test_imports_are_tenant_scoped_and_not_self_referential():
    neo = ScriptedNeo4j()
    neo.run = AsyncMock(return_value=[{"n": 1}])
    assert await SchemaRegistry(neo).add_import("acme", "a", "b") is True
    q = neo.run.await_args.args[0]
    assert q.count("tenant: $tenant") == 2 and "IMPORTS" in q
    with pytest.raises(sr.SchemaError):
        await SchemaRegistry(neo).add_import("acme", "a", "a")


@pytest.mark.asyncio
async def test_active_label_is_cached_and_tenant_keyed():
    neo = ScriptedNeo4j({"acme": [{**v("1", "d" * 64, active=True), "name": "acme-onto", "version": "1.2.0"}],
                         "other": []})
    reg = SchemaRegistry(neo)
    assert await reg.active_label("acme") == "acme-onto@1.2.0#" + "d" * 12
    assert await reg.active_label("acme") == "acme-onto@1.2.0#" + "d" * 12
    assert len(neo.calls) == 1
    assert await reg.active_label("other") is None


# ── OntologyRegistry.load integration ────────────────────────────────────────

def _neo(upsert_rows):
    neo = AsyncMock()
    calls = {"i": 0}

    async def run(cypher, **params):
        calls["i"] += 1
        if cypher is sr.UPSERT_AND_ACTIVATE:
            return upsert_rows
        if "RETURN o.id AS id, coalesce(o.content_hash" in cypher:
            return neo.versions
        return []

    neo.run = AsyncMock(side_effect=run)
    neo.run_read = neo.run  # the real client reads through run_read
    neo.versions = []
    return neo


@pytest.mark.asyncio
async def test_load_registers_a_full_hash_with_dataset_and_profile():
    neo = _neo([{"version_id": "v1", "created": True, "was_active": False, "prior_hashes": []}])
    reg = OntologyRegistry(neo, tenant="aerospace")
    await reg.load(["ORG"])
    upsert = next(c for c in neo.run.await_args_list if c.args[0] is sr.UPSERT_AND_ACTIVATE)
    p = upsert.kwargs
    assert len(p["content_hash"]) == 64 and p["tenant"] == "aerospace" and p["dataset_id"] == "aerospace"
    assert p["name"] and p["version"] and p["source_uri"].endswith(".yml")  # aerospace YAML found
    assert reg.version_id == "v1"
    assert reg.schema_label == f"{p['name']}@{p['version']}#{p['content_hash'][:12]}"
    assert reg.schema_identity.content_hash == p["content_hash"]


@pytest.mark.asyncio
async def test_load_same_content_twice_yields_the_same_hash():
    hashes = []
    for _ in range(2):
        neo = _neo([{"version_id": "v1", "created": True, "was_active": False, "prior_hashes": []}])
        await OntologyRegistry(neo, tenant="aerospace").load(["ORG"])
        hashes.append(next(c for c in neo.run.await_args_list
                           if c.args[0] is sr.UPSERT_AND_ACTIVATE).kwargs["content_hash"])
    assert hashes[0] == hashes[1]


@pytest.mark.asyncio
async def test_load_with_a_changed_schema_records_drift_event():
    neo = _neo([{"version_id": "v2", "created": True, "was_active": False, "prior_hashes": ["f" * 64]}])
    reg = OntologyRegistry(neo, tenant="acme")
    reg.record_schema_event = AsyncMock()
    with patch.object(sr, "record_drift") as metric:
        await reg.load(["ORG"])
    reg.record_schema_event.assert_awaited_once()
    assert reg.record_schema_event.await_args.kwargs["event_type"] == "schema_drift"
    metric.assert_called_once_with("version_change")


@pytest.mark.asyncio
async def test_load_reactivating_a_known_version_is_reported_as_rollback():
    neo = _neo([{"version_id": "v1", "created": False, "was_active": False, "prior_hashes": ["f" * 64]}])
    reg = OntologyRegistry(neo, tenant="acme")
    reg.record_schema_event = AsyncMock()
    with patch.object(sr, "record_drift") as metric:
        await reg.load(["ORG"])
    metric.assert_called_once_with("rollback")


@pytest.mark.asyncio
async def test_load_unchanged_schema_is_not_drift():
    neo = _neo([{"version_id": "v1", "created": False, "was_active": True, "prior_hashes": []}])
    reg = OntologyRegistry(neo, tenant="acme")
    reg.record_schema_event = AsyncMock()
    with patch.object(sr, "record_drift") as metric:
        await reg.load(["ORG"])
    reg.record_schema_event.assert_not_called()
    metric.assert_called_once_with("match")


@pytest.mark.asyncio
async def test_enforce_mode_refuses_an_unregistered_schema_before_writing_anything():
    neo = _neo([])
    neo.versions = [v("1", "e" * 64, active=True)]
    reg = OntologyRegistry(neo, tenant="acme")
    cfg = MagicMock()
    cfg.ontology = {"schema_registry": {"mode": "enforce"}, "migration_map": {}}
    with patch("graphrag.core.config.get_settings", return_value=cfg), \
         patch.object(sr, "record_drift") as metric:
        with pytest.raises(UnknownSchemaError):
            await reg.load(["ORG"])
    assert not any(c.args[0] is sr.UPSERT_AND_ACTIVATE for c in neo.run.await_args_list)
    assert not reg.is_loaded
    metric.assert_called_once_with("blocked")


@pytest.mark.asyncio
async def test_check_schema_drift_is_read_only():
    neo = _neo([])
    neo.versions = [v("1", "e" * 64, active=True)]
    res = await OntologyRegistry(neo, tenant="acme").check_schema_drift(["ORG"])
    assert res["status"] == "drift" and res["dataset_id"] == "acme"
    assert not any(c.args[0] is sr.UPSERT_AND_ACTIVATE for c in neo.run.await_args_list)


@pytest.mark.asyncio
async def test_startup_drift_check_reports_per_tenant_and_never_raises():
    neo = AsyncMock()

    async def run(cypher, **params):
        if "RETURN DISTINCT o.tenant" in cypher:
            return [{"tenant": "acme"}, {"tenant": "beta"}]
        if "RETURN o.id AS id" in cypher:
            return [v("1", "e" * 64, active=True)] if params["tenant"] == "acme" else []
        return []

    neo.run = AsyncMock(side_effect=run)
    neo.run_read = neo.run
    with patch.object(sr, "record_drift") as metric:
        results = await sr.startup_drift_check(neo)
    by_tenant = {r["tenant"]: r["status"] for r in results}
    assert by_tenant == {"acme": "drift", "beta": "unregistered"}
    metric.assert_called_once_with("startup_drift")

    broken = AsyncMock()
    broken.run = AsyncMock(side_effect=RuntimeError("neo4j down"))
    broken.run_read = broken.run
    assert await sr.startup_drift_check(broken) == []


# ── provenance plumbing ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_schema_version_uses_active_label_and_degrades_with_backoff():
    from graphrag.retrieval import hybrid_retriever as hr

    hr._schema_lookup_failed_until = 0.0
    neo = ScriptedNeo4j({"acme": [{**v("1", "d" * 64, active=True), "name": "o", "version": "2"}]})
    with patch.object(hr, "get_neo4j", return_value=neo):
        assert await hr._resolve_schema_version("acme") == "o@2#" + "d" * 12
        assert await hr._resolve_schema_version("unregistered-tenant") == "platform/v1"

    SchemaRegistry.invalidate_labels()
    broken = MagicMock()
    broken.run = AsyncMock(side_effect=RuntimeError("down"))
    with patch.object(hr, "get_neo4j", return_value=broken):
        assert await hr._resolve_schema_version("acme") == "platform/v1"
        broken.run.reset_mock()
        assert await hr._resolve_schema_version("acme") == "platform/v1"
        broken.run.assert_not_called()  # backoff: no repeated stall
    hr._schema_lookup_failed_until = 0.0


def test_cache_key_changes_when_the_schema_version_changes():
    from graphrag.retrieval.query_cache import QueryCacheContext, build_cache_key

    def ctx(version):
        return QueryCacheContext(corpus_revision=1, requested_mode="hybrid", effective_mode="hybrid",
                                 model_route="m", prompt_version="p", retrieval_config={},
                                 ontology_version=version, valid_at=None, transaction_at=None)

    assert build_cache_key("q", "acme", ctx("o@1#aaa")) != build_cache_key("q", "acme", ctx("o@1#bbb"))
    assert build_cache_key("q", "acme", ctx("o@1#aaa")) == build_cache_key("q", "acme", ctx("o@1#aaa"))


def test_query_result_carries_schema_version():
    from graphrag.core.models import QueryResult

    assert QueryResult(question="q", answer="a").schema_version == ""
    assert QueryResult(question="q", answer="a", schema_version="o@1#abc").schema_version == "o@1#abc"


@pytest.mark.asyncio
async def test_validation_report_and_quarantine_carry_the_schema_version():
    from graphrag.core.models import Chunk, Document, Entity, IngestionRunManifest
    from graphrag.graph.validation import PublicationGate
    from tests.unit.test_publication_gate import FakeNeo4j

    neo = FakeNeo4j()
    gate = PublicationGate(neo)
    doc = Document(filename="f.txt", source_path="f.txt", raw_text="x", tenant="acme")
    chunk = Chunk(document_id=doc.id, text="x", chunk_index=0, tenant="acme")
    manifest = IngestionRunManifest(job_id="j", tenant="acme", filename="f.txt", content_hash="h",
                                    model_provider="p", model_version="v")
    registry = MagicMock()
    registry.schema_label = "acme-onto@1.2.0#abcdef123456"
    registry.is_loaded = False
    staged = await gate.stage(doc, [chunk], [([Entity(name="", type="ORG", tenant="acme")], [])],
                              manifest, registry=registry)
    assert staged.schema_version == "acme-onto@1.2.0#abcdef123456"
    assert staged.report.to_dict()["schema_version"] == staged.schema_version
    assert manifest.stage_metrics["publication"]["schema_version"] == staged.schema_version
    await gate.quarantine(staged.rejected, doc=doc, document_id="d", manifest=manifest,
                          schema_version=staged.schema_version)
    from graphrag.graph.validation.quarantine_store import _UPSERT
    params = next(p for q, p in neo.writes if q == _UPSERT)
    assert params["schema_version"] == staged.schema_version


@pytest.mark.asyncio
async def test_document_is_stamped_with_the_schema_version_it_was_ingested_under():
    from graphrag.ingestion.graph_writer import GraphWriter

    w = GraphWriter.__new__(GraphWriter)
    w._neo4j = MagicMock()
    w._neo4j.run = AsyncMock()
    ident = _identity()
    registry = MagicMock()
    registry.schema_identity = ident
    await w.stamp_document_schema_version("doc-1", "acme", registry)
    q = w._neo4j.run.await_args
    assert "tenant: $tenant" in q.args[0] and q.kwargs["tenant"] == "acme"
    assert q.kwargs["label"] == ident.label and q.kwargs["hash"] == ident.content_hash
    # no registered schema -> nothing written
    w._neo4j.run.reset_mock()
    await w.stamp_document_schema_version("doc-1", "acme", None)
    w._neo4j.run.assert_not_called()


def test_schema_cypher_declares_the_registry_constraints():
    from graphrag.graph.schema_statements import load_schema_statements

    text = "\n".join(load_schema_statements())
    for needle in ("(o.tenant, o.dataset_id, o.content_hash) IS UNIQUE",
                   "(d:Dataset) REQUIRE (d.tenant, d.id) IS UNIQUE",
                   "(p:ValidationProfile) REQUIRE (p.tenant, p.content_hash) IS UNIQUE"):
        assert needle in text


def test_registry_statements_never_run_outside_a_tenant_filter():
    for q in (sr.UPSERT_AND_ACTIVATE, sr._LOOKUP):
        assert "$tenant" in q
    # every MATCH/MERGE on a registry node names the tenant
    for line in sr.UPSERT_AND_ACTIVATE.splitlines():
        if line.startswith(("MERGE (d:", "MERGE (p:", "MERGE (o:", "OPTIONAL MATCH (other")):
            assert "tenant" in line or "tenant" in sr.UPSERT_AND_ACTIVATE.split(line)[1].splitlines()[1]
