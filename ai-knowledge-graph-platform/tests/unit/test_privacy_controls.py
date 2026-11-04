"""Erasure propagation, data minimisation before providers, the extraction
record contract and the permission-freshness bound (2026-11-03)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from rdflib import Dataset, Graph, Literal, Namespace

from graphrag.context_graph.knowledge import BusinessContext, Fact
from graphrag.context_graph.knowledge_evaluation import can_read
from graphrag.context_graph.knowledge_release import load_cases, load_pack, publish_pack
from graphrag.context_graph.knowledge_service import KnowledgeEvaluationService
from graphrag.core.llm_cache import LLMCache
from graphrag.core.llm_client import FallbackLLM
from graphrag.core.models import Chunk
from graphrag.core.outbound import (
    OutboundDisabledError, OutboundPolicy, minimise, minimise_fields, recent_outbound_audits,
    set_outbound_policies,
)
from graphrag.domains.energy.governance_store import GovernanceStore
from graphrag.enterprise import access as acl
from graphrag.enterprise.models import AccessContext, ACLState, DocumentAccessPolicy
from graphrag.graph.embedding_cache import get_embedding_cache
from graphrag.graph.erasure_registry import STORES, erasure_table, resolve
from graphrag.graph.named_graphs import erase_resource, graph_iri, load_graph
from graphrag.ingestion.extractor import PROMPT_VERSION, Extractor, diff_extraction_records, extraction_records
from graphrag.observability.correlation import tenant_context
from graphrag.retrieval.query_cache import QueryCache, QueryCacheContext, build_cache_key

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "config" / "knowledge_packs" / "packaging_labeling.yml"
CASES = ROOT / "evals" / "knowledge_eval" / "packaging_labeling_cases.json"
EX = Namespace("https://example.test/")
SECRET = "Contact alice.doe@example.com or +1 415-555-0100, SSN 123-45-6789, from 10.1.2.3."


@pytest.fixture(autouse=True)
def _reset_policies():
    set_outbound_policies(None)
    yield
    set_outbound_policies(None)


# -- erasure: the registry ------------------------------------------------------------


def test_every_derived_store_is_accounted_for():
    names = [s.name for s in STORES]
    assert len(names) == len(set(names))
    for store in STORES:
        if store.coverage in ("erased", "redacted") and store.handler:
            assert callable(resolve(store.handler)), store.name
        if store.coverage in ("immutable", "not_covered", "expires"):
            assert store.rule, f"{store.name} needs its rule or the reason it is not covered"
    assert {"erased", "redacted", "immutable"} <= {s.coverage for s in STORES}
    assert erasure_table().count("\n") == len(STORES) + 2


# -- erasure: propagation through the stores that run offline ----------------------------


def _context() -> QueryCacheContext:
    return QueryCacheContext(corpus_revision=1, requested_mode="local", effective_mode="local", model_route={},
                             prompt_version="p", retrieval_config={}, ontology_version="o")


async def test_answer_cache_erasure_is_per_tenant():
    cache = QueryCache(redis_url=None)
    for tenant in ("tenant-a", "tenant-b"):
        await cache.set("who is alice", tenant, _context(), {"answer": f"alice at {tenant}"},
                        source_query_id="q", source_trace_id="t", entities_used=["Alice"])
    assert await cache.flush_tenant("tenant-a") >= 1
    assert await cache.get("who is alice", "tenant-a", _context()) is None
    assert await cache.get("who is alice", "tenant-b", _context()) is not None
    await cache.invalidate_for("tenant-b", entity_names=["Alice"])
    assert await cache.get("who is alice", "tenant-b", _context()) is None


async def test_gdpr_erasure_clears_cached_embeddings_and_media_for_that_tenant_only():
    from graphrag.graph.gdpr import GDPRService

    embeddings = get_embedding_cache()
    embeddings.set("tenant-a", "Alice", "PERSON", [0.1, 0.2])
    embeddings.set("tenant-b", "Alice", "PERSON", [0.3, 0.4])
    neo4j = SimpleNamespace(run=AsyncMock(return_value=[{"n": 2}]))
    await GDPRService(neo4j)._erase_derived("Alice", "PERSON", "tenant-a")
    assert embeddings.get("tenant-a", "Alice", "PERSON") is None
    assert embeddings.get("tenant-b", "Alice", "PERSON") is not None
    media_query = neo4j.run.await_args
    assert "MediaAttachment" in media_query.args[0] and media_query.kwargs["tenant"] == "tenant-a"


class _Graph:
    """Answers the erasure's queries from canned rows and records every call."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def run(self, query, **kwargs):
        self.calls.append((query, kwargs))
        for marker, rows in self.rows.items():
            if marker in query and "RETURN" in query:
                return rows
        return []

    def writes(self, marker):
        return [kw for q, kw in self.calls if marker in q and "SET" in q]


async def test_erasure_drops_stale_vectors_redacts_summaries_and_deletes_aliases():
    from graphrag.graph.gdpr import GDPRService

    graph = _Graph({
        "c.text AS text": [{"chunk_id": "c1", "text": "Alice met Bob"}],
        "(s:Community {tenant": [{"id": "k1", "summary": "Led by alice."}, {"id": "k2", "summary": "No match"}],
        "(s:CommunitySummarySnapshot {tenant": [{"id": "s1", "summary": "ALICE wrote it"}],
        "AS deleted": [{"deleted": 2}],
    })
    service = GDPRService(graph)
    report = await service._forget_entity("Alice", "PERSON", "tenant-a", "dpo", "r1")
    chunk = graph.writes("MATCH (c:Chunk {id")
    assert chunk == [{"chunk_id": "c1", "tenant": "tenant-a", "text": "[REDACTED] met Bob"}]
    assert "c.embedding = null" in next(q for q, _ in graph.calls if "MATCH (c:Chunk {id" in q)
    summaries = graph.writes("{id: $id, tenant: $tenant}")
    assert [(w["id"], w["summary"]) for w in summaries] == [("k1", "Led by [REDACTED]."), ("s1", "[REDACTED] wrote it")]
    assert all(w["tenant"] == "tenant-a" for w in summaries)
    assert report["summaries_redacted"] == 2 and report["aliases_deleted"] == 2
    order = [q for q, _ in graph.calls]
    alias_at = next(i for i, q in enumerate(order) if "ALIAS_OF" in q)
    entity_at = next(i for i, q in enumerate(order) if "DETACH DELETE e" in q)
    assert alias_at < entity_at, "aliases are found through the entity, so they go first"
    alias_query, alias_kwargs = graph.calls[alias_at]
    assert "other <> e" in alias_query and alias_kwargs["tenant"] == "tenant-a"


async def test_reembed_recomputes_flagged_vectors_through_the_given_embedder():
    from graphrag.graph.gdpr import GDPRService

    graph = _Graph({
        "MATCH (n:Chunk {tenant": [{"id": "c1", "text": "[REDACTED] met Bob"}],
        "MATCH (n:Community {tenant": [{"id": "k1", "text": None}],
    })
    embedder = SimpleNamespace(embed=AsyncMock(return_value=[[0.5, 0.5]]))
    counts = await GDPRService(graph).reembed_redacted("tenant-a", embedder)
    embedder.embed.assert_awaited_once_with(["[REDACTED] met Bob"])
    assert counts == {"Chunk": 1, "Community": 0, "CommunitySummarySnapshot": 0}
    assert graph.writes("n.embedding = $embedding") == [{"id": "c1", "tenant": "tenant-a", "embedding": [0.5, 0.5]}]


def test_the_tenant_turtle_export_is_removed_and_other_tenants_keep_theirs(tmp_path, monkeypatch):
    from graphrag.graph.gdpr import GDPRService

    monkeypatch.setenv("GRAPHRAG_RDF_EXPORT_DIR", str(tmp_path))
    for tenant in ("tenant-a", "tenant-b"):
        (tmp_path / tenant).mkdir()
        (tmp_path / tenant / "graph_export.ttl").write_text("<x> <y> 'Alice' .", encoding="utf-8")
    assert GDPRService._remove_rdf_export("tenant-a") is True
    assert not (tmp_path / "tenant-a" / "graph_export.ttl").exists()
    assert (tmp_path / "tenant-b" / "graph_export.ttl").exists()
    assert GDPRService._remove_rdf_export("tenant-a") is False


async def test_decision_log_cache_and_trust_reports_forget_the_subject(tmp_path):
    store = GovernanceStore(f"sqlite+aiosqlite:///{(tmp_path / 'g.sqlite').as_posix()}", blob_root=tmp_path / "b")
    await store.open()
    try:
        await publish_pack(store, load_pack(PACK), published_by="user:p", cases=load_cases(CASES))
        service = KnowledgeEvaluationService(store)
        case = load_cases(CASES)[0]
        alice, bob = AccessContext.from_claims({"sub": "alice"}), AccessContext.from_claims({"sub": "bob"})
        facts = {k: Fact.model_validate(v) for k, v in case.facts.items()}
        for who in (alice, bob):
            await service.evaluate(tenant=case.tenant, domain="packaging_labeling", access=who,
                                   context=BusinessContext.model_validate(case.context), facts=facts)
        alice_record = next(r for r in await service.decision_log(case.tenant, "packaging_labeling")
                            if r["subject"] == "alice")
        await service.report_distrust(case.tenant, "packaging_labeling", alice_record["record_id"],
                                      reporter=alice, reason="mine")
        result = await service.erase_subject(case.tenant, "alice")
        records = await service.decision_log(case.tenant, "packaging_labeling")
        reports = await service.trust_reports(case.tenant, "packaging_labeling")
    finally:
        await store.close()
    assert result["records_redacted"] == 1 and result["cache_entries_evicted"] >= 1
    serialized = json.dumps(records)
    assert '"alice"' not in serialized and "redacted_at" in serialized
    bob_record = next(r for r in records if r["subject"] == "bob")
    assert bob_record["inputs"]["facts"], "another subject's records are untouched"
    assert reports[0]["reporter"] == "[redacted]" and reports[0]["bundle"]["question"]["facts"] == []


def test_triple_store_erasure_and_correction_stay_inside_one_tenant():
    dataset = Dataset()
    for tenant in ("tenant-a", "tenant-b"):
        graph = Graph()
        graph.add((EX.alice, EX.email, Literal(f"alice@{tenant}")))
        graph.add((EX.acme, EX.employs, EX.alice))
        graph.add((EX.acme, EX.city, Literal("Berlin")))
        load_graph(dataset, graph_iri(tenant, "hr", "v1"), graph)
    assert erase_resource(dataset, "tenant-a", EX.alice) == 2
    a, b = dataset.graph(graph_iri("tenant-a", "hr", "v1")), dataset.graph(graph_iri("tenant-b", "hr", "v1"))
    assert (EX.alice, None, None) not in a and (None, None, EX.alice) not in a
    assert len(b) == 3, "the other tenant keeps its data"
    a.remove((EX.acme, EX.city, None))
    a.add((EX.acme, EX.city, Literal("Munich")))
    assert a.value(EX.acme, EX.city) == Literal("Munich") and b.value(EX.acme, EX.city) == Literal("Berlin")


def test_the_llm_extraction_cache_can_be_purged(tmp_path):
    cache = LLMCache(cache_dir=tmp_path)
    cache.set(model="m", temperature=0.0, json_mode=True, prompt="text about alice", response="{}")
    assert list(tmp_path.glob("*.json"))
    assert cache.purge() == 1 and not list(tmp_path.glob("*.json"))


# -- data minimisation ------------------------------------------------------------------------


class _Capture:
    def __init__(self):
        self.prompts: list[str] = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return "ok"


def _llm(primary):
    return FallbackLLM(primary, "fake", _Capture(), fallback_exceptions=(RuntimeError,))


async def test_direct_identifiers_never_reach_the_provider_and_only_hash_and_size_are_audited():
    primary = _Capture()
    with tenant_context("tenant-a"):
        await _llm(primary).generate(f"Summarise: {SECRET}")
    sent = primary.prompts[0]
    for identifier in ("alice.doe@example.com", "415-555-0100", "123-45-6789", "10.1.2.3"):
        assert identifier not in sent
    assert "[EMAIL_REDACTED]" in sent and "[SSN_REDACTED]" in sent
    audit = recent_outbound_audits()[-1]
    assert audit["sha256"] == hashlib.sha256(sent.encode("utf-8")).hexdigest() and audit["chars"] == len(sent)
    assert audit["tenant"] == "tenant-a" and set(audit) == {"tenant", "purpose", "sha256", "chars", "redactions"}
    assert "alice" not in json.dumps(audit)


async def test_a_disabled_provider_is_never_called_for_that_tenant():
    set_outbound_policies({"": OutboundPolicy(), "clinic": OutboundPolicy(providers_enabled=False)})
    primary = _Capture()
    with tenant_context("clinic"), pytest.raises(OutboundDisabledError):
        await _llm(primary).generate("anything")
    assert primary.prompts == []
    with tenant_context("other"):
        await _llm(primary).generate("allowed")
    assert primary.prompts == ["allowed"]


async def test_embedding_inputs_are_minimised_too():
    from graphrag.core.llm_client import OpenAIEmbedder

    captured: list = []
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder._model = "m"
    embedder._client = SimpleNamespace(embeddings=SimpleNamespace(
        create=lambda model, input: captured.extend(input) or SimpleNamespace(data=[SimpleNamespace(embedding=[0.0])
                                                                                   for _ in input])))
    await embedder.embed([SECRET])
    assert captured and "alice.doe@example.com" not in captured[0]


def test_field_allowlist_and_disabling_redaction_are_per_tenant():
    set_outbound_policies({"": OutboundPolicy(), "lean": OutboundPolicy(allowed_fields=("title",), redact=False)})
    assert minimise_fields({"title": "t", "owner_email": "x@y.z"}, tenant="lean") == {"title": "t"}
    assert minimise_fields({"title": "t", "owner_email": "x@y.z"}, tenant="other") == {"title": "t",
                                                                                     "owner_email": "x@y.z"}
    assert minimise("a@b.co", tenant="lean")[0] == "a@b.co"
    assert minimise("a@b.co", tenant="other")[0] == "[EMAIL_REDACTED]"


# -- extraction record contract ------------------------------------------------------------------


async def _extract(model: str, payload: dict, text: str):
    from graphrag.core import llm_client

    class FakeLLM:
        async def generate(self, prompt, **kw):
            llm_client._report_openai_compatible_usage(SimpleNamespace(model=model, usage=None))
            return json.dumps(payload)

    ex = Extractor.__new__(Extractor)
    ex._model_name = "configured"
    ex._entity_types = ["ORG"]
    chunk = Chunk(id="c1", document_id="d1", text=text, tenant="t", chunk_index=0)
    with patch("graphrag.ingestion.extractor.get_llm", return_value=FakeLLM()), \
         patch("graphrag.ingestion.extractor.get_settings") as settings:
        settings.return_value.llm_cache_enabled = False
        return await ex.extract(chunk)


PAYLOAD = {"entities": [{"name": "Acme", "type": "ORG"}, {"name": "Bolt", "type": "ORG"},
                        {"name": "Ghost Corp", "type": "ORG"}],
           "relations": [{"source": "Acme", "target": "Bolt", "relation": "OWNS", "confidence": 0.95},
                         {"source": "Bolt", "target": "Acme", "relation": "SUPPLIES", "confidence": 0.3},
                         {"source": "Acme", "target": "Ghost Corp", "relation": "OWNS", "confidence": 0.9}]}


async def test_every_extracted_assertion_carries_the_full_record():
    entities, relations = await _extract("model-a", PAYLOAD, "Acme owns Bolt.")
    records = extraction_records(entities, relations)
    for row in records:
        assert row["source_chunk_ids"] == ["c1"] and row["model"] == "model-a"
        assert row["prompt_version"] == PROMPT_VERSION and 0.0 <= row["confidence"] <= 1.0
        assert row["review_state"] in ("accepted", "uncertain", "unsupported")
    states = {row["key"]: row["review_state"] for row in records}
    assert states["ORG:Ghost Corp"] == "unsupported", "not in the source text"
    assert states["Bolt-SUPPLIES->Acme"] == "uncertain", "below the review confidence"
    assert states["Acme-OWNS->Bolt"] == "accepted"
    assert states["Acme-OWNS->Ghost Corp"] == "unsupported"
    assert PROMPT_VERSION.startswith("v1-") and len(PROMPT_VERSION) == 11


async def test_a_replay_with_another_model_produces_a_diffable_record():
    first = extraction_records(*await _extract("model-a", PAYLOAD, "Acme owns Bolt."))
    second = extraction_records(*await _extract("model-b", PAYLOAD, "Acme owns Bolt."))
    diff = diff_extraction_records(first, second)
    assert diff["added"] == [] and diff["removed"] == []
    assert diff["changed"] and all(change["fields"] == ["model"] for change in diff["changed"])


async def test_unaccepted_assertions_go_to_review_not_the_graph():
    from graphrag.ingestion.graph_writer import GraphWriter

    entities, relations = await _extract("model-a", PAYLOAD, "Acme owns Bolt.")
    writer = GraphWriter.__new__(GraphWriter)
    chunk = Chunk(id="c1", document_id="d1", text="Acme owns Bolt.", tenant="t", chunk_index=0)
    kept = writer._hold_for_review(entities, chunk)
    assert {e.name for e in kept} == {"Acme", "Bolt"}
    held = writer.drain_rejections()
    assert [(r.record_kind, r.rule_ids) for r in held] == [("entity", ["ENT-REVIEW-001"])]
    assert held[0].payload["review_state"] == "unsupported"


# -- permission freshness ---------------------------------------------------------------------------


NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)


def _restricted(observed_at):
    return DocumentAccessPolicy(mode="restricted", state=ACLState.KNOWN, allow_principals=["user:alice"],
                                observed_at=observed_at)


def test_a_restricted_acl_older_than_the_bound_is_denied():
    fresh = _restricted(NOW - acl.ACL_MAX_AGE + timedelta(minutes=1))
    stale = _restricted(NOW - acl.ACL_MAX_AGE - timedelta(minutes=1))
    assert acl.acl_is_fresh(fresh, NOW) and not acl.acl_is_fresh(stale, NOW)
    assert acl.acl_is_fresh(_restricted(None), NOW), "legacy policies without an observation time keep working"
    alice = AccessContext.from_claims({"sub": "alice"})
    with patch("graphrag.enterprise.access.datetime") as clock:
        clock.now.return_value = NOW
        assert can_read(fresh, alice) and not can_read(stale, alice)


def test_the_cypher_predicate_and_params_carry_the_bound():
    params = acl.access_params(AccessContext.from_claims({"sub": "alice"}), enabled=True)
    assert "acl_fresh_after" in params
    predicate = acl.document_access_predicate("d")
    assert "d.acl_observed_at IS NULL OR d.acl_observed_at >= $acl_fresh_after" in predicate
    stored = acl.normalise_policy(_restricted(NOW))
    assert stored["acl_observed_at"] >= params["acl_fresh_after"][:4]


def test_cached_answers_stop_being_served_when_the_acl_window_turns_over():
    base = _context()
    now_key = build_cache_key("q", "t", QueryCacheContext(**{**base.canonical_content(),
                                                             "acl_epoch": acl.acl_epoch(NOW)}))
    later_key = build_cache_key("q", "t", QueryCacheContext(**{**base.canonical_content(),
                                                               "acl_epoch": acl.acl_epoch(NOW + acl.ACL_MAX_AGE)}))
    assert now_key != later_key


class _FakeSource:
    def __init__(self):
        self.allowed = ["alice-id"]
        self.reachable = True

    async def content(self, item_id):
        return b"quarterly figures"

    async def permissions(self, item_id):
        if not self.reachable:
            raise httpx.ConnectError("permission service down")
        return [{"grantedToV2": {"user": {"id": user}}} for user in self.allowed]


def _connector(source):
    from graphrag.enterprise.sharepoint import SharePointSourceConfig, SharePointSyncConnector

    config = SharePointSourceConfig(source_id="sp", tenant_id="tid", client_id="cid", client_secret_env="UNUSED",
                                    site_id="site", drive_id="drive", tenant="t")
    return SharePointSyncConnector(config, graph_client=source, sync_service=SimpleNamespace())


async def test_a_revoked_permission_takes_effect_on_resync_and_within_the_bound_without_one():
    source = _FakeSource()
    connector = _connector(source)
    item = {"id": "doc-1", "name": "q3.txt", "webUrl": "https://sp/q3.txt", "eTag": "1"}
    alice = AccessContext.from_claims({"sub": "alice-id"})
    first = (await connector._to_change(item)).access_policy
    assert first.observed_at is not None and can_read(first, alice)

    source.allowed = ["bob-id"]                       # revoked at the source
    resynced = (await connector._to_change(item)).access_policy
    assert not can_read(resynced, alice), "a re-sync denies immediately"

    with patch("graphrag.enterprise.access.datetime") as clock:  # never re-synced: denied after the bound
        clock.now.return_value = first.observed_at + acl.ACL_MAX_AGE + timedelta(seconds=1)
        assert not can_read(first, alice)


async def test_an_unreachable_permission_source_is_denied_and_audited():
    source = _FakeSource()
    source.reachable = False
    connector = _connector(source)
    change = await connector._to_change({"id": "doc-2", "name": "x.txt", "webUrl": "https://sp/x.txt", "eTag": "1"})
    assert change.access_policy.state == ACLState.UNKNOWN
    assert not can_read(change.access_policy, AccessContext.from_claims({"sub": "alice-id"}))
    assert connector.acl_audit == [{"item_id": "doc-2", "event": "acl_unresolved", "outcome": "denied",
                                    "reason": "ConnectError", "observed_at": connector.acl_audit[0]["observed_at"]}]
