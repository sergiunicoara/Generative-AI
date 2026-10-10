"""Phase 3: dependency tracking and targeted invalidation.

Graph semantics of the Cypher are proven in tests/e2e/test_live_invalidation.py
(CI). These tests drive the real service/index/state-store/worker code against
a fake that answers each query constant from an in-memory model of the graph,
keyed by tenant, so dependency closure, idempotency, fallbacks and tenant
isolation are exercised deterministically.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph.invalidation import (
    ALLOWED_TRANSITIONS,
    ArtifactKind,
    ArtifactState,
    EntityRef,
    EventKind,
    InvalidationEvent,
    InvalidTransition,
    RelationRef,
    check_transition,
)
from graphrag.graph.invalidation import dependency_index as di
from graphrag.graph.invalidation import recompute as rc
from graphrag.graph.invalidation import state_store as ss
from graphrag.graph.invalidation.dependency_index import DependencyIndex
from graphrag.graph.invalidation.recompute import RecomputeWorker, parse_relation_key
from graphrag.graph.invalidation.service import InvalidationService, emit
from graphrag.graph.validation.graph_checks import assert_read_only

T, OTHER = "acme", "victim"


class FakeGraph:
    """Minimal per-tenant model answering the invalidation queries."""

    def __init__(self):
        self.t: dict[str, dict] = {}
        self.writes: list[tuple[str, dict]] = []
        self.reads: list[tuple[str, dict]] = []
        self.events: dict[tuple[str, str], dict] = {}
        self.states: dict[tuple[str, str, str], dict] = {}
        self.advance_corpus_revision = AsyncMock(return_value=2)

    def tenant(self, name: str) -> dict:
        return self.t.setdefault(name, {
            "doc_chunks": {}, "doc_relations": {}, "entity": {}, "inferred": [],
            "snapshots": [], "decisions": [],
        })

    async def run_read(self, q, **p):
        self.reads.append((q, p))
        g = self.tenant(p["tenant"])
        if q is di._DOC_CHUNKS:
            return [{"id": c} for d in p["ids"] for c in g["doc_chunks"].get(d, [])]
        if q is di._DOC_RELATIONS:
            return [r for d in p["ids"] for r in g["doc_relations"].get(d, [])]
        if q is di._DEPENDENT_INFERRED:
            out = []
            for e in g["inferred"]:
                prem = e.get("premises", [])
                if any(x in p["keys"] for x in prem) or any(
                        x.startswith(tok + "|") or x.endswith("|" + tok) for x in prem for tok in p["tokens"]):
                    out.append({k: e[k] for k in ("src_name", "src_type", "relation", "tgt_name",
                                                  "tgt_type", "rule")})
            return out
        if q is di._ENTITY_CHUNKS_AND_IDS:
            return [{"entity_id": g["entity"][(e["name"], e["type"])]["id"],
                     "chunk_ids": g["entity"][(e["name"], e["type"])]["chunks"]}
                    for e in p["entities"] if (e["name"], e["type"]) in g["entity"]]
        if q is di._SNAPSHOTS:
            return [{"id": s["id"]} for s in g["snapshots"]
                    if set(s["chunks"]) & set(p["chunks"]) or set(s["docs"]) & set(p["docs"])
                    or set(s["entities"]) & set(p["entity_ids"])]
        if q is di._DECISIONS:
            return [{"id": d["id"]} for d in g["decisions"]
                    if set(d["chunks"]) & set(p["chunks"]) or set(d["docs"]) & set(p["docs"])]
        if q is di._DECISIONS_UNDER_OTHER_SCHEMA:
            return [{"id": d["id"]} for d in g["decisions"] if d.get("schema") != p["schema_version"]]
        return []

    async def run(self, q, **p):
        self.writes.append((q, p))
        if q is ss._UPSERT_EVENT:
            key = (p["tenant"], p["id"])
            if key in self.events:
                e = self.events[key]
                return [{"status": e["status"], "summary_json": e.get("summary"), "created": False}]
            self.events[key] = {"status": "PROCESSING"}
            return [{"status": "PROCESSING", "summary_json": None, "created": True}]
        if q is ss._FINISH_EVENT:
            self.events[(p["tenant"], p["id"])].update(status="PROCESSED", summary=p["summary"])
            return []
        if q is ss._MARK:
            n = 0
            for aid in p["ids"]:
                st = self.states.setdefault((p["tenant"], p["kind"], aid),
                                            {"state": "VALID", "version": 0, "events": set()})
                if p["event_id"] in st["events"]:
                    continue
                st["events"].add(p["event_id"])
                st.update(state="NEEDS_REVIEW", version=st["version"] + 1)
                n += 1
            return [{"marked": n}]
        if q is ss._CLAIM:
            out = []
            for (tn, kind, aid), st in self.states.items():
                if tn == p["tenant"] and st["state"] == "NEEDS_REVIEW" and not st.get("human"):
                    st.update(state="RECOMPUTING", version=st["version"] + 1)
                    out.append({"kind": kind, "artifact_id": aid, "version": st["version"], "event_id": None})
            return out[: p["limit"]]
        if q is ss._COMPLETE:
            st = self.states.get((p["tenant"], p["kind"], p["artifact_id"]))
            if st and st["state"] == "RECOMPUTING" and st["version"] == p["version"]:
                st.update(state=p["state"], version=st["version"] + 1, human=p["human"], detail=p["detail"])
                return [{"n": 1}]
            return [{"n": 0}]
        return []

    def state(self, tenant, kind, aid):
        return self.states.get((tenant, kind.value if hasattr(kind, "value") else kind, aid), {}).get("state")


def ownership_world(g: FakeGraph, tenant: str = T) -> None:
    """Acme OWNS Bolt (doc d1). Inferred: Acme CONTROLS_INDIRECTLY Nut via Bolt OWNS Nut.
    Second-order inferred edge depends on the first (multi-hop)."""
    w = g.tenant(tenant)
    w["doc_chunks"] = {"d1": ["c1", "c2"], "d2": ["c3"]}
    owns = {"src_name": "Acme", "src_type": "ORG", "relation": "OWNS", "tgt_name": "Bolt", "tgt_type": "ORG"}
    w["doc_relations"] = {"d1": [owns]}
    w["entity"] = {("Acme", "ORG"): {"id": "e-acme", "chunks": ["c1"]},
                   ("Bolt", "ORG"): {"id": "e-bolt", "chunks": ["c2"]},
                   ("Zed", "ORG"): {"id": "e-zed", "chunks": ["c9"]}}
    w["inferred"] = [
        {"src_name": "Acme", "src_type": "ORG", "relation": "OWNS", "tgt_name": "Nut", "tgt_type": "ORG",
         "rule": "owns_transitivity", "premises": ["ORG:Acme|OWNS|ORG:Bolt", "ORG:Bolt|OWNS|ORG:Nut"]},
        {"src_name": "Acme", "src_type": "ORG", "relation": "CONTROLS", "tgt_name": "Nut", "tgt_type": "ORG",
         "rule": "owns_controls", "premises": ["ORG:Acme|OWNS|ORG:Nut"]},
        {"src_name": "Zed", "src_type": "ORG", "relation": "OWNS", "tgt_name": "Yak", "tgt_type": "ORG",
         "rule": "unrelated", "premises": ["ORG:Zed|OWNS|ORG:Xen", "ORG:Xen|OWNS|ORG:Yak"]},
    ]
    w["snapshots"] = [{"id": "s-acme", "chunks": ["c1"], "docs": [], "entities": ["e-acme"]},
                      {"id": "s-zed", "chunks": ["c9"], "docs": [], "entities": ["e-zed"]}]
    w["decisions"] = [{"id": "dec-acme", "chunks": ["c1"], "docs": ["d1"], "schema": "o@1#aaa"},
                      {"id": "dec-zed", "chunks": ["c9"], "docs": ["d9"], "schema": "o@1#aaa"}]


OWNS = RelationRef(src_name="Acme", src_type="ORG", relation="OWNS", tgt_name="Bolt", tgt_type="ORG")


def owns_event(**kw) -> InvalidationEvent:
    return InvalidationEvent(tenant=kw.pop("tenant", T), kind=EventKind.RELATION_CHANGED,
                             relations=[OWNS], cause=kw.pop("cause", "audit-1"), **kw)


class SharedCache:
    shared = True

    def __init__(self):
        self.invalidate_for = AsyncMock(return_value=3)


def service(g, cache=None, *, inline=False):
    svc = InvalidationService(g, cache_getter=AsyncMock(return_value=cache))
    svc._recompute_inline = inline
    return svc


# ── state machine ─────────────────────────────────────────────────────────────

def test_state_machine_matches_the_spec():
    S = ArtifactState
    check_transition(S.VALID, S.NEEDS_REVIEW)
    check_transition(S.NEEDS_REVIEW, S.RECOMPUTING)
    check_transition(S.RECOMPUTING, S.VALID)
    for terminal in (S.INSUFFICIENT_EVIDENCE, S.VALIDATION_FAILED, S.RECOMPUTE_FAILED):
        check_transition(S.RECOMPUTING, terminal)
        check_transition(terminal, S.NEEDS_REVIEW)
        with pytest.raises(InvalidTransition):
            check_transition(terminal, S.VALID)
    with pytest.raises(InvalidTransition):
        check_transition(S.VALID, S.RECOMPUTING)  # must be invalidated first
    assert set(ALLOWED_TRANSITIONS) == set(S)


def test_event_id_is_deterministic_and_order_independent():
    a = InvalidationEvent(tenant=T, kind=EventKind.SUPERSEDED, document_ids=["b", "a"], cause="x")
    b = InvalidationEvent(tenant=T, kind=EventKind.SUPERSEDED, document_ids=["a", "b"], cause="x")
    assert a.id == b.id
    assert a.id != InvalidationEvent(tenant=OTHER, kind=EventKind.SUPERSEDED, document_ids=["a", "b"], cause="x").id
    assert a.id != InvalidationEvent(tenant=T, kind=EventKind.SUPERSEDED, document_ids=["a", "b"], cause="y").id


def test_relation_key_round_trip():
    assert parse_relation_key(OWNS.key) == OWNS.model_dump()


# ── dependency closure ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_relation_change_reaches_multi_hop_inferred_edges_and_their_dependents_only():
    g = FakeGraph()
    ownership_world(g)
    c = await DependencyIndex(g).resolve(owns_event())
    assert set(c.inferred_edges) == {"ORG:Acme|OWNS|ORG:Nut", "ORG:Acme|CONTROLS|ORG:Nut"}  # 2 hops
    assert c.depth_reached >= 2
    assert "ORG:Zed|OWNS|ORG:Yak" not in c.inferred_edges            # unrelated derivation untouched
    assert c.snapshot_ids == {"s-acme"} and c.decision_ids == {"dec-acme"}
    assert {"c1", "c2"} <= c.chunk_ids and "c9" not in c.chunk_ids
    assert not c.truncated


@pytest.mark.asyncio
async def test_superseded_document_reaches_its_chunks_relations_and_derivations():
    g = FakeGraph()
    ownership_world(g)
    c = await DependencyIndex(g).resolve(
        InvalidationEvent(tenant=T, kind=EventKind.SUPERSEDED, document_ids=["d1"], cause="new-doc"))
    assert {"c1", "c2"} <= c.chunk_ids
    assert OWNS.key in c.relation_keys                      # edge sourced by the superseded doc
    assert "ORG:Acme|OWNS|ORG:Nut" in c.inferred_edges      # derived from that edge
    assert c.decision_ids == {"dec-acme"}


@pytest.mark.asyncio
async def test_entity_correction_reaches_premises_through_that_entity():
    g = FakeGraph()
    ownership_world(g)
    c = await DependencyIndex(g).resolve(InvalidationEvent(
        tenant=T, kind=EventKind.FACT_CORRECTED, entities=[EntityRef(name="Bolt", type="ORG")], cause="q"))
    assert "ORG:Acme|OWNS|ORG:Nut" in c.inferred_edges and "c2" in c.chunk_ids


@pytest.mark.asyncio
async def test_schema_change_reaches_decisions_recorded_under_another_version():
    g = FakeGraph()
    ownership_world(g)
    g.tenant(T)["decisions"].append({"id": "dec-new", "chunks": [], "docs": [], "schema": "o@2#bbb"})
    c = await DependencyIndex(g).resolve(InvalidationEvent(
        tenant=T, kind=EventKind.SCHEMA_CHANGED, schema_version="o@2#bbb", cause="h"))
    assert c.decision_ids == {"dec-acme", "dec-zed"}


@pytest.mark.asyncio
async def test_closure_is_tenant_scoped_and_read_only():
    g = FakeGraph()
    ownership_world(g, T)
    ownership_world(g, OTHER)
    c = await DependencyIndex(g).resolve(owns_event())
    assert c.decision_ids == {"dec-acme"}  # same ids exist in OTHER, never queried
    assert g.writes == []
    assert all(p["tenant"] == T for _, p in g.reads)
    for q, _ in g.reads:
        assert_read_only(q)


@pytest.mark.asyncio
async def test_cap_or_depth_overflow_marks_the_closure_truncated():
    g = FakeGraph()
    ownership_world(g)
    assert (await DependencyIndex(g, cap=1).resolve(owns_event())).truncated
    assert (await DependencyIndex(g, max_depth=1).resolve(owns_event())).truncated


def test_every_index_query_is_read_only_and_tenant_filtered():
    for q in (di._DOC_CHUNKS, di._DOC_RELATIONS, di._ENTITY_CHUNKS_AND_IDS, di._DEPENDENT_INFERRED,
              di._SNAPSHOTS, di._DECISIONS, di._DECISIONS_UNDER_OTHER_SCHEMA,
              rc._EDGE, rc._PREMISES_OK, rc._SNAPSHOT_EVIDENCE, rc._DECISION_EVIDENCE):
        assert_read_only(q)
        assert "$tenant" in q
    for q in (ss._UPSERT_EVENT, ss._FINISH_EVENT, ss._MARK, ss._FLAG_INFERRED, ss._FLAG_SNAPSHOTS,
              ss._CLAIM, ss._COMPLETE, rc._RETRACT_INFERRED, rc._REVALIDATE_INFERRED, rc._CLOSE_SNAPSHOT):
        assert "$tenant" in q


# ── service: targeted marking, eviction, fallbacks, idempotency ──────────────

@pytest.mark.asyncio
async def test_targeted_event_marks_only_dependents_and_evicts_only_their_answers():
    g = FakeGraph()
    ownership_world(g)
    cache = SharedCache()
    out = await service(g, cache).handle(owns_event())
    assert out["fallback"] is None
    assert g.state(T, ArtifactKind.DECISION, "dec-acme") == "NEEDS_REVIEW"
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s-acme") == "NEEDS_REVIEW"
    assert g.state(T, ArtifactKind.INFERRED_EDGE, "ORG:Acme|CONTROLS|ORG:Nut") == "NEEDS_REVIEW"
    # unrelated artifacts are not touched
    assert g.state(T, ArtifactKind.DECISION, "dec-zed") is None
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s-zed") is None
    assert g.state(T, ArtifactKind.INFERRED_EDGE, "ORG:Zed|OWNS|ORG:Yak") is None
    kw = cache.invalidate_for.await_args.kwargs
    assert "Acme" in kw["entity_names"] and "Zed" not in kw["entity_names"]
    assert "c1" in kw["chunk_ids"] and "c9" not in kw["chunk_ids"]
    assert cache.invalidate_for.await_args.args == (T,)
    g.advance_corpus_revision.assert_not_called()  # no tenant-wide bump


@pytest.mark.asyncio
async def test_repeated_event_is_a_no_op():
    g = FakeGraph()
    ownership_world(g)
    cache = SharedCache()
    first = await service(g, cache).handle(owns_event())
    versions = {k: v["version"] for k, v in g.states.items()}
    second = await service(g, cache).handle(owns_event())
    assert second["duplicate"] is True and second["event_id"] == first["event_id"]
    assert {k: v["version"] for k, v in g.states.items()} == versions
    assert cache.invalidate_for.await_count == 1


@pytest.mark.asyncio
async def test_a_different_cause_is_a_new_event_but_does_not_double_mark():
    g = FakeGraph()
    ownership_world(g)
    await service(g, SharedCache()).handle(owns_event(cause="a"))
    v1 = g.states[(T, "decision", "dec-acme")]["version"]
    await service(g, SharedCache()).handle(owns_event(cause="b"))
    assert g.states[(T, "decision", "dec-acme")]["version"] == v1 + 1  # second event recorded once
    assert len(g.states[(T, "decision", "dec-acme")]["events"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("cache,expected", [
    ("unshared", "unshared_cache"),
    ("failing", "eviction_failed"),
    ("broken_getter", "cache_unavailable"),
])
async def test_unprovable_eviction_falls_back_to_revision_bump(cache, expected):
    g = FakeGraph()
    ownership_world(g)
    if cache == "unshared":
        c = SharedCache()
        c.shared = False
        svc = service(g, c)
    elif cache == "failing":
        c = SharedCache()
        c.invalidate_for = AsyncMock(side_effect=RuntimeError("redis down"))
        svc = service(g, c)
    else:
        svc = InvalidationService(g, cache_getter=AsyncMock(side_effect=RuntimeError("x")))
        svc._recompute_inline = False
    out = await svc.handle(owns_event())
    assert out["fallback"] == expected
    g.advance_corpus_revision.assert_awaited_once()
    # dependents are still marked for review
    assert g.state(T, ArtifactKind.DECISION, "dec-acme") == "NEEDS_REVIEW"


@pytest.mark.asyncio
async def test_truncated_closure_falls_back_to_revision_bump():
    g = FakeGraph()
    ownership_world(g)
    svc = service(g, SharedCache())
    svc._index.cap = 1
    out = await svc.handle(owns_event())
    assert out["fallback"] == "dependency_limit" and out["truncated"] is True
    g.advance_corpus_revision.assert_awaited_once()


@pytest.mark.asyncio
async def test_disabled_answer_cache_needs_no_fallback():
    g = FakeGraph()
    ownership_world(g)
    out = await service(g, None).handle(owns_event())
    assert out["fallback"] is None
    g.advance_corpus_revision.assert_not_called()


@pytest.mark.asyncio
async def test_tenant_isolation_during_invalidation():
    g = FakeGraph()
    ownership_world(g, T)
    ownership_world(g, OTHER)
    cache = SharedCache()
    await service(g, cache).handle(owns_event(tenant=T))
    assert not any(k[0] == OTHER for k in g.states)
    assert all(p.get("tenant") in (T, None) for _, p in g.writes + g.reads)
    assert cache.invalidate_for.await_args.args == (T,)


@pytest.mark.asyncio
async def test_emit_never_raises_and_falls_back():
    g = FakeGraph()
    with patch("graphrag.graph.invalidation.service.InvalidationService.handle",
               AsyncMock(side_effect=RuntimeError("boom"))):
        out = await emit(owns_event(), g)
    assert out["fallback"] == "error"
    g.advance_corpus_revision.assert_awaited_once()


@pytest.mark.asyncio
async def test_emit_outside_a_mutation_bumps_for_additive_events_only():
    g = FakeGraph()
    ownership_world(g)
    with patch("graphrag.graph.invalidation.service.InvalidationService.handle",
               AsyncMock(return_value={"fallback": None})):
        await emit(owns_event(additive=True), g, inside_mutation=False)
        assert g.advance_corpus_revision.await_count == 1
        await emit(owns_event(cause="z"), g, inside_mutation=False)
        assert g.advance_corpus_revision.await_count == 1
        await emit(owns_event(additive=True, cause="y"), g, inside_mutation=True)
        assert g.advance_corpus_revision.await_count == 1


# ── recomputation ─────────────────────────────────────────────────────────────

class RecomputeGraph(FakeGraph):
    def __init__(self, *, edge=None, premises_ok=None, snapshot=None, decision=None):
        super().__init__()
        self.edge, self.premises_ok, self.snapshot, self.decision = edge, premises_ok, snapshot, decision

    async def run_read(self, q, **p):
        self.reads.append((q, p))
        if q is rc._EDGE:
            return [self.edge] if self.edge else []
        if q is rc._PREMISES_OK:
            return [{"premise": x, "ok": self.premises_ok.get(x, False)} for x in p["premises"]]
        if q is rc._SNAPSHOT_EVIDENCE:
            return [self.snapshot] if self.snapshot else []
        if q is rc._DECISION_EVIDENCE:
            return [self.decision] if self.decision else []
        return []


def queue(g: FakeGraph, kind: ArtifactKind, aid: str) -> None:
    g.states[(T, kind.value, aid)] = {"state": "NEEDS_REVIEW", "version": 1, "events": {"e"}}


@pytest.mark.asyncio
async def test_inferred_edge_with_valid_premises_returns_to_valid():
    key = "ORG:Acme|OWNS|ORG:Nut"
    g = RecomputeGraph(edge={"source_type": "inferred", "rule": "r", "premises": ["p1", "p2"], "state": "INFERRED"},
                       premises_ok={"p1": True, "p2": True})
    queue(g, ArtifactKind.INFERRED_EDGE, key)
    out = await RecomputeWorker(g, engine=MagicMock()).run_once(T)
    assert out["outcomes"] == {"VALID": 1}
    assert g.state(T, ArtifactKind.INFERRED_EDGE, key) == "VALID"
    assert not any(q is rc._RETRACT_INFERRED for q, _ in g.writes)


@pytest.mark.asyncio
async def test_inferred_edge_is_rederived_when_an_alternative_derivation_exists():
    key = "ORG:Acme|OWNS|ORG:Nut"
    g = RecomputeGraph(edge={"source_type": "inferred", "rule": "r", "premises": ["p1"], "state": "INFERRED"},
                       premises_ok={"p1": False})
    engine = MagicMock()
    engine.derivation_for = AsyncMock(return_value={"premises": ["p9"], "confidence": 0.7})
    queue(g, ArtifactKind.INFERRED_EDGE, key)
    await RecomputeWorker(g, engine=engine).run_once(T)
    assert g.state(T, ArtifactKind.INFERRED_EDGE, key) == "VALID"
    write = next(p for q, p in g.writes if q is rc._REVALIDATE_INFERRED)
    assert write["premises"] == ["p9"] and write["tenant"] == T
    assert engine.derivation_for.await_args.kwargs["tenant"] == T


@pytest.mark.asyncio
async def test_inferred_edge_without_valid_derivation_is_retracted_not_deleted():
    key = "ORG:Acme|OWNS|ORG:Nut"
    g = RecomputeGraph(edge={"source_type": "inferred", "rule": "r", "premises": ["p1"], "state": "INFERRED"},
                       premises_ok={"p1": False})
    engine = MagicMock()
    engine.derivation_for = AsyncMock(return_value=None)
    queue(g, ArtifactKind.INFERRED_EDGE, key)
    await RecomputeWorker(g, engine=engine).run_once(T)
    assert g.state(T, ArtifactKind.INFERRED_EDGE, key) == "INSUFFICIENT_EVIDENCE"
    assert any(q is rc._RETRACT_INFERRED for q, _ in g.writes)
    assert not any("DELETE" in q for q, _ in g.writes)


@pytest.mark.asyncio
async def test_snapshot_with_lost_evidence_is_closed_and_history_kept():
    g = RecomputeGraph(snapshot={"current": True, "missing_chunks": 1, "bad_docs": 0, "bad_entities": 0})
    queue(g, ArtifactKind.COMMUNITY_SNAPSHOT, "s1")
    await RecomputeWorker(g).run_once(T)
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s1") == "INSUFFICIENT_EVIDENCE"
    closed = next(q for q, _ in g.writes if q is rc._CLOSE_SNAPSHOT)
    assert "transaction_to" in closed and "DELETE" not in closed


@pytest.mark.asyncio
async def test_snapshot_with_intact_evidence_returns_to_valid():
    g = RecomputeGraph(snapshot={"current": True, "missing_chunks": 0, "bad_docs": 0, "bad_entities": 0})
    queue(g, ArtifactKind.COMMUNITY_SNAPSHOT, "s1")
    await RecomputeWorker(g).run_once(T)
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s1") == "VALID"


@pytest.mark.asyncio
async def test_decision_without_surviving_evidence_is_insufficient_evidence():
    g = RecomputeGraph(decision={"total": 3, "surviving": 0})
    queue(g, ArtifactKind.DECISION, "dec")
    await RecomputeWorker(g).run_once(T)
    assert g.state(T, ArtifactKind.DECISION, "dec") == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_decision_with_partial_evidence_goes_to_human_review_and_is_not_reclaimed():
    g = RecomputeGraph(decision={"total": 3, "surviving": 2})
    queue(g, ArtifactKind.DECISION, "dec")
    w = RecomputeWorker(g)
    await w.run_once(T)
    st = g.states[(T, "decision", "dec")]
    assert st["state"] == "NEEDS_REVIEW" and st["human"] is True
    assert (await w.run_once(T))["claimed"] == 0


@pytest.mark.asyncio
async def test_recompute_error_becomes_recompute_failed():
    g = RecomputeGraph()
    g.run_read = AsyncMock(side_effect=RuntimeError("db"))
    queue(g, ArtifactKind.COMMUNITY_SNAPSHOT, "s1")
    await RecomputeWorker(g).run_once(T)
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s1") == "RECOMPUTE_FAILED"


@pytest.mark.asyncio
async def test_completion_after_a_newer_invalidation_does_not_overwrite_it():
    g = RecomputeGraph(snapshot={"current": True, "missing_chunks": 0, "bad_docs": 0, "bad_entities": 0})
    queue(g, ArtifactKind.COMMUNITY_SNAPSHOT, "s1")
    store = ss.StateStore(g)
    [claimed] = await store.claim(T)
    g.states[(T, "community_snapshot", "s1")].update(state="NEEDS_REVIEW", version=99)  # re-invalidated
    assert await store.complete(T, claimed, ArtifactState.VALID) is False
    assert g.state(T, ArtifactKind.COMMUNITY_SNAPSHOT, "s1") == "NEEDS_REVIEW"


@pytest.mark.asyncio
async def test_complete_rejects_non_completion_states():
    with pytest.raises(ValueError):
        await ss.StateStore(FakeGraph()).complete(T, {"kind": "decision", "artifact_id": "x", "version": 1},
                                                  ArtifactState.RECOMPUTING)


# ── answer cache: chunk-level provenance and targeted eviction ───────────────

@pytest.mark.asyncio
async def test_answer_cache_evicts_only_answers_citing_the_changed_chunks():
    from graphrag.retrieval.query_cache import QueryCache, QueryCacheContext

    cache = QueryCache(ttl=60, max_memory_entries=10)
    ctx = QueryCacheContext(corpus_revision=1, requested_mode="h", effective_mode="h", model_route={},
                            prompt_version="p", retrieval_config={}, ontology_version="o",
                            valid_at=None, transaction_at=None)
    await cache.set("q1", T, ctx, {"answer": "1"}, source_query_id="a", source_trace_id="t",
                    entities_used=["Acme"], chunks_used=["c1"])
    await cache.set("q2", T, ctx, {"answer": "2"}, source_query_id="b", source_trace_id="t",
                    entities_used=["Zed"], chunks_used=["c9"])
    await cache.set("q1", OTHER, ctx, {"answer": "3"}, source_query_id="c", source_trace_id="t",
                    entities_used=["Acme"], chunks_used=["c1"])
    assert await cache.invalidate_for(T, chunk_ids=["c1"]) == 1
    assert await cache.get("q1", T, ctx) is None
    assert await cache.get("q2", T, ctx) is not None          # unrelated answer still cached
    assert await cache.get("q1", OTHER, ctx) is not None      # other tenant untouched
    assert cache.shared is False
    # an entity name equal to a chunk id cannot collide with the chunk namespace
    assert await cache.invalidate_for(T, entity_names=["c9"]) == 0


@pytest.mark.asyncio
async def test_corpus_mutation_can_complete_without_advancing_the_revision():
    from graphrag.graph.corpus_revision import CorpusMutation

    neo = AsyncMock()
    neo.complete_corpus_update = AsyncMock(return_value=4)
    async with CorpusMutation(neo, T, "x", advance_revision=False):
        pass
    assert neo.complete_corpus_update.await_args.kwargs["advance_revision"] is False
    async with CorpusMutation(neo, T, "y"):
        pass
    assert "advance_revision" not in neo.complete_corpus_update.await_args.kwargs


def test_complete_corpus_update_advances_invalidation_seq_always_and_revision_conditionally():
    import inspect
    from graphrag.graph.neo4j_client import Neo4jClient

    src = inspect.getsource(Neo4jClient.complete_corpus_update)
    assert "CASE WHEN $advance THEN 1 ELSE 0 END" in src
    assert "s.invalidation_seq = coalesce(s.invalidation_seq, 0) + 1" in src
    assert "invalidation_seq" in inspect.getsource(Neo4jClient.get_corpus_state)


# ── inference engine records premises and ignores invalid ones ───────────────

@pytest.mark.asyncio
async def test_inferred_edges_store_premises_and_rule_version():
    from graphrag.graph.inference_engine import DEFAULT_RULES, ForwardChainingEngine, rule_version

    neo = AsyncMock()
    neo.run = AsyncMock(side_effect=lambda q, **p: [
        {"src": "A", "src_type": "ORG", "tgt": "B", "tgt_type": "ORG", "conf": 0.9,
         "premises": ["ORG:A|FOUNDED|ORG:B"]}] if "LIMIT 500" in q else [])
    engine = ForwardChainingEngine(neo, rules=[r for r in DEFAULT_RULES if r.name == "founded_inverse"])
    await engine.run(tenant=T, max_iterations=1)
    write = next(c for c in neo.run.await_args_list if "premise_keys" in c.args[0])
    assert write.kwargs["premises"] == ["ORG:A|FOUNDED|ORG:B"]
    assert write.kwargs["rule_version"] == rule_version(engine.rule_named("founded_inverse"))


def test_rule_version_changes_with_rule_logic():
    from dataclasses import replace

    from graphrag.graph.inference_engine import DEFAULT_RULES, rule_version
    r = DEFAULT_RULES[0]
    assert rule_version(r) == rule_version(replace(r))
    assert rule_version(r) != rule_version(replace(r, confidence_decay=0.5))


def test_rule_queries_exclude_retracted_expired_and_quarantined_premises():
    from graphrag.graph.inference_engine import DEFAULT_RULES, ForwardChainingEngine

    e = ForwardChainingEngine(MagicMock())
    trans = next(r for r in DEFAULT_RULES if r.rule_type == "transitivity")
    for q in (e._transitivity_query(trans), e._flip_query(), e._composition_query(),
              e._transitivity_query(trans, pinned=True)):
        assert "'RETRACTED'" in q and "valid_to" in q and "quarantined" in q
        assert "premises" in q


@pytest.mark.asyncio
async def test_derivation_for_pins_both_endpoints_and_returns_premises():
    from graphrag.graph.inference_engine import ForwardChainingEngine

    neo = AsyncMock()
    neo.run = AsyncMock(return_value=[{"conf": 0.8, "premises": ["ORG:A|FOUNDED|ORG:B"]}])
    out = await ForwardChainingEngine(neo).derivation_for(
        "founded_inverse", src_name="B", src_type="ORG", tgt_name="A", tgt_type="ORG", tenant=T)
    assert out["premises"] == ["ORG:A|FOUNDED|ORG:B"]
    kw = neo.run.await_args.kwargs
    assert kw["src_name"] == "B" and kw["tgt_name"] == "A" and kw["tenant"] == T
    assert await ForwardChainingEngine(neo).derivation_for(
        "no_such_rule", src_name="B", src_type="ORG", tgt_name="A", tgt_type="ORG", tenant=T) is None


# ── expiry sweep ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_expiry_sweep_emits_once_per_valid_to_and_marks_processed():
    from graphrag.graph.invalidation import expiry

    neo = AsyncMock()
    reads = {"docs": [{"id": "d1", "valid_to": "2025-01-01T00:00:00Z"}], "edges": []}

    async def run_read(q, **p):
        return reads["docs"] if q is expiry._EXPIRED_DOCS else reads["edges"]

    neo.run_read = AsyncMock(side_effect=run_read)
    with patch.object(expiry, "emit", AsyncMock(return_value={"fallback": None})) as em:
        out = await expiry.sweep_expired(neo, T)
        ev = em.await_args.args[0]
        assert ev.kind is EventKind.EVIDENCE_EXPIRED and ev.document_ids == ["d1"] and ev.tenant == T
        cause1 = ev.cause
        reads["docs"] = []
        assert (await expiry.sweep_expired(neo, T))["event"] is None
        assert em.await_count == 1
    assert out["documents"] == 1
    assert any("expiry_processed_for" in c.args[0] for c in neo.run.await_args_list)
    assert cause1  # deterministic over (id, valid_to)


# ── routes ────────────────────────────────────────────────────────────────────

def _route_client(router, tenant=T):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from api.auth.dependencies import get_current_user
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u1", "scope": "read write", "tenant": tenant}
    return TestClient(app)


@pytest.mark.parametrize("target,additive", [("RETRACTED", False), ("DISPUTED", False), ("APPROVED", True)])
def test_confidence_transition_is_targeted_when_weakening(target, additive):
    from api.routes.kg import confidence as routes

    neo = AsyncMock()
    neo.complete_corpus_update = AsyncMock(return_value=3)
    svc = AsyncMock()
    svc.transition_relation = AsyncMock(return_value={"event_id": "ev1", "from": "ASSERTED", "to": target})
    with patch.object(routes, "get_neo4j", return_value=neo), \
         patch("graphrag.graph.confidence_lifecycle.ConfidenceLifecycleService", return_value=svc), \
         patch("graphrag.graph.invalidation.emit", AsyncMock(return_value={"fallback": None})) as em:
        r = _route_client(routes.router).post("/confidence/transition", json={
            "src_name": "Acme", "src_type": "ORG", "relation": "OWNS", "tgt_name": "Bolt",
            "tgt_type": "ORG", "target_state": target})
    assert r.status_code == 200, r.text
    ev = em.await_args.args[0]
    assert ev.kind is EventKind.RELATION_CHANGED and ev.tenant == T and ev.cause == "ev1"
    assert ev.additive is additive
    kwargs = neo.complete_corpus_update.await_args.kwargs
    assert kwargs.get("advance_revision", True) is additive


def test_quarantine_single_entity_is_targeted_and_release_is_additive():
    from api.routes import corrections as routes

    neo = AsyncMock()
    neo.complete_corpus_update = AsyncMock(return_value=3)
    with patch.object(routes, "get_neo4j", return_value=neo), \
         patch.object(routes, "QuarantineService", return_value=AsyncMock()), \
         patch.object(routes, "emit", AsyncMock(return_value={"fallback": None})) as em:
        c = _route_client(routes.router)
        assert c.post("/entity/quarantine", json={"entity_name": "Acme", "entity_type": "ORG",
                                                   "reason": "bad"}).status_code == 200
        assert neo.complete_corpus_update.await_args.kwargs["advance_revision"] is False
        assert em.await_args.args[0].kind is EventKind.FACT_CORRECTED
        assert em.await_args.args[0].additive is False
        assert c.post("/entity/release", json={"entity_name": "Acme", "entity_type": "ORG",
                                                "released_by": "x"}).status_code == 200
        assert "advance_revision" not in neo.complete_corpus_update.await_args.kwargs
        assert em.await_args.args[0].additive is True


def test_edge_reject_emits_a_targeted_relation_change_with_resolved_types():
    from api.routes import corrections as routes

    neo = AsyncMock()
    neo.complete_corpus_update = AsyncMock(return_value=3)
    neo.run = AsyncMock(return_value=[{"deleted": 1, "confidence": 0.9, "source_doc_id": "d1",
                                       "endpoint_types": [{"src_type": "ORG", "tgt_type": "ORG"}]}])
    with patch.object(routes, "get_neo4j", return_value=neo), \
         patch("graphrag.graph.audit_trail.AuditTrail", return_value=AsyncMock()), \
         patch.object(routes, "emit", AsyncMock(return_value={"fallback": None})) as em:
        r = _route_client(routes.router).post("/edge/reject", json={
            "src_entity": "Acme", "tgt_entity": "Bolt", "relation": "OWNS"})
    assert r.status_code == 200, r.text
    ev = em.await_args.args[0]
    assert ev.relations[0].key == "ORG:Acme|OWNS|ORG:Bolt" and ev.additive is False
    assert neo.complete_corpus_update.await_args.kwargs["advance_revision"] is False
