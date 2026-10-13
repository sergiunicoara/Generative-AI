"""Hash-chained auditor sign-off: chain verification, tamper detection, append retry, and the API boundary."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import iatf_audit as routes
from graphrag.domains.automotive import signoff as so
from graphrag.graph.validation.graph_checks import assert_read_only


def _chain(n):
    entries, prev = [], so.GENESIS
    for seq in range(1, n + 1):
        e = {"id": f"id{seq}", "tenant": "t", "seq": seq, "clause": "8.4", "decision": "accepted",
             "reviewer": "r", "note": "", "evidence_digest": "d", "evidence_as_of": None, "signed_at": "s",
             "prev_hash": prev}
        e["hash"] = so.entry_hash(e)
        entries.append(e)
        prev = e["hash"]
    return entries


def test_intact_chain_verifies_and_reports_head():
    chain = _chain(3)
    assert so.verify_chain(chain, head_hash=chain[-1]["hash"]) == {"valid": True, "entries": 3,
                                                                   "head": chain[-1]["hash"]}
    assert so.verify_chain([], head_hash=so.GENESIS)["valid"] is True


@pytest.mark.parametrize("tamper,reason,at", [
    (lambda c: c[1].update(decision="rejected"), "content changed", 2),
    (lambda c: c.pop(1), "sequence gap: expected 2, found 3", 2),
    (lambda c: c[2].update(prev_hash="f" * 64), "prev_hash mismatch", 3),
])
def test_any_edit_or_removal_breaks_the_chain(tamper, reason, at):
    chain = _chain(3)
    head = chain[-1]["hash"]
    tamper(chain)
    result = so.verify_chain(chain, head_hash=head)
    assert result["valid"] is False and result["reason"] == reason and result["broken_at_seq"] == at


def test_truncating_the_tail_is_caught_by_the_head():
    chain = _chain(3)
    result = so.verify_chain(chain[:2], head_hash=chain[-1]["hash"])
    assert result["valid"] is False and "head" in result["reason"]


def test_read_queries_are_read_only_and_append_locks_before_comparing():
    assert_read_only(so.HEAD)
    assert_read_only(so.LIST)
    lock = so.APPEND.index("SET h.locked_at")
    assert lock < so.APPEND.index("WHERE h.hash = $prev_hash")


class _FakeNeo4j:
    """Simulates the head moving between read and conditional append ``moves`` times."""

    def __init__(self, moves=0):
        self.moves, self.head, self.entries, self.schema = moves, {"hash": so.GENESIS, "seq": 0}, [], 0

    async def run(self, cypher, **p):
        if cypher.startswith("CREATE CONSTRAINT"):
            self.schema += 1
            return []
        if cypher is so.HEAD:
            return [dict(self.head)]
        if cypher is so.APPEND:
            if self.moves:
                self.moves -= 1
                return []
            assert p["prev_hash"] == self.head["hash"]
            self.head = {"hash": p["hash"], "seq": p["seq"]}
            self.entries.append(p)
            return [{"id": p["id"]}]
        if cypher is so.LIST:
            return [{"entry": e} for e in self.entries if p["clause"] in (None, e["clause"])]
        raise AssertionError(cypher)


@pytest.fixture(autouse=True)
def _reset_schema_flag():
    so.SignoffStore._schema_ready = False
    yield
    so.SignoffStore._schema_ready = False


async def test_append_builds_a_valid_chain_and_creates_constraints_once():
    neo = _FakeNeo4j()
    store = so.SignoffStore(neo)
    first = await store.append(tenant="t", clause="8.4", decision="accepted", reviewer="a", evidence_digest="d1")
    await store.append(tenant="t", clause="8.7", decision="needs_more_evidence", reviewer="b", evidence_digest="d2")
    assert neo.schema == len(so.SCHEMA) and first["seq"] == 1
    assert (await store.verify("t"))["valid"] is True
    assert set(await store.latest_by_clause("t")) == {"8.4", "8.7"}


async def test_append_retries_when_the_head_moves_and_gives_up_after_the_budget():
    store = so.SignoffStore(_FakeNeo4j(moves=2), max_retries=3)
    assert (await store.append(tenant="t", clause="8.4", decision="accepted", reviewer="a",
                               evidence_digest="d"))["seq"] == 1
    with pytest.raises(so.ChainConflict):
        await so.SignoffStore(_FakeNeo4j(moves=5), max_retries=3).append(
            tenant="t", clause="8.4", decision="accepted", reviewer="a", evidence_digest="d")


@pytest.mark.parametrize("kw,msg", [({"decision": "approved"}, "decision"), ({"reviewer": ""}, "reviewer"),
                                    ({"evidence_digest": ""}, "required")])
async def test_append_validates_inputs(kw, msg):
    args = {"tenant": "t", "clause": "8.4", "decision": "accepted", "reviewer": "a", "evidence_digest": "d", **kw}
    with pytest.raises(ValueError, match=msg):
        await so.SignoffStore(_FakeNeo4j()).append(**args)


# ── API boundary ──────────────────────────────────────────────────────────────────────────────────────────────
def _client(tenant="automotive", sub="auditor-1", scope="read write"):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": scope, "sub": sub, "tenant": tenant}
    return TestClient(app)


def test_tenant_without_a_configured_corpus_gets_404():
    assert _client(tenant="aerospace").get("/audit/iatf/clauses").status_code == 404


def test_coverage_route_serves_the_configured_tenant():
    body = _client().get("/audit/iatf/clauses").json()
    assert body["documents_considered"] == 30 and body["mapping_status"] == "proposed"


def test_signoff_takes_the_reviewer_from_the_token_and_the_digest_from_the_server():
    neo = _FakeNeo4j()
    with patch.object(routes, "get_neo4j", return_value=neo):
        response = _client().post("/audit/iatf/clauses/8.4/signoff",
                                  json={"decision": "accepted", "note": "ok", "reviewer": "someone-else",
                                        "evidence_digest": "forged"})
    assert response.status_code == 200
    entry = response.json()
    assert entry["reviewer"] == "auditor-1" and entry["evidence_digest"] != "forged" and len(entry["hash"]) == 64


def test_signoff_without_subject_is_forbidden_and_read_scope_cannot_sign():
    with patch.object(routes, "get_neo4j", return_value=_FakeNeo4j()):
        assert _client(sub="").post("/audit/iatf/clauses/8.4/signoff",
                                    json={"decision": "accepted"}).status_code == 403
        assert _client(scope="read").post("/audit/iatf/clauses/8.4/signoff",
                                          json={"decision": "accepted"}).status_code == 403


def test_unknown_clause_is_404_and_csv_export_works():
    with patch.object(routes, "get_neo4j", return_value=_FakeNeo4j()):
        client = _client()
        assert client.post("/audit/iatf/clauses/99.9/signoff", json={"decision": "accepted"}).status_code == 404
        csv_text = client.get("/audit/iatf/export?format=csv").text
    assert csv_text.startswith("clause,heading,status")
