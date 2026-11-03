"""GraphQL consumer API: every resolver goes through the MCP capability registry.

The registry is replaced by a recording fake, so these tests prove what the
GraphQL layer *sends* (capability id@version, arguments, identity) and how it
maps refusals; the registry's own enforcement is covered by its own tests.
Offline: no Neo4j, no LLM.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
import pytest
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import graphql_api
from mcp_server import audit
from mcp_server.registry import DeniedCapabilityCall

CLAIMS = {"scope": "read", "sub": "analyst-1", "tenant": "tenant-a"}


class FakeRegistry:
    """Records (name, args, identity); returns canned results per capability name."""

    def __init__(self, results: dict[str, Any] | None = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict, Any]] = []
        self._results = results or {}
        self._raises = raises

    async def call(self, name: str, args: dict, identity, *, dry_run: bool = False):
        self.calls.append((name, dict(args), identity))
        if self._raises is not None:
            raise self._raises
        return self._results.get(name, {"ok": True})


@pytest.fixture
def registry(monkeypatch) -> FakeRegistry:
    fake = FakeRegistry()
    monkeypatch.setattr(graphql_api, "get_registry", lambda: fake)
    return fake


def _client(claims: dict | None = CLAIMS) -> TestClient:
    app = FastAPI()
    app.include_router(graphql_api.router)
    if claims is not None:
        app.dependency_overrides[get_current_user] = lambda: claims
    return TestClient(app)


def _post(query: str, *, variables: dict | None = None, claims: dict | None = CLAIMS, **extra):
    body: dict[str, Any] = {"query": query, **extra}
    if variables is not None:
        body["variables"] = variables
    return _client(claims).post("/graphql", json=body)


class TestCapabilityMapping:
    @pytest.mark.parametrize("query,capability,args", [
        ("{ facts(question: \"q\") }", "kg.facts.query@1.0.0", {"question": "q"}),
        ("{ facts(question: \"q\", limit: 5) }", "kg.facts.query@1.0.0", {"question": "q", "limit": 5}),
        ("{ entity(name: \"n\", asOf: \"2026-01-01\", limit: 3) }", "kg.entity.lookup@1.0.0",
         {"name": "n", "as_of": "2026-01-01", "limit": 3}),
        ("{ precedents(policyVersionId: \"pv-1\", limit: 2) }", "cg.precedent.find@1.0.0",
         {"policy_version_id": "pv-1", "limit": 2}),
        ("{ capabilities }", "platform.capabilities.discover@1.0.0", {}),
        ("{ graphStats { entities } }", "kg.graph.stats@1.0.0", {}),
        ("{ evaluateKnowledge(domain: \"d\", context: {market: \"X\"}, semanticCandidates: [\"a\"],"
         " expectApplicableKnowledge: true) }", "kg.knowledge.evaluate@1.0.0",
         {"domain": "d", "context": {"market": "X"}, "semantic_candidates": ["a"],
          "expect_applicable_knowledge": True}),
    ])
    def test_field_calls_expected_capability_with_mapped_args(self, registry, query, capability, args):
        response = _post(query)
        assert response.status_code == 200, response.text
        assert [(name, call_args) for name, call_args, _ in registry.calls] == [(capability, args)]

    def test_graph_stats_exposes_typed_counts(self, monkeypatch):
        fake = FakeRegistry({"kg.graph.stats@1.0.0": {
            "entities": 4, "edges": 3, "communities": 2, "documents": 1,
            "provenance_receipt": {"operation_id": "op-1"}}})
        monkeypatch.setattr(graphql_api, "get_registry", lambda: fake)
        body = _post("{ graphStats { entities edges communities documents provenanceReceipt } }").json()
        assert body["data"]["graphStats"] == {
            "entities": 4, "edges": 3, "communities": 2, "documents": 1,
            "provenanceReceipt": {"operation_id": "op-1"}}

    def test_capability_payload_is_returned_as_json(self, monkeypatch):
        payload = {"facts": [{"a": 1}], "n": 1}
        fake = FakeRegistry({"kg.facts.query@1.0.0": payload})
        monkeypatch.setattr(graphql_api, "get_registry", lambda: fake)
        assert _post("{ facts(question: \"q\") }").json() == {"data": {"facts": payload}}

    def test_json_variables_reach_the_capability(self, registry):
        query = "query($c: JSON!) { evaluateKnowledge(domain: \"d\", context: $c) }"
        _post(query, variables={"c": {"market": "NORTHLAND"}})
        assert registry.calls[0][1] == {"domain": "d", "context": {"market": "NORTHLAND"}}

    def test_operation_name_selects_the_operation(self, registry):
        query = "query A { facts(question: \"a\") } query B { facts(question: \"b\") }"
        _post(query, operationName="B")
        assert registry.calls[0][1] == {"question": "b"}


class TestIdentityAndTenant:
    def test_identity_is_built_from_the_verified_token(self, registry):
        _post("{ capabilities }")
        identity = registry.calls[0][2]
        assert (identity.subject, identity.tenant, identity.authenticated, "read" in identity.scopes) == (
            "analyst-1", "tenant-a", True, True)

    def test_tenant_in_variables_never_reaches_the_registry(self, registry):
        _post("query($q: String!) { facts(question: $q) }", variables={"q": "x", "tenant": "tenant-b"})
        name, args, identity = registry.calls[0]
        assert ("tenant" in args, identity.tenant) == (False, "tenant-a")

    def test_caller_cannot_add_a_tenant_argument(self, registry):
        body = _post("{ facts(question: \"q\", tenant: \"tenant-b\") }").json()
        assert (registry.calls, "errors" in body) == ([], True)

    def test_tenant_nested_in_json_input_is_not_promoted_to_a_top_level_argument(self, registry):
        _post("{ evaluateKnowledge(domain: \"d\", context: {tenant: \"tenant-b\"}) }")
        assert "tenant" not in registry.calls[0][1]

    def test_schema_exposes_no_tenant_argument_anywhere(self):
        query_type = graphql_api.SCHEMA.query_type
        assert [f for f, d in query_type.fields.items() if "tenant" in d.args] == []


class TestAccessControl:
    def test_missing_read_scope_is_forbidden(self, registry):
        response = _post("{ capabilities }", claims={**CLAIMS, "scope": "write"})
        assert (response.status_code, registry.calls) == (403, [])

    def test_unauthenticated_request_is_rejected_before_any_call(self, registry):
        assert (_post("{ capabilities }", claims=None).status_code, registry.calls) == (401, [])

    def test_token_without_tenant_is_forbidden(self, registry):
        response = _post("{ capabilities }", claims={"scope": "read", "sub": "analyst-1"})
        assert (response.status_code, registry.calls) == (403, [])


class TestErrorMapping:
    def test_denial_becomes_a_graphql_error_with_the_reason_as_code(self, monkeypatch):
        denied = DeniedCapabilityCall(capability="kg.facts.query@1.0.0", reason="missing_scope",
                                      detail="missing scopes: ['read']", operation_id="op-9")
        monkeypatch.setattr(graphql_api, "get_registry",
                            lambda: FakeRegistry({"kg.facts.query@1.0.0": denied}))
        response = _post("{ facts(question: \"q\") }")
        error = response.json()["errors"][0]
        assert (response.status_code, error["extensions"]["code"], response.json()["data"]) == (
            200, "missing_scope", {"facts": None})

    def test_denial_error_carries_operation_id_and_detail(self, monkeypatch):
        denied = DeniedCapabilityCall(capability="kg.facts.query@1.0.0", reason="invalid_arg",
                                      detail="argument 'limit' too large", operation_id="op-9")
        monkeypatch.setattr(graphql_api, "get_registry",
                            lambda: FakeRegistry({"kg.facts.query@1.0.0": denied}))
        error = _post("{ facts(question: \"q\") }").json()["errors"][0]
        assert (error["message"], error["extensions"]["operationId"]) == ("argument 'limit' too large", "op-9")

    def test_error_result_becomes_a_graphql_error_with_its_code(self, monkeypatch):
        result = {"error": {"code": "no_published_release", "detail": "none for d"}}
        monkeypatch.setattr(graphql_api, "get_registry",
                            lambda: FakeRegistry({"kg.knowledge.evaluate@1.0.0": result}))
        error = _post("{ evaluateKnowledge(domain: \"d\", context: {}) }").json()["errors"][0]
        assert (error["extensions"]["code"], "none for d" in error["message"]) == ("no_published_release", True)

    def test_unexpected_exception_is_masked(self, monkeypatch):
        monkeypatch.setattr(graphql_api, "get_registry",
                            lambda: FakeRegistry(raises=RuntimeError("bolt://secret-host:7687 refused")))
        response = _post("{ capabilities }")
        error = response.json()["errors"][0]
        assert (response.status_code, error["extensions"]["code"], "secret-host" in response.text) == (
            200, "internal_error", False)

    def test_one_failing_field_does_not_hide_the_others(self, monkeypatch):
        denied = DeniedCapabilityCall(capability="x", reason="timeout", detail="t", operation_id="o")
        monkeypatch.setattr(graphql_api, "get_registry", lambda: FakeRegistry({
            "kg.facts.query@1.0.0": denied, "platform.capabilities.discover@1.0.0": {"capabilities": []}}))
        body = _post("{ facts(question: \"q\") capabilities }").json()
        assert (body["data"], len(body["errors"])) == ({"facts": None, "capabilities": {"capabilities": []}}, 1)

    def test_answer_generation_is_not_exposed(self, registry):
        # kg.answer.query writes answer traces and may call an LLM; this API is read-only
        body = _post("{ answer(question: \"q\") }").json()
        assert (registry.calls, "errors" in body) == ([], True)

    def test_syntax_error_is_a_graphql_error_not_a_server_error(self, registry):
        response = _post("{ facts(")
        assert (response.status_code, "errors" in response.json(), registry.calls) == (200, True, [])


class TestReadOnlySchema:
    def test_schema_has_no_mutation_or_subscription_type(self):
        schema = graphql_api.SCHEMA
        assert (schema.mutation_type, schema.subscription_type) == (None, None)

    def test_mutation_is_rejected_without_calling_the_registry(self, registry):
        body = _post("mutation { facts(question: \"q\") }").json()
        assert (registry.calls, "errors" in body) == ([], True)

    def test_introspection_works_with_read_scope(self, registry):
        body = _post("{ __schema { queryType { name } mutationType { name } } }").json()
        assert body["data"]["__schema"] == {"queryType": {"name": "Query"}, "mutationType": None}

    def test_standard_full_introspection_query_is_within_the_depth_limit(self, registry):
        from graphql import get_introspection_query
        response = _post(get_introspection_query())
        assert (response.status_code, "errors" in response.json()) == (200, False)


class TestGuardrails:
    def test_query_longer_than_the_cap_is_a_request_error(self, registry):
        padding = " " * graphql_api.MAX_QUERY_CHARS
        assert (_post("{ capabilities }" + padding).status_code, registry.calls) == (422, [])

    def test_unknown_body_field_is_rejected(self, registry):
        assert _post("{ capabilities }", extensions={"x": 1}).status_code == 422

    def test_oversized_variables_are_a_request_error(self, registry):
        big = {"v": "x" * graphql_api.MAX_VARIABLES_CHARS}
        assert _post("query($v: JSON) { capabilities }", variables=big).status_code == 422

    def test_selection_deeper_than_the_cap_is_refused_before_any_call(self, registry):
        depth = graphql_api.MAX_DEPTH
        query = "{ __schema { queryType " + "{ ofType " * depth + "{ name }" + " }" * depth + " } }"
        body = _post(query).json()
        assert (registry.calls, "depth" in str(body["errors"]).lower()) == ([], True)

    def test_fragments_cannot_hide_depth(self, registry):
        depth = graphql_api.MAX_DEPTH
        fragments = "".join(f"fragment G{i} on __Type {{ ofType {{ ...G{i + 1} }} }} " for i in range(depth))
        query = "{ __schema { queryType { ...G0 } } } " + fragments + f"fragment G{depth} on __Type {{ name }}"
        body = _post(query).json()
        assert (registry.calls, "depth" in str(body["errors"]).lower()) == ([], True)

    def test_too_many_top_level_fields_are_refused_before_any_call(self, registry):
        count = graphql_api.MAX_TOP_LEVEL_FIELDS + 1
        query = "{ " + " ".join(f"a{i}: capabilities" for i in range(count)) + " }"
        body = _post(query).json()
        assert (registry.calls, "top-level" in str(body["errors"]).lower()) == ([], True)

    def test_fields_at_the_cap_are_executed(self, registry):
        count = graphql_api.MAX_TOP_LEVEL_FIELDS
        query = "{ " + " ".join(f"a{i}: capabilities" for i in range(count)) + " }"
        _post(query)
        assert len(registry.calls) == count


class TestRealRegistry:
    def test_discover_runs_through_the_real_registry_and_audits_as_graphql(self, monkeypatch):
        events: list[tuple[dict, str]] = []

        async def fake_record(event, *, transport, neo4j=None):
            events.append((event, transport))

        monkeypatch.setattr(audit, "record_audit_event", fake_record)
        body = _post("{ capabilities }").json()
        names = {c["qualified_name"] for c in body["data"]["capabilities"]["capabilities"]}
        assert ("kg.facts.query@1.0.0" in names, [(e["tenant"], t) for e, t in events]) == (
            True, [("tenant-a", "graphql")])


class TestAuditTransport:
    @pytest.mark.asyncio
    async def test_audit_sink_labels_events_with_the_requested_transport(self, monkeypatch):
        seen: list[str] = []

        async def fake_record(event, *, transport, neo4j=None):
            seen.append(transport)

        monkeypatch.setattr(audit, "record_audit_event", fake_record)
        await audit.make_audit_sink("graphql")({"tenant": "t"})
        await audit.neo4j_audit_sink({"tenant": "t"})
        assert seen == ["graphql", "mcp"]
