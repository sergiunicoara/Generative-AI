"""Phase 7: guarded MCP / agent graph operations — adversarial tests.

Tenant-filter removal, scope/parameter override, Cypher injection, unrestricted
traversal, oversized results, unauthorized mutation, fabricated operation
names, timeouts, read-only execution, receipts and audit.
"""
from __future__ import annotations

import asyncio
import inspect
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag.graph.execution_scope import ResultTooLarge, current_scope, execution_scope
from mcp_server.identity import CallerIdentity
from mcp_server.registry import CapabilityRegistry, CapabilitySpec, DeniedCapabilityCall


def ident(**kw) -> CallerIdentity:
    base = dict(subject="agent-1", tenant="acme", scopes=frozenset({"read"}), token_type="m2m",
                authenticated=True)
    base.update(kw)
    return CallerIdentity(**base)


def reg(*specs, **kw) -> tuple[CapabilityRegistry, list]:
    audit: list = []

    async def sink(event):
        audit.append(event)

    r = CapabilityRegistry(audit_sink=sink, **kw)
    for s in specs:
        r.register(s)
    return r, audit


def read_spec(fn=None, **kw) -> CapabilitySpec:
    async def _echo(tenant: str, name: str = "", limit: int = 10) -> dict:
        return {"tenant": tenant, "name": name, "limit": limit, "scope": current_scope()}

    return CapabilitySpec(capability_id="kg.test.read", version="1.0.0", title="t", kind="read", risk="safe",
                          fn=fn or _echo, required_scopes=("read",),
                          arg_schema={"tenant": {"type": str}, "name": {"type": str, "max_length": 64},
                                      "limit": {"type": int, "min": 1, "max": 100}}, **kw)


# ── tenant filter removal / scope override ───────────────────────────────────

@pytest.mark.asyncio
async def test_caller_cannot_name_another_tenant():
    r, audit = reg(read_spec())
    out = await r.call("kg.test.read", {"tenant": "victim"}, ident())
    assert isinstance(out, DeniedCapabilityCall) and out.reason == "tenant_mismatch"
    assert audit[-1]["outcome"] == "tenant_mismatch"


@pytest.mark.asyncio
async def test_omitted_tenant_is_bound_to_the_identity_not_a_default():
    r, _ = reg(read_spec())
    out = await r.call("kg.test.read", {"name": "x"}, ident(tenant="acme"))
    assert out["tenant"] == "acme"


@pytest.mark.asyncio
async def test_identity_without_tenant_is_refused():
    r, _ = reg(read_spec())
    out = await r.call("kg.test.read", {}, ident(tenant=""))
    assert out.reason == "tenant_required"


@pytest.mark.asyncio
@pytest.mark.parametrize("smuggled", [
    {"scopes": ["admin"]}, {"identity": "admin"}, {"tenant_override": "victim"},
    {"cypher": "MATCH (n) DETACH DELETE n"}, {"label": "Secret"}, {"rel_type": "OWNS"},
    {"where": "1=1"}, {"depth": 50},
])
async def test_undeclared_arguments_are_rejected(smuggled):
    r, _ = reg(read_spec())
    out = await r.call("kg.test.read", {"name": "x", **smuggled}, ident())
    assert isinstance(out, DeniedCapabilityCall) and out.reason == "invalid_arg"
    assert "unknown argument" in out.detail


# ── Cypher injection ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_injection_text_only_ever_reaches_cypher_as_a_parameter():
    from mcp_server.tools import lookup_entity

    neo = MagicMock()
    neo.get_relations_for_entity = AsyncMock(return_value=[])
    registry = MagicMock()
    registry.resolve = MagicMock(return_value=("ORG", "ORG"))
    payload = "x'}) DETACH DELETE (n) //"
    with patch("mcp_server.tools.get_neo4j", return_value=neo), \
         patch("mcp_server.tools.load_alias_registry", AsyncMock(return_value=registry), create=True):
        try:
            await lookup_entity(payload, tenant="acme")
        except Exception:  # noqa: BLE001 - only the query shape matters here
            pass
    from graphrag.graph.neo4j_client import Neo4jClient
    c = Neo4jClient.__new__(Neo4jClient)
    c.run = AsyncMock(return_value=[])
    await c.get_relations_for_entity(payload, "ORG", tenant="acme")
    q, kw = c.run.await_args.args[0], c.run.await_args.kwargs
    assert payload not in q and kw["name"] == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["ORG) DETACH DELETE (n", "ORG`", "1ORG", "ORG:Secret", "a" * 65])
async def test_label_like_arguments_must_be_identifiers(bad):
    from graphrag.agents.tool_policy import ToolPolicy

    policy = ToolPolicy.from_defaults(caller_scopes=["write", "quarantine", "tenant:acme"])
    out = await policy.call("quarantine_entity",
                            {"entity_name": "X", "entity_type": bad, "tenant": "acme"}, tenant="acme")
    assert getattr(out, "reason", None) == "invalid_arg"


@pytest.mark.asyncio
async def test_entity_lookup_rejects_non_iso_as_of_and_long_names():
    from mcp_server.capabilities import build_registry

    r = build_registry()
    r._audit_sink = AsyncMock()
    out = await r.call("kg.entity.lookup", {"name": "X", "as_of": "2024-01-01' OR 1=1"}, ident())
    assert out.reason == "invalid_arg"
    out = await r.call("kg.entity.lookup", {"name": "X" * 300}, ident())
    assert out.reason == "invalid_arg"


def test_no_capability_or_tool_accepts_query_language_or_schema_names():
    from graphrag.agents.tool_policy import ToolPolicy
    from mcp_server.capabilities import build_registry

    forbidden = re.compile(r"cypher|label|rel(ationship)?_?type|property|where|clause|sparql", re.I)
    for spec in build_registry()._by_qualified.values():
        assert not any(forbidden.search(k) for k in spec.arg_schema), spec.qualified_name
    for spec in ToolPolicy.from_defaults()._tools.values():
        assert not any(forbidden.search(k) for k in spec.arg_schema), spec.name


# ── unrestricted traversal / oversized results ───────────────────────────────

@pytest.mark.asyncio
async def test_limits_above_the_cap_are_rejected():
    r, _ = reg(read_spec())
    assert (await r.call("kg.test.read", {"limit": 100_000}, ident())).reason == "invalid_arg"


@pytest.mark.asyncio
async def test_oversized_result_is_refused():
    async def big(tenant: str) -> dict:
        return {"rows": ["x" * 1000] * 50}

    r, audit = reg(read_spec(fn=big, max_result_bytes=10_000))
    out = await r.call("kg.test.read", {}, ident())
    assert out.reason == "result_too_large" and audit[-1]["outcome"] == "result_too_large"


class _Rec:
    def __init__(self, i):
        self.i = i

    def data(self):
        return {"i": self.i}


class _Result:
    def __init__(self, n):
        self.n = n

    def __aiter__(self):
        async def gen():
            for i in range(self.n):
                yield _Rec(i)
        return gen()


class _Session:
    def __init__(self, n, log):
        self.n, self.log = n, log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, parameters=None):
        self.log.append(("run", query))
        return _Result(self.n)


class _Driver:
    def __init__(self, n=5):
        self.n, self.log = n, []

    def session(self, **kwargs):
        self.log.append(("session", kwargs))
        return _Session(self.n, self.log)


def _client(n=5):
    from graphrag.graph.neo4j_client import Neo4jClient
    c = Neo4jClient.__new__(Neo4jClient)
    c._driver = _Driver(n)
    c._in_flight = 0
    return c


@pytest.mark.asyncio
async def test_row_limit_stops_reading_and_raises():
    c = _client(n=50)
    with execution_scope(read_only=True, max_rows=10, operation="op"):
        with pytest.raises(ResultTooLarge):
            await c.run("MATCH (n) RETURN n")


@pytest.mark.asyncio
async def test_read_operations_use_a_read_session_with_a_server_timeout():
    from neo4j import READ_ACCESS, Query

    c = _client()
    with execution_scope(read_only=True, timeout_s=5.0):
        await c.run("MATCH (n) RETURN n")
    (_, kwargs), (_, query) = c._driver.log[0], c._driver.log[1]
    assert kwargs == {"default_access_mode": READ_ACCESS}
    assert isinstance(query, Query) and query.timeout == 5.0
    c2 = _client()
    await c2.run("MATCH (n) RETURN n")  # outside any guarded operation: unchanged
    assert c2._driver.log[0] == ("session", {}) and c2._driver.log[1][1] == "MATCH (n) RETURN n"


@pytest.mark.asyncio
async def test_registry_runs_read_capabilities_inside_a_read_only_scope():
    r, _ = reg(read_spec())
    out = await r.call("kg.test.read", {}, ident())
    assert out["scope"].read_only is True and out["scope"].max_rows == 1000
    assert out["scope"].operation == "kg.test.read@1.0.0"
    assert current_scope() is None  # restored afterwards


@pytest.mark.asyncio
async def test_sync_capabilities_also_see_the_scope_in_their_thread():
    def sync_fn(tenant: str) -> dict:
        s = current_scope()
        return {"read_only": bool(s and s.read_only)}

    r, _ = reg(read_spec(fn=sync_fn))
    assert (await r.call("kg.test.read", {}, ident()))["read_only"] is True


@pytest.mark.asyncio
async def test_slow_operation_times_out():
    async def slow(tenant: str) -> dict:
        await asyncio.sleep(1)
        return {}

    r, _ = reg(read_spec(fn=slow, timeout_s=0.05))
    assert (await r.call("kg.test.read", {}, ident())).reason == "timeout"


# ── unauthorized mutation ────────────────────────────────────────────────────

def write_spec(**kw) -> CapabilitySpec:
    async def _mutate(tenant: str) -> dict:
        return {"outcome": "executed", "tenant": tenant}

    return CapabilitySpec(capability_id="kg.test.write", version="1.0.0", title="w", kind="write",
                          risk="destructive", fn=_mutate, required_scopes=("write",), **kw)


@pytest.mark.asyncio
async def test_write_without_an_approval_path_is_never_executed():
    r, _ = reg(write_spec())
    out = await r.call("kg.test.write", {}, ident(scopes=frozenset({"write"})))
    assert out.reason == "approval_required"


@pytest.mark.asyncio
async def test_approval_hook_decides_and_sees_the_identity():
    seen = {}

    async def deny(spec, args, identity):
        seen["subject"] = identity.subject
        return False

    r, _ = reg(write_spec(), approval_hook=deny)
    out = await r.call("kg.test.write", {}, ident(scopes=frozenset({"write"})))
    assert out.reason == "approval_denied" and seen["subject"] == "agent-1"

    async def allow(spec, args, identity):
        return True

    r2, _ = reg(write_spec(), approval_hook=allow)
    out = await r2.call("kg.test.write", {}, ident(scopes=frozenset({"write"})))
    assert out["outcome"] == "executed" and out["provenance_receipt"]["kind"] == "write"
    assert out["provenance_receipt"]["read_only_session"] is False


@pytest.mark.asyncio
async def test_read_scope_cannot_invoke_a_write():
    r, _ = reg(write_spec(approval_enforced_by="svc"))
    assert (await r.call("kg.test.write", {}, ident())).reason == "missing_scope"


def test_real_write_capabilities_declare_their_approval_path():
    from mcp_server.capabilities import build_registry
    for spec in build_registry()._by_qualified.values():
        if spec.kind != "read":
            assert spec.requires_approval and spec.approval_enforced_by, spec.qualified_name


# ── fabricated operations, receipts, audit ───────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["kg.cypher.run", "kg.test.read@9.9.9", "../../admin", "x" * 500])
async def test_fabricated_operation_names_are_denied_with_a_bounded_metric_label(name):
    r, audit = reg(read_spec())
    with patch("mcp_server.registry.record_capability_call") as metric:
        out = await r.call(name, {}, ident())
    assert out.reason == "not_found" and out.operation_id
    assert metric.call_args.kwargs["capability"] == "unknown"
    assert audit[-1]["operation"] is None and len(audit[-1]["requested_name"]) <= 200


@pytest.mark.asyncio
async def test_receipt_and_audit_for_a_successful_read():
    r, audit = reg(read_spec())
    out = await r.call("kg.test.read", {"name": "Acme"}, ident())
    receipt = out["provenance_receipt"]
    assert receipt["operation"] == "kg.test.read@1.0.0" and receipt["tenant"] == "acme"
    assert receipt["subject"] == "agent-1" and receipt["read_only_session"] is True
    assert len(receipt["args_sha256"]) == 64 and len(receipt["result_sha256"]) == 64
    [event] = audit
    assert event["operation_id"] == receipt["operation_id"] and event["outcome"] == "executed"
    assert "Acme" not in str(event)  # arguments are hashed, never stored


@pytest.mark.asyncio
async def test_audit_failure_never_changes_the_result():
    async def broken(event):
        raise RuntimeError("db down")

    r = CapabilityRegistry(audit_sink=broken)
    r.register(read_spec())
    out = await r.call("kg.test.read", {}, ident())
    assert out["tenant"] == "acme"


@pytest.mark.asyncio
async def test_answer_capability_records_traces_so_is_not_forced_read_only():
    from mcp_server.capabilities import build_registry
    spec = build_registry()._find("kg.answer.query")
    assert spec.kind == "read" and spec.read_only_session is False
    assert spec.arg_schema["mode"]["allowed"] == ["local", "global", "hybrid"]


# ── REST agent tools ─────────────────────────────────────────────────────────

def _agent_client(scope="read write quarantine admin gdpr_officer tenant:acme", tenant="acme"):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from api.auth.dependencies import get_current_user
    from api.routes import agent
    app = FastAPI()
    app.include_router(agent.router)
    app.dependency_overrides[get_current_user] = lambda: {"sub": "real-user", "scope": scope, "tenant": tenant}
    return TestClient(app)


@pytest.fixture
def no_audit():
    with patch("mcp_server.audit.record_audit_event", AsyncMock()) as m:
        yield m


def test_rest_mutation_requires_confirmation(no_audit):
    r = _agent_client().post("/tool", json={"tool": "quarantine_entity", "args": {
        "entity_name": "X", "entity_type": "ORG", "tenant": "acme"}})
    assert r.json()["reason"] == "confirmation_required"
    assert no_audit.await_args.args[0]["outcome"] == "confirmation_required"


def test_rest_mutation_cannot_target_another_tenant_even_with_its_scope(no_audit):
    r = _agent_client(scope="read write quarantine tenant:acme tenant:victim").post("/tool", json={
        "tool": "quarantine_entity", "confirm": True,
        "args": {"entity_name": "X", "entity_type": "ORG", "tenant": "victim"}})
    assert r.json()["reason"] == "tenant_mismatch"


def test_rest_erase_actor_comes_from_the_token(no_audit):
    from graphrag.agents import tool_policy

    captured = {}

    async def fake_call(self, tool, args, tenant="default"):
        captured.update(args)
        return {"erased": True}

    with patch.object(tool_policy.ToolPolicy, "call", fake_call):
        r = _agent_client().post("/tool", json={"tool": "erase_entity", "confirm": True, "args": {
            "entity_name": "X", "entity_type": "ORG", "tenant": "acme", "requested_by": "someone-else"}})
    assert r.json()["outcome"] == "executed"
    assert captured["requested_by"] == "real-user" and captured["tenant"] == "acme"


def test_rest_graph_tools_are_unavailable_under_acl(no_audit):
    settings = MagicMock()
    settings.access_control = {"enabled": True}
    with patch("api.routes.agent.get_settings", return_value=settings):
        r = _agent_client().post("/tool", json={"tool": "get_neighbors", "args": {"entity_name": "X"}})
    assert r.json()["reason"] == "acl_unsupported"


def test_rest_audit_is_durable_and_tenant_scoped():
    with patch("mcp_server.audit.list_audit_events", AsyncMock(return_value=[{"operation_id": "o"}])) as m:
        r = _agent_client().get("/audit")
    assert r.json()["entries"] == [{"operation_id": "o"}]
    assert m.await_args.args[0] == "acme"


def test_erase_and_quarantine_go_through_audited_services():
    from graphrag.agents import tool_policy
    src = inspect.getsource(tool_policy.ToolPolicy.from_defaults)
    assert "GDPRService(get_neo4j()).forget_entity" in src
    assert "QuarantineService(neo4j).quarantine_entity" in src
    assert "tenant:$tn}) DETACH DELETE e" not in src and "SET e.quarantined=true" not in src


# ── RDF export / SPARQL ──────────────────────────────────────────────────────

def test_rdf_export_is_strictly_single_tenant():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" / "export_rdf.py").read_text(encoding="utf-8")
    assert "$tenant = 'default' OR" not in src
    assert 'parser.add_argument("--tenant", required=True' in src
    assert "coalesce(r.confidence_state, 'ASSERTED') <> 'RETRACTED'" in src


def test_sparql_persist_requires_admin(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from api.auth.dependencies import get_current_user
    from api.routes.kg import knowledge
    app = FastAPI()
    app.include_router(knowledge.router)
    app.dependency_overrides[get_current_user] = lambda: {"sub": "u", "scope": "read write", "tenant": "acme"}
    r = TestClient(app).post("/sparql/update", json={
        "query": 'INSERT DATA { <http://e/x> <http://e/p> "v" }', "persist": True})
    assert r.status_code == 403
