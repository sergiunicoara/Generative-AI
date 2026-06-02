"""Guardrail tests for the agent tool safety layer.

Proves the ToolPolicy gate works correctly for every category of denied
request.  No live services — all high-risk tool functions are stubbed.

Test categories matching the spec:
  1. unsafe tool denied        (risk=high/restricted without required scopes)
  2. unknown tool denied       (not in allowlist)
  3. missing tenant denied     (required tenant arg missing)
  4. cross-tenant denied       (caller scoped to A, request references B)
  5. invalid args denied       (wrong type, out-of-range, bad enum)
  6. high-risk scope required  (ingest/quarantine/erase need write+admin)
  7. timeout handled           (async tool that hangs → DeniedAction)
  8. audit log written         (every call recorded, outcome correct)
"""

from __future__ import annotations

import asyncio
import pytest

from graphrag.agents.tool_policy import DeniedAction, ToolPolicy, ToolSpec


# ── Shared stubs ──────────────────────────────────────────────────────────────

async def _read_tool(question: str = "", tenant: str = "") -> list[dict]:
    return [{"result": "ok"}]

async def _write_tool(entity_name: str = "", entity_type: str = "",
                      tenant: str = "", reason: str = "",
                      doc_type: str = "") -> dict:
    return {"status": "done"}

async def _erase_tool(entity_name: str = "", entity_type: str = "",
                      tenant: str = "", requested_by: str = "") -> dict:
    return {"erased": True}

async def _read_tool_with_tenant(question: str = "", tenant: str = "") -> list[dict]:
    return [{"result": "ok"}]

async def _slow_tool(question: str = "") -> list[dict]:
    await asyncio.sleep(99)
    return []


def _make_policy(scopes: list[str] | None = None, dry_run: bool = False) -> ToolPolicy:
    """Build a test policy with the full risk-level spectrum."""
    tools = [
        # LOW RISK
        ToolSpec("local_search",  _read_tool_with_tenant, scopes=["read"],
                 timeout_s=5.0, risk="low",
                 arg_schema={"question": {"type": str, "required": True},
                             "tenant":   {"type": str}}),
        ToolSpec("global_search", _read_tool, scopes=["read"],
                 timeout_s=5.0, risk="low",
                 arg_schema={"question": {"type": str, "required": True}}),
        ToolSpec("get_neighbors", _read_tool, scopes=["read"],
                 timeout_s=5.0, risk="low",
                 arg_schema={"question": {"type": str, "required": True}}),

        # MEDIUM RISK
        ToolSpec("search_graph",  _read_tool, scopes=["read"],
                 timeout_s=5.0, risk="medium",
                 arg_schema={"question": {"type": str, "required": True},
                             "top_k":    {"type": int, "min": 1, "max": 50}}),

        # HIGH RISK
        ToolSpec("ingest_document", _write_tool, scopes=["write", "ingest"],
                 timeout_s=60.0, risk="high",
                 arg_schema={"entity_name": {"type": str, "required": True},
                             "entity_type": {"type": str, "required": True},
                             "tenant":      {"type": str, "required": True},
                             "doc_type":    {"type": str,
                                             "allowed": {"regulatory", "manufacturer",
                                                         "internal", "informal"}},
                             "reason":      {"type": str}}),
        ToolSpec("quarantine_entity", _write_tool, scopes=["write", "quarantine"],
                 timeout_s=10.0, risk="high",
                 arg_schema={"entity_name": {"type": str, "required": True},
                             "entity_type": {"type": str, "required": True},
                             "tenant":      {"type": str, "required": True},
                             "reason":      {"type": str}}),

        # RESTRICTED
        ToolSpec("erase_entity", _erase_tool,
                 scopes=["write", "admin", "gdpr_officer"],
                 timeout_s=30.0, risk="restricted",
                 arg_schema={"entity_name":  {"type": str, "required": True},
                             "entity_type":  {"type": str, "required": True},
                             "tenant":       {"type": str, "required": True},
                             "requested_by": {"type": str, "required": True}}),

        # For timeout tests
        ToolSpec("slow_tool", _slow_tool, scopes=["read"],
                 timeout_s=0.05, risk="low",
                 arg_schema={"question": {"type": str}}),
    ]
    return ToolPolicy(tools=tools, caller_scopes=scopes or [], dry_run=dry_run)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unsafe tool denied
# ═══════════════════════════════════════════════════════════════════════════

class TestUnsafeToolDenied:
    """High-risk and restricted tools must be denied without the correct scopes."""

    @pytest.mark.parametrize("tool,required_scopes,partial_scopes", [
        ("ingest_document",   ["write", "ingest"],              ["write"]),
        ("quarantine_entity", ["write", "quarantine"],          ["write"]),
        ("erase_entity",      ["write", "admin", "gdpr_officer"], ["write", "admin"]),
    ])
    def test_high_risk_tool_denied_with_partial_scopes(self, tool, required_scopes, partial_scopes):
        p = _make_policy(scopes=partial_scopes)
        args = {"entity_name": "Boeing", "entity_type": "ORG",
                "tenant": "aerospace", "reason": "test",
                "requested_by": "admin", "doc_type": "regulatory"}
        result = asyncio.run(p.call(tool, args, tenant="aerospace"))
        assert isinstance(result, DeniedAction), f"{tool} should be denied with {partial_scopes}"
        assert result.reason == "missing_scope"

    def test_restricted_erase_requires_gdpr_officer(self):
        p = _make_policy(scopes=["write", "admin"])   # missing gdpr_officer
        result = asyncio.run(p.call(
            "erase_entity",
            {"entity_name": "Boeing", "entity_type": "ORG",
             "tenant": "aerospace", "requested_by": "admin"},
        ))
        assert isinstance(result, DeniedAction)
        assert "gdpr_officer" in result.detail

    def test_high_risk_tool_allowed_with_full_scopes(self):
        p = _make_policy(scopes=["read", "write", "ingest"])
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "doc_001", "entity_type": "Document",
             "tenant": "aerospace", "doc_type": "regulatory"},
        ))
        assert isinstance(result, dict)
        assert result.get("status") == "done"

    def test_erase_allowed_with_full_scopes(self):
        p = _make_policy(scopes=["write", "admin", "gdpr_officer"])
        result = asyncio.run(p.call(
            "erase_entity",
            {"entity_name": "Boeing", "entity_type": "ORG",
             "tenant": "aerospace", "requested_by": "legal_team"},
        ))
        assert isinstance(result, dict)
        assert result.get("erased") is True


# ═══════════════════════════════════════════════════════════════════════════
# 2. Unknown tool denied
# ═══════════════════════════════════════════════════════════════════════════

class TestUnknownToolDenied:
    """Tools not in the allowlist must be refused regardless of scopes."""

    @pytest.mark.parametrize("tool_name", [
        "drop_database",
        "exec_cypher",
        "delete_tenant",
        "run_arbitrary_code",
        "__import__",
        "",
    ])
    def test_unknown_tool_denied(self, tool_name):
        p = _make_policy(scopes=["read", "write", "admin", "gdpr_officer"])
        result = asyncio.run(p.call(tool_name, {}))
        assert isinstance(result, DeniedAction)
        assert result.reason == "not_allowed"
        assert result.tool == tool_name

    def test_known_low_risk_tool_executes(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("local_search", {"question": "FAA AD 2024"}))
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Missing tenant denied
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingTenantDenied:
    """Tools that require a tenant arg must fail if it's absent."""

    @pytest.mark.parametrize("tool", [
        "ingest_document", "quarantine_entity", "erase_entity"
    ])
    def test_missing_required_tenant_denied(self, tool):
        p = _make_policy(scopes=["read", "write", "ingest", "quarantine",
                                  "admin", "gdpr_officer"])
        # Provide all other required args but omit tenant
        args = {
            "entity_name": "Boeing", "entity_type": "ORG",
            "requested_by": "admin", "doc_type": "regulatory",
            "reason": "test",
        }
        result = asyncio.run(p.call(tool, args))
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "tenant" in result.detail


# ═══════════════════════════════════════════════════════════════════════════
# 4. Cross-tenant request denied
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossTenantDenied:
    """A caller scoped to tenant A cannot reference tenant B in a tool arg."""

    def test_cross_tenant_write_denied(self):
        """Caller is aerospace; tries to ingest into banking."""
        p = _make_policy(scopes=["read", "write", "ingest", "tenant:aerospace"])
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "Goldman Sachs", "entity_type": "ORG",
             "tenant": "banking", "doc_type": "regulatory"},
        ))
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "cross-tenant" in result.detail

    def test_cross_tenant_erase_denied(self):
        """GDPR officer for aerospace cannot erase banking entities."""
        p = _make_policy(scopes=["write", "admin", "gdpr_officer", "tenant:aerospace"])
        result = asyncio.run(p.call(
            "erase_entity",
            {"entity_name": "BNP Paribas", "entity_type": "ORG",
             "tenant": "banking", "requested_by": "dpo"},
        ))
        assert isinstance(result, DeniedAction)
        assert "cross-tenant" in result.detail

    def test_same_tenant_allowed(self):
        p = _make_policy(scopes=["read", "write", "ingest", "tenant:aerospace"])
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "FAA", "entity_type": "REGULATOR",
             "tenant": "aerospace", "doc_type": "regulatory"},
        ))
        assert isinstance(result, dict)

    def test_no_tenant_scope_passes_all_tenants(self):
        """When caller has no tenant: scope, no cross-tenant check is applied."""
        p = _make_policy(scopes=["read", "write", "ingest"])
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "Goldman", "entity_type": "ORG",
             "tenant": "banking", "doc_type": "internal"},
        ))
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Invalid args denied
# ═══════════════════════════════════════════════════════════════════════════

class TestInvalidArgsDenied:
    """Argument schema violations must be caught before execution."""

    def test_wrong_type_denied(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"question": 42}))  # int not str
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "str" in result.detail

    def test_top_k_above_max_denied(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"question": "FAA", "top_k": 999}))
        assert isinstance(result, DeniedAction)
        assert "maximum" in result.detail

    def test_top_k_below_min_denied(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"question": "FAA", "top_k": 0}))
        assert isinstance(result, DeniedAction)
        assert "minimum" in result.detail

    def test_bad_enum_value_denied(self):
        """doc_type must be one of the allowed set."""
        p = _make_policy(scopes=["read", "write", "ingest"])
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "FAA", "entity_type": "ORG",
             "tenant": "aerospace", "doc_type": "classified"},   # not in allowed set
        ))
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "allowed" in result.detail or "classified" in result.detail

    def test_missing_required_arg_denied(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("local_search", {}))   # question is required
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "question" in result.detail

    def test_valid_args_execute(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("local_search", {"question": "What is FAA?"}))
        assert isinstance(result, list)


# ═══════════════════════════════════════════════════════════════════════════
# 6. High-risk tool requires scope
# ═══════════════════════════════════════════════════════════════════════════

class TestHighRiskScopeRequired:
    """Every combination of partial scopes for high-risk tools must be denied."""

    @pytest.mark.parametrize("scopes", [
        [],
        ["read"],
        ["write"],
        ["ingest"],
        ["read", "write"],
    ])
    def test_ingest_document_partial_scopes_denied(self, scopes):
        p = _make_policy(scopes=scopes)
        result = asyncio.run(p.call(
            "ingest_document",
            {"entity_name": "FAA", "entity_type": "ORG",
             "tenant": "aerospace", "doc_type": "regulatory"},
        ))
        assert isinstance(result, DeniedAction), f"should deny with scopes={scopes}"

    @pytest.mark.parametrize("scopes", [
        [], ["read"], ["write"], ["admin"], ["gdpr_officer"],
        ["write", "admin"], ["admin", "gdpr_officer"],
    ])
    def test_erase_entity_partial_scopes_denied(self, scopes):
        p = _make_policy(scopes=scopes)
        result = asyncio.run(p.call(
            "erase_entity",
            {"entity_name": "Boeing", "entity_type": "ORG",
             "tenant": "aerospace", "requested_by": "dpo"},
        ))
        assert isinstance(result, DeniedAction), f"should deny with scopes={scopes}"

    def test_risk_level_readable_from_spec(self):
        """Risk levels are inspectable without calling the tool."""
        p = _make_policy(scopes=["read"])
        risk_map = {name: spec.risk for name, spec in p._tools.items()}
        assert risk_map["local_search"]    == "low"
        assert risk_map["search_graph"]    == "medium"
        assert risk_map["ingest_document"] == "high"
        assert risk_map["erase_entity"]    == "restricted"


# ═══════════════════════════════════════════════════════════════════════════
# 7. Timeout handled
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeoutHandled:
    """A hanging tool must produce a DeniedAction, not propagate CancelledError."""

    def test_slow_tool_produces_denied_action(self):
        p = _make_policy(scopes=["read"])
        result = asyncio.run(p.call("slow_tool", {"question": "q"}))
        assert isinstance(result, DeniedAction)
        assert result.reason == "timeout"
        assert "0.05s" in result.detail

    def test_timeout_does_not_raise(self):
        """The consumer must not receive an exception — only a DeniedAction."""
        p = _make_policy(scopes=["read"])
        raised = False
        try:
            asyncio.run(p.call("slow_tool", {"question": "q"}))
        except Exception:
            raised = True
        assert not raised, "Timeout should not propagate as an exception"

    def test_timeout_outcome_in_audit(self):
        p = _make_policy(scopes=["read"])
        asyncio.run(p.call("slow_tool", {"question": "q"}))
        entry = p.audit_log()[0]
        assert entry.outcome == "timeout"
        assert entry.tool == "slow_tool"


# ═══════════════════════════════════════════════════════════════════════════
# 8. Audit log written
# ═══════════════════════════════════════════════════════════════════════════

class TestAuditLogWritten:
    """Every call (allowed, denied, timeout) must produce an audit entry."""

    def test_executed_call_recorded(self):
        p = _make_policy(scopes=["read"])
        asyncio.run(p.call("local_search", {"question": "test"}))
        entry = p.audit_log()[0]
        assert entry.outcome == "executed"
        assert entry.tool == "local_search"
        assert entry.latency_ms >= 0   # sub-ms in tests; positive in production

    def test_denied_call_recorded(self):
        p = _make_policy(scopes=[])
        asyncio.run(p.call("local_search", {"question": "test"}))
        entry = p.audit_log()[0]
        assert entry.outcome == "denied"
        assert entry.reason == "missing_scope"

    def test_unknown_tool_denial_recorded(self):
        p = _make_policy(scopes=["read", "write", "admin", "gdpr_officer"])
        asyncio.run(p.call("drop_database", {}))
        entry = p.audit_log()[0]
        assert entry.outcome == "denied"
        assert entry.reason == "not_allowed"

    def test_multiple_calls_all_logged(self):
        p = _make_policy(scopes=["read"])
        asyncio.run(p.call("local_search",  {"question": "q1"}))
        asyncio.run(p.call("global_search", {"question": "q2"}))
        asyncio.run(p.call("drop_database", {}))
        assert len(p.audit_log()) == 3
        outcomes = [e.outcome for e in p.audit_log()]
        assert outcomes.count("executed") == 2
        assert outcomes.count("denied") == 1

    def test_audit_summary_counts_by_outcome(self):
        p = _make_policy(scopes=["read"])
        asyncio.run(p.call("local_search",  {"question": "q1"}))
        asyncio.run(p.call("local_search",  {"question": "q2"}))
        asyncio.run(p.call("drop_database", {}))
        s = p.audit_summary()
        assert s["executed"] == 2
        assert s["denied"]   == 1

    def test_audit_entry_contains_tenant(self):
        p = _make_policy(scopes=["read"])
        asyncio.run(p.call("local_search", {"question": "q"}, tenant="aerospace"))
        entry = p.audit_log()[0]
        assert entry.tenant == "aerospace"

    def test_dry_run_still_writes_audit(self):
        p = _make_policy(scopes=["read", "write", "admin", "gdpr_officer"], dry_run=True)
        asyncio.run(p.call("local_search", {"question": "q"}))
        entry = p.audit_log()[0]
        assert entry.outcome == "denied"
        assert entry.reason  == "dry_run"
