"""Guardrail tests for the agent tool safety layer.

Every test proves a specific denial behaviour so CI catches regressions
in the policy gate before they reach production.  No live services are
required — all tool functions are replaced with lightweight stubs.
"""

from __future__ import annotations

import asyncio
import pytest

from graphrag.agents.tool_policy import DeniedAction, ToolPolicy, ToolSpec


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _ok_tool(query_text: str = "", top_k: int = 5, tenant: str = "") -> list[dict]:
    return [{"result": "ok", "query": query_text}]


async def _slow_tool(query_text: str = "") -> list[dict]:
    await asyncio.sleep(99)   # will time out
    return []


async def _destructive_tool(entity_name: str = "") -> bool:
    return True   # never actually called — used only as a registered stub


def _policy(
    *,
    scopes: list[str] | None = None,
    dry_run: bool = False,
    extra_tools: list[ToolSpec] | None = None,
) -> ToolPolicy:
    """Build a test policy with a small set of stub tools."""
    tools = [
        ToolSpec(
            name="search_graph",
            fn=_ok_tool,
            scopes=["read"],
            timeout_s=5.0,
            risk="safe",
            arg_schema={
                "query_text": {"type": str, "required": True},
                "top_k":      {"type": int, "min": 1, "max": 50},
            },
        ),
        ToolSpec(
            name="slow_tool",
            fn=_slow_tool,
            scopes=["read"],
            timeout_s=0.05,   # deliberately tiny — triggers timeout in tests
            risk="safe",
            arg_schema={"query_text": {"type": str}},
        ),
        ToolSpec(
            name="delete_entity",
            fn=_destructive_tool,
            scopes=["write", "admin"],
            timeout_s=5.0,
            risk="destructive",
            arg_schema={"entity_name": {"type": str, "required": True}},
        ),
        ToolSpec(
            name="scoped_tool",
            fn=_ok_tool,
            scopes=["premium"],
            timeout_s=5.0,
            risk="safe",
            arg_schema={},
        ),
        *(extra_tools or []),
    ]
    return ToolPolicy(tools=tools, caller_scopes=scopes or [], dry_run=dry_run)


# ── Allowlist ─────────────────────────────────────────────────────────────────

class TestAllowlist:
    def test_unknown_tool_is_denied(self):
        p = _policy(scopes=["read", "write", "admin"])
        result = asyncio.run(p.call("drop_database", {}, tenant="test"))
        assert isinstance(result, DeniedAction)
        assert result.reason == "not_allowed"
        assert result.tool == "drop_database"

    def test_known_tool_with_correct_scope_executes(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"query_text": "FAA AD 2024"}))
        assert isinstance(result, list)
        assert result[0]["result"] == "ok"

    def test_audit_log_records_both_allowed_and_denied(self):
        p = _policy(scopes=["read", "write", "admin"])
        asyncio.run(p.call("search_graph", {"query_text": "test"}))
        asyncio.run(p.call("nonexistent_tool", {}))
        log = p.audit_log()
        assert len(log) == 2
        outcomes = {e.outcome for e in log}
        assert "executed" in outcomes
        assert "denied" in outcomes


# ── Scope enforcement ─────────────────────────────────────────────────────────

class TestScopes:
    def test_missing_required_scope_is_denied(self):
        p = _policy(scopes=["read"])     # missing "premium"
        result = asyncio.run(p.call("scoped_tool", {}))
        assert isinstance(result, DeniedAction)
        assert result.reason == "missing_scope"
        assert "premium" in result.detail

    def test_partial_scopes_denied_for_destructive_tool(self):
        p = _policy(scopes=["write"])    # needs both "write" and "admin"
        result = asyncio.run(p.call("delete_entity", {"entity_name": "Boeing"}))
        assert isinstance(result, DeniedAction)
        assert result.reason == "missing_scope"

    def test_full_scopes_allow_destructive_tool(self):
        p = _policy(scopes=["read", "write", "admin"])
        result = asyncio.run(p.call("delete_entity", {"entity_name": "Boeing"}))
        assert result is True   # stub returns True

    def test_denied_action_carries_caller_scopes(self):
        caller = ["read", "other"]
        p = _policy(scopes=caller)
        result = asyncio.run(p.call("scoped_tool", {}))
        assert isinstance(result, DeniedAction)
        assert result.caller_scopes == caller


# ── Argument validation ───────────────────────────────────────────────────────

class TestArgValidation:
    def test_missing_required_arg_is_denied(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {}))    # query_text missing
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "query_text" in result.detail

    def test_wrong_type_is_denied(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"query_text": 42}))  # int not str
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "str" in result.detail

    def test_value_above_max_is_denied(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"query_text": "q", "top_k": 200}))
        assert isinstance(result, DeniedAction)
        assert "maximum" in result.detail

    def test_value_below_min_is_denied(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"query_text": "q", "top_k": 0}))
        assert isinstance(result, DeniedAction)
        assert "minimum" in result.detail

    def test_valid_args_execute(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("search_graph", {"query_text": "FAA", "top_k": 10}))
        assert isinstance(result, list)


# ── Cross-tenant access ───────────────────────────────────────────────────────

class TestCrossTenant:
    def test_cross_tenant_arg_denied(self):
        """Caller scoped to 'aerospace' cannot access 'banking' tenant."""
        cross_tool = ToolSpec(
            name="tenant_tool",
            fn=_ok_tool,
            scopes=["read"],
            timeout_s=5.0,
            risk="safe",
            arg_schema={"query_text": {"type": str}, "tenant": {"type": str}},
        )
        p = _policy(scopes=["read", "tenant:aerospace"], extra_tools=[cross_tool])
        result = asyncio.run(
            p.call("tenant_tool", {"query_text": "q", "tenant": "banking"})
        )
        assert isinstance(result, DeniedAction)
        assert result.reason == "invalid_arg"
        assert "cross-tenant" in result.detail

    def test_same_tenant_allowed(self):
        cross_tool = ToolSpec(
            name="tenant_tool",
            fn=_ok_tool,
            scopes=["read"],
            timeout_s=5.0,
            risk="safe",
            arg_schema={"query_text": {"type": str}, "tenant": {"type": str}},
        )
        p = _policy(scopes=["read", "tenant:aerospace"], extra_tools=[cross_tool])
        result = asyncio.run(
            p.call("tenant_tool", {"query_text": "q", "tenant": "aerospace"})
        )
        assert isinstance(result, list)


# ── Destructive actions ───────────────────────────────────────────────────────

class TestDestructiveActions:
    def test_destructive_tool_requires_all_scopes(self):
        for scopes in [[], ["read"], ["write"], ["admin"], ["read", "write"]]:
            p = _policy(scopes=scopes)
            result = asyncio.run(
                p.call("delete_entity", {"entity_name": "Boeing"})
            )
            assert isinstance(result, DeniedAction), (
                f"should be denied with scopes={scopes}"
            )

    def test_destructive_tool_allowed_with_full_scopes(self):
        p = _policy(scopes=["read", "write", "admin"])
        result = asyncio.run(p.call("delete_entity", {"entity_name": "Boeing"}))
        assert result is True


# ── Dry-run mode ──────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_denies_every_tool(self):
        p = _policy(scopes=["read", "write", "admin"], dry_run=True)
        for tool in ["search_graph", "delete_entity", "scoped_tool"]:
            args = {"query_text": "q", "entity_name": "Boeing"}
            result = asyncio.run(p.call(tool, args))
            assert isinstance(result, DeniedAction), f"{tool} should be denied in dry-run"
            assert result.reason == "dry_run"

    def test_dry_run_audit_log_records_denials(self):
        p = _policy(scopes=["read", "write", "admin"], dry_run=True)
        asyncio.run(p.call("search_graph", {"query_text": "test"}))
        asyncio.run(p.call("delete_entity", {"entity_name": "Boeing"}))
        assert all(e.outcome == "denied" for e in p.audit_log())
        assert all(e.reason == "dry_run" for e in p.audit_log())


# ── Timeout handling ──────────────────────────────────────────────────────────

class TestTimeout:
    def test_slow_tool_times_out(self):
        p = _policy(scopes=["read"])
        result = asyncio.run(p.call("slow_tool", {"query_text": "q"}))
        assert isinstance(result, DeniedAction)
        assert result.reason == "timeout"
        assert "0.05s" in result.detail

    def test_timeout_recorded_in_audit(self):
        p = _policy(scopes=["read"])
        asyncio.run(p.call("slow_tool", {"query_text": "q"}))
        entry = p.audit_log()[0]
        assert entry.outcome == "timeout"


# ── Audit summary ─────────────────────────────────────────────────────────────

class TestAuditSummary:
    def test_summary_counts_correctly(self):
        p = _policy(scopes=["read"])
        asyncio.run(p.call("search_graph", {"query_text": "a"}))
        asyncio.run(p.call("search_graph", {"query_text": "b"}))
        asyncio.run(p.call("nonexistent", {}))
        summary = p.audit_summary()
        assert summary["executed"] == 2
        assert summary["denied"] == 1
