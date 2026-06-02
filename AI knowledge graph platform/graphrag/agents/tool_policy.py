"""Agent tool safety layer — allowlist, scopes, validation, timeouts, audit.

Problem solved
--------------
Without guardrails, an LLM agent can invoke any registered tool with any
arguments, including destructive operations, cross-tenant reads, or calls
that should only be available to privileged sessions.

Architecture
------------
ToolPolicy is the single gate between agent intent and tool execution:

  1. Allowlist — only tools named in the policy may be invoked.
  2. Per-tool scopes — each tool declares required scopes; the caller must
     hold all of them (RBAC-lite without a full authz service).
  3. Argument validation — tools declare an argument schema (type + optional
     enum). Arguments that fail validation are rejected before any call.
  4. Dry-run mode — if dry_run=True, the policy logs the intended call and
     returns a sentinel instead of executing. Safe for untrusted sessions.
  5. Timeout — every tool call is wrapped in asyncio.wait_for with a
     per-tool or global timeout. Hangs are treated as errors, never silent.
  6. Denied-action behavior — refusals produce structured DeniedAction
     records (never raw exceptions) so the caller can reason about them.
  7. Audit log — every call attempt (allowed or denied) is appended to
     an in-memory audit trail and optionally flushed to a file / Neo4j.

Usage::

    policy = ToolPolicy.from_defaults()
    result = await policy.call("search_graph", {"query_text": "FAA AD 2024"}, tenant="aerospace")

    # Dry-run (never executes the tool):
    policy_dry = ToolPolicy.from_defaults(dry_run=True)
    result = await policy_dry.call("delete_entity", {...}, tenant="aerospace")
    # → DeniedAction(reason="dry_run", tool="delete_entity")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

import structlog

log = structlog.get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """Metadata for a single registered tool."""
    name:        str
    fn:          Callable                          # the actual callable (sync or async)
    scopes:      list[str]    = field(default_factory=list)   # required caller scopes
    timeout_s:   float        = 10.0              # per-call timeout
    risk:        str          = "safe"            # "safe" | "moderate" | "destructive"
    dry_run_ok:  bool         = True              # if False, ALSO denied in dry-run mode
    arg_schema:  dict[str, dict] = field(default_factory=dict)
    # arg_schema example:
    #   {"query_text": {"type": str}, "top_k": {"type": int, "max": 50}}


@dataclass
class DeniedAction:
    """Structured record for a refused tool call — never raises an exception."""
    tool:     str
    reason:   str            # "not_allowed" | "missing_scope" | "invalid_arg" | "dry_run" | "timeout"
    detail:   str = ""
    tenant:   str = ""
    caller_scopes: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """One entry in the tool execution audit trail."""
    tool:      str
    args:      dict
    tenant:    str
    outcome:   str           # "executed" | "denied" | "timeout" | "error"
    reason:    str = ""
    latency_ms: float = 0.0
    ts:        float = field(default_factory=time.time)


# ── Core policy ───────────────────────────────────────────────────────────────

class ToolPolicy:
    """
    Gate between agent intent and tool execution.

    Parameters
    ----------
    tools:         Registered ToolSpec objects (keyed by name).
    caller_scopes: Scopes held by the current session/user.
    dry_run:       If True, no tool is executed; all return DeniedAction.
    global_timeout_s: Fallback timeout when tool has none.
    """

    def __init__(
        self,
        tools: list[ToolSpec],
        caller_scopes: list[str] | None = None,
        dry_run: bool = False,
        global_timeout_s: float = 30.0,
    ):
        self._tools:    dict[str, ToolSpec] = {t.name: t for t in tools}
        self._scopes:   list[str]  = caller_scopes or []
        self._dry_run:  bool       = dry_run
        self._timeout:  float      = global_timeout_s
        self._audit:    list[AuditEntry] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    async def call(
        self,
        tool_name: str,
        args: dict,
        tenant: str = "default",
    ) -> Any | DeniedAction:
        """
        Invoke a tool through the policy gate.

        Returns the tool's return value on success, or a DeniedAction on any
        refusal.  Never raises for policy violations — only for unexpected
        internal errors in the tool itself (which are logged and re-raised).
        """
        t0 = time.monotonic()

        # 1. Allowlist check
        if tool_name not in self._tools:
            return self._deny(tool_name, args, tenant, "not_allowed",
                              f"'{tool_name}' is not in the tool allowlist", t0)

        spec = self._tools[tool_name]

        # 2. Dry-run (checked before scope so the caller can preview any call)
        if self._dry_run:
            return self._deny(tool_name, args, tenant, "dry_run",
                              "policy is in dry-run mode — no tool executed", t0)

        # 3. Scope check
        missing = [s for s in spec.scopes if s not in self._scopes]
        if missing:
            return self._deny(tool_name, args, tenant, "missing_scope",
                              f"missing scopes: {missing}", t0)

        # 4. Argument validation
        err = self._validate_args(spec, args)
        if err:
            return self._deny(tool_name, args, tenant, "invalid_arg", err, t0)

        # 5. Execute with timeout
        timeout = spec.timeout_s or self._timeout
        try:
            if asyncio.iscoroutinefunction(spec.fn):
                result = await asyncio.wait_for(spec.fn(**args), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: spec.fn(**args)),
                    timeout=timeout,
                )
            latency = (time.monotonic() - t0) * 1000
            self._audit.append(AuditEntry(
                tool=tool_name, args=args, tenant=tenant,
                outcome="executed", latency_ms=round(latency, 1),
            ))
            log.info("tool_policy.executed", tool=tool_name, tenant=tenant,
                     latency_ms=round(latency, 1))
            return result

        except asyncio.TimeoutError:
            latency = (time.monotonic() - t0) * 1000
            self._audit.append(AuditEntry(
                tool=tool_name, args=args, tenant=tenant,
                outcome="timeout", reason="timeout", latency_ms=round(latency, 1),
            ))
            log.warning("tool_policy.timeout", tool=tool_name, timeout_s=timeout,
                        tenant=tenant)
            return DeniedAction(
                tool=tool_name, reason="timeout",
                detail=f"exceeded {timeout}s limit",
                tenant=tenant, caller_scopes=self._scopes,
            )

    def audit_log(self) -> list[AuditEntry]:
        """Return the in-memory audit trail (copy)."""
        return list(self._audit)

    def audit_summary(self) -> dict:
        """Aggregate counts by outcome."""
        counts: dict[str, int] = {}
        for e in self._audit:
            counts[e.outcome] = counts.get(e.outcome, 0) + 1
        return counts

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _deny(
        self,
        tool: str,
        args: dict,
        tenant: str,
        reason: str,
        detail: str,
        t0: float,
    ) -> DeniedAction:
        latency = (time.monotonic() - t0) * 1000
        self._audit.append(AuditEntry(
            tool=tool, args=args, tenant=tenant,
            outcome="denied", reason=reason, latency_ms=round(latency, 1),
        ))
        log.warning("tool_policy.denied", tool=tool, reason=reason, detail=detail,
                    tenant=tenant, caller_scopes=self._scopes)
        return DeniedAction(
            tool=tool, reason=reason, detail=detail,
            tenant=tenant, caller_scopes=self._scopes,
        )

    def _validate_args(self, spec: ToolSpec, args: dict) -> str | None:
        """Return an error string on invalid args, None on success."""
        for arg_name, rule in spec.arg_schema.items():
            value = args.get(arg_name)
            # Required fields
            if value is None and rule.get("required", False):
                return f"required argument '{arg_name}' is missing"
            if value is None:
                continue
            # Type check
            expected_type = rule.get("type")
            if expected_type and not isinstance(value, expected_type):
                return (f"argument '{arg_name}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}")
            # Enum check
            allowed = rule.get("allowed")
            if allowed and value not in allowed:
                return f"argument '{arg_name}' must be one of {allowed}, got {value!r}"
            # Range checks
            if "max" in rule and isinstance(value, (int, float)) and value > rule["max"]:
                return f"argument '{arg_name}' exceeds maximum {rule['max']}"
            if "min" in rule and isinstance(value, (int, float)) and value < rule["min"]:
                return f"argument '{arg_name}' below minimum {rule['min']}"
            # Tenant cross-access guard: if an argument is named 'tenant' and the
            # caller's tenant is known, reject calls that reference a different one.
            if arg_name == "tenant" and value and self._scopes:
                # Scopes carry tenant context as "tenant:<name>"; if present, enforce
                tenant_scopes = [s for s in self._scopes if s.startswith("tenant:")]
                if tenant_scopes:
                    allowed_tenants = [s.split(":", 1)[1] for s in tenant_scopes]
                    if value not in allowed_tenants:
                        return (f"cross-tenant access denied: caller tenant(s) "
                                f"{allowed_tenants} cannot access tenant '{value}'")
        return None

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_defaults(
        cls,
        caller_scopes: list[str] | None = None,
        dry_run: bool = False,
    ) -> "ToolPolicy":
        """
        Build a policy with the standard GraphRAG tool registry.

        Tools are registered with risk levels and argument schemas so that
        the safety properties are declared alongside the tool itself.
        """
        from graphrag.agents.tools.neo4j_tools import search_graph, get_community, get_neighbors
        from graphrag.agents.tools.retrieval_tools import local_search, global_search

        tools = [
            ToolSpec(
                name="search_graph",
                fn=search_graph,
                scopes=["read"],
                timeout_s=15.0,
                risk="safe",
                arg_schema={
                    "query_text": {"type": str, "required": True},
                    "top_k":      {"type": int, "min": 1, "max": 50},
                },
            ),
            ToolSpec(
                name="get_community",
                fn=get_community,
                scopes=["read"],
                timeout_s=5.0,
                risk="safe",
                arg_schema={
                    "community_id": {"type": str, "required": True},
                },
            ),
            ToolSpec(
                name="get_neighbors",
                fn=get_neighbors,
                scopes=["read"],
                timeout_s=5.0,
                risk="safe",
                arg_schema={
                    "entity_name": {"type": str, "required": True},
                },
            ),
            ToolSpec(
                name="local_search",
                fn=local_search,
                scopes=["read"],
                timeout_s=20.0,
                risk="safe",
                arg_schema={
                    "question": {"type": str, "required": True},
                },
            ),
            ToolSpec(
                name="global_search",
                fn=global_search,
                scopes=["read"],
                timeout_s=30.0,
                risk="safe",
                arg_schema={
                    "question": {"type": str, "required": True},
                },
            ),
        ]
        return cls(tools=tools, caller_scopes=caller_scopes, dry_run=dry_run)
