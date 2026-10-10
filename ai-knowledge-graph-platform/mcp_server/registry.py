"""Versioned MCP capability registry.

A `CapabilitySpec` is the MCP-facing analogue of `graphrag.agents.tool_policy.ToolSpec`
-- reuses the exact same argument-schema shape and `validate_args()` function
(no second validation implementation), but adds what an agent-internal tool
never needed: a dotted stable id, a semver version, deprecation/replacement
fields, and legacy-name aliases so an existing wire registration keeps
working across a breaking internal rename.

`CapabilityRegistry.discover()` is entitlement-filtered -- a caller without
`biz:write` never sees that a write capability exists at all, not just that
it's denied. `contract_snapshot()` is entitlement-*independent* (the full
registry contents) -- it exists purely so a golden-file test can catch a
breaking change to the registry shape before it ships.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import structlog

from graphrag.agents.tool_policy import validate_args
from graphrag.graph.execution_scope import ResultTooLarge, execution_scope
from graphrag.observability.agent_telemetry import (
    record_capability_call, record_operational_write_receipt,
)

log = structlog.get_logger(__name__)

# Defaults for every guarded operation (docs/mcp-security.md).
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ROWS = 1000
DEFAULT_MAX_RESULT_BYTES = 1_000_000

ApprovalHook = Callable[["CapabilitySpec", dict, Any], Awaitable[bool]]
AuditSink = Callable[[dict], Awaitable[None]]


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str  # dotted stable id, e.g. "kg.graph.stats"
    version: str  # semver, e.g. "1.0.0"
    title: str
    kind: str  # "read" | "write"
    risk: str  # "safe" | "moderate" | "destructive"
    fn: Callable
    required_scopes: tuple[str, ...] = ()
    arg_schema: dict[str, dict] = field(default_factory=dict)
    dry_run_ok: bool = True
    requires_approval: bool = False
    deprecated: bool = False
    replacement: str | None = None
    legacy_aliases: tuple[str, ...] = ()
    # Read capabilities only ever need `tenant` (always injected below); a
    # write capability that builds a CommandEnvelope also needs the actor's
    # identity (subject, token type) to bind `actor_id`/`actor_type` from
    # the token, never from caller-supplied args. Opt-in and narrow rather
    # than passing identity to every capability, so existing read `fn`
    # signatures (e.g. `_graph_stats(tenant)`) need no change.
    pass_identity: bool = False
    # ── Phase 7 guard rails (not part of the wire contract snapshot) ─────────
    # Read capabilities run in a READ-access session unless they record
    # derived artifacts (answer traces, caches) and opt out explicitly.
    read_only_session: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_rows: int = DEFAULT_MAX_ROWS
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    # A write must be approved: either the service it calls enforces approval
    # itself (named here) or the registry's approval hook must allow the call.
    approval_enforced_by: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.capability_id}@{self.version}"


@dataclass(frozen=True)
class DeniedCapabilityCall:
    """Structured refusal -- capability calls never raise for policy reasons."""

    capability: str
    reason: str  # "not_found" | "missing_scope" | "unauthenticated" | "tenant_required" |
    #               "tenant_mismatch" | "invalid_arg" | "dry_run" | "approval_required" |
    #               "approval_denied" | "timeout" | "result_too_large"
    detail: str = ""
    operation_id: str = ""


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class CapabilityRegistry:
    """Entitlement-aware, versioned lookup and invocation for capabilities."""

    def __init__(self, *, approval_hook: ApprovalHook | None = None,
                 audit_sink: AuditSink | None = None) -> None:
        self._by_qualified: dict[str, CapabilitySpec] = {}
        self._by_capability_id: dict[str, list[CapabilitySpec]] = {}
        self._by_alias: dict[str, CapabilitySpec] = {}
        self._approval_hook = approval_hook
        self._audit_sink = audit_sink

    def register(self, spec: CapabilitySpec) -> None:
        if spec.qualified_name in self._by_qualified:
            raise ValueError(f"capability {spec.qualified_name!r} already registered")
        for alias in spec.legacy_aliases:
            if alias in self._by_alias:
                raise ValueError(f"legacy alias {alias!r} already registered")
        self._by_qualified[spec.qualified_name] = spec
        self._by_capability_id.setdefault(spec.capability_id, []).append(spec)
        for alias in spec.legacy_aliases:
            self._by_alias[alias] = spec

    def _find(self, name: str) -> CapabilitySpec | None:
        if name in self._by_qualified:
            return self._by_qualified[name]
        if name in self._by_alias:
            return self._by_alias[name]
        specs = self._by_capability_id.get(name)
        if not specs:
            return None
        return sorted(specs, key=lambda s: _version_key(s.version))[-1]

    def resolve(self, name: str, identity) -> CapabilitySpec | DeniedCapabilityCall:
        """Look up a capability by qualified name, bare id, or legacy alias.

        Existence is checked before entitlement (a caller who names a real
        capability they lack scope for gets `missing_scope`, not
        `not_found`) -- that distinction matters for `call()`'s error
        surface, even though `discover()` hides ungranted capabilities from
        a listing entirely.
        """
        spec = self._find(name)
        if spec is None:
            return DeniedCapabilityCall(
                capability=name, reason="not_found",
                detail=f"no capability registered as {name!r}",
            )
        missing = [s for s in spec.required_scopes if not identity.has_scope(s)]
        if missing:
            return DeniedCapabilityCall(
                capability=spec.qualified_name, reason="missing_scope",
                detail=f"missing scopes: {missing}",
            )
        return spec

    async def call(
        self, name: str, args: dict, identity, *, dry_run: bool = False,
    ) -> Any | DeniedCapabilityCall:
        """Resolve, authenticate, authorize, validate, approve and invoke a capability.

        Never raises for a policy refusal -- every denial path returns a
        `DeniedCapabilityCall` the same way `ToolPolicy.call()` returns a
        `DeniedAction`, so callers (the MCP tool wrappers) can serialize it
        directly instead of branching on exception types.

        Guard rails (docs/mcp-security.md): mandatory identity-bound tenant;
        undeclared arguments rejected; writes need approval (service-enforced or
        the approval hook); read capabilities execute in a READ-access session;
        server-side timeout, row limit and result-size limit; every call gets an
        ``operation_id``, an audit record and, for dict results, a provenance
        receipt.
        """
        started_at = time.monotonic()
        operation_id = str(uuid.uuid4())
        # Bounded metric label: a fabricated name must not create a new series.
        capability = "unknown"
        outcome = "error"
        spec: CapabilitySpec | None = None
        result: Any = None

        def deny(reason: str, detail: str = "") -> DeniedCapabilityCall:
            nonlocal outcome
            outcome = reason
            return DeniedCapabilityCall(capability=spec.qualified_name if spec else name,
                                        reason=reason, detail=detail, operation_id=operation_id)

        try:
            resolved = self.resolve(name, identity)
            if isinstance(resolved, DeniedCapabilityCall):
                outcome = resolved.reason
                if resolved.reason != "not_found":
                    capability = resolved.capability
                result = DeniedCapabilityCall(capability=resolved.capability, reason=resolved.reason,
                                              detail=resolved.detail, operation_id=operation_id)
                return result
            spec = resolved
            capability = spec.qualified_name

            if not identity.authenticated:
                result = deny("unauthenticated",
                              "no valid caller identity — set GRAPHRAG_MCP_TOKEN to a scoped token")
                return result
            if not getattr(identity, "tenant", ""):
                result = deny("tenant_required", "the caller identity is not bound to a tenant")
                return result

            # `tenant` in caller-supplied args is an *assertion*, never an
            # authority: it must match the identity-bound tenant or the call is
            # denied outright, before argument validation even runs.
            caller_tenant = args.get("tenant")
            if caller_tenant and caller_tenant != identity.tenant:
                result = deny("tenant_mismatch", (
                    f"caller is bound to tenant {identity.tenant!r}, "
                    f"cannot act on tenant {caller_tenant!r}"))
                return result

            # The tenant is identity-bound below, so validate without it: a
            # caller never needs (or gets) a say in which tenant is used.
            err = validate_args(spec.arg_schema, {k: v for k, v in args.items() if k != "tenant"},
                                list(identity.scopes))
            if err:
                result = deny("invalid_arg", err)
                return result

            if dry_run:
                if not spec.dry_run_ok:
                    result = deny("dry_run_not_allowed", "this capability cannot be safely previewed")
                    return result
                result = deny("dry_run", "dry-run — capability not executed")
                return result

            if spec.kind != "read" and not spec.approval_enforced_by:
                if self._approval_hook is None:
                    result = deny("approval_required",
                                  "mutating capability without an approval path is not executable")
                    return result
                if not await self._approval_hook(spec, dict(args), identity):
                    result = deny("approval_denied", "the approval hook rejected this call")
                    return result

            call_args = dict(args)
            call_args["tenant"] = identity.tenant  # always identity-bound, never caller-supplied
            if spec.pass_identity:
                call_args["identity"] = identity
            read_only = spec.kind == "read" and spec.read_only_session
            try:
                with execution_scope(read_only=read_only, timeout_s=spec.timeout_s,
                                     max_rows=spec.max_rows, operation=spec.qualified_name):
                    if asyncio.iscoroutinefunction(spec.fn):
                        coro = spec.fn(**call_args)
                    else:
                        ctx = contextvars.copy_context()
                        coro = asyncio.get_event_loop().run_in_executor(
                            None, lambda: ctx.run(spec.fn, **call_args))
                    result = await asyncio.wait_for(coro, timeout=spec.timeout_s)
            except asyncio.TimeoutError:
                result = deny("timeout", f"exceeded {spec.timeout_s}s")
                return result
            except ResultTooLarge as exc:
                result = deny("result_too_large", str(exc))
                return result

            encoded = json.dumps(result, default=str, sort_keys=True)
            if len(encoded.encode("utf-8")) > spec.max_result_bytes:
                result = deny("result_too_large", f"result exceeds {spec.max_result_bytes} bytes")
                return result
            # Governed write adapters return a CommandReceipt as a dict. Keep
            # the MCP-call metric and the write-outcome metric semantically
            # useful by exposing approval/stale/dry-run/denied separately.
            receipt_outcome = result.get("outcome") if isinstance(result, dict) else None
            outcome = str(receipt_outcome or "executed")
            if spec.kind == "write" and receipt_outcome:
                record_operational_write_receipt(
                    capability=spec.qualified_name, outcome=outcome,
                    tenant=identity.tenant,
                )
            if isinstance(result, dict) and "provenance_receipt" not in result:
                result = {**result, "provenance_receipt": self._receipt(
                    spec, operation_id, identity, args, encoded, read_only)}
            return result
        finally:
            record_capability_call(
                capability=capability,
                outcome=outcome,
                tenant=getattr(identity, "tenant", "") or "anonymous",
                started_at=started_at,
            )
            await self._audit(operation_id, name, spec, identity, args, outcome, started_at)

    @staticmethod
    def _digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, default=str, sort_keys=True).encode("utf-8")).hexdigest()

    def _receipt(self, spec: CapabilitySpec, operation_id: str, identity, args: dict, encoded: str,
                 read_only: bool) -> dict:
        return {
            "operation_id": operation_id,
            "operation": spec.qualified_name,
            "kind": spec.kind,
            "read_only_session": read_only,
            "tenant": identity.tenant,
            "subject": getattr(identity, "subject", ""),
            "args_sha256": self._digest({k: v for k, v in args.items() if k != "tenant"}),
            "result_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "executed_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _audit(self, operation_id: str, requested: str, spec: CapabilitySpec | None, identity,
                     args: dict, outcome: str, started_at: float) -> None:
        event = {
            "operation_id": operation_id,
            "operation": spec.qualified_name if spec else None,
            "requested_name": requested[:200],
            "kind": spec.kind if spec else None,
            "tenant": getattr(identity, "tenant", "") or "",
            "subject": getattr(identity, "subject", "") or "",
            "outcome": outcome,
            "args_sha256": self._digest({k: v for k, v in (args or {}).items() if k != "tenant"}),
            "duration_ms": round((time.monotonic() - started_at) * 1000, 1),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info("capability.audit", **{k: v for k, v in event.items() if k != "args_sha256"})
        if self._audit_sink is not None:
            try:
                await self._audit_sink(event)
            except Exception as exc:  # noqa: BLE001 - audit persistence must not mask the result
                log.error("capability.audit_persist_failed", operation_id=operation_id, error=str(exc)[:200])

    def discover(self, identity) -> list[dict]:
        """Entitlement-filtered listing: a capability the caller lacks scope
        for is omitted entirely, not shown-then-denied."""
        return [
            self._describe(spec)
            for spec in self._by_qualified.values()
            if all(identity.has_scope(s) for s in spec.required_scopes)
        ]

    def contract_snapshot(self) -> list[dict]:
        """Full, entitlement-independent registry contents for the golden
        compatibility-test fixture (`test_mcp_contract_compat.py`)."""
        return [
            self._describe(spec)
            for spec in sorted(self._by_qualified.values(), key=lambda s: s.qualified_name)
        ]

    @staticmethod
    def _describe(spec: CapabilitySpec) -> dict:
        return {
            "capability_id": spec.capability_id,
            "version": spec.version,
            "qualified_name": spec.qualified_name,
            "title": spec.title,
            "kind": spec.kind,
            "risk": spec.risk,
            "required_scopes": list(spec.required_scopes),
            "arg_schema_keys": sorted(spec.arg_schema.keys()),
            "dry_run_ok": spec.dry_run_ok,
            "requires_approval": spec.requires_approval,
            "deprecated": spec.deprecated,
            "replacement": spec.replacement,
            "legacy_aliases": list(spec.legacy_aliases),
        }
