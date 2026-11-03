"""Data minimisation before text leaves for an LLM or embedding provider.

Every prompt and embedding input that goes through the shared clients passes
this guard first: it is applied in ``FallbackLLM.generate`` (what ``get_llm``
and ``get_fast_llm`` return) and ``OpenAIEmbedder.embed`` (``get_embedder``):

* a per-tenant policy (``config/outbound_policy.yml``) can **disable external
  providers** for a tenant; a disabled provider is never called and the
  caller gets :class:`OutboundDisabledError`;
* direct identifiers are **redacted by pattern** (e-mail, phone, SSN, IP
  addresses by default; the classes are configurable per tenant);
* structured records sent by callers can be cut to a per-tenant **field
  allowlist** (:func:`minimise_fields`);
* what was sent is **audited as hash and size only**, never the content.

The tenant comes from the request's tenant context
(``graphrag.observability.correlation.tenant_context``); without one, the
default policy applies.  Pattern-based redaction is incomplete by nature:
names, free-text descriptions and identifiers without a recognisable shape are
not caught.  It reduces what leaves; it does not make text anonymous.  Clients
constructed directly (a bare provider client, or the optional RAGAS
evaluator) are not covered.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "outbound_policy.yml"
DEFAULT_CLASSES = ("EMAIL", "PHONE", "PHONE_INTL", "SSN", "IP_V4", "IP_V6")
_AUDIT: deque[dict[str, Any]] = deque(maxlen=1000)


class OutboundDisabledError(RuntimeError):
    """External model providers are disabled for this tenant."""


@dataclass(frozen=True)
class OutboundPolicy:
    providers_enabled: bool = True
    redact: bool = True
    redact_classes: tuple[str, ...] = DEFAULT_CLASSES
    allowed_fields: tuple[str, ...] | None = None


@dataclass(frozen=True)
class OutboundAudit:
    tenant: str
    purpose: str
    sha256: str
    chars: int
    redactions: dict[str, int] = field(default_factory=dict)


_override: dict[str, OutboundPolicy] | None = None


def _policy_from(raw: Mapping[str, Any] | None, base: OutboundPolicy) -> OutboundPolicy:
    raw = dict(raw or {})
    fields = raw.get("allowed_fields", base.allowed_fields)
    return OutboundPolicy(
        providers_enabled=bool(raw.get("providers_enabled", base.providers_enabled)),
        redact=bool(raw.get("redact", base.redact)),
        redact_classes=tuple(raw.get("redact_classes", base.redact_classes)),
        allowed_fields=tuple(fields) if fields is not None else None,
    )


@lru_cache(maxsize=1)
def _load(path: str) -> dict[str, OutboundPolicy]:
    file = Path(path)
    raw = yaml.safe_load(file.read_text(encoding="utf-8")) if file.is_file() else {}
    default = _policy_from((raw or {}).get("default"), OutboundPolicy())
    policies = {"": default}
    for tenant, spec in ((raw or {}).get("tenants") or {}).items():
        policies[str(tenant)] = _policy_from(spec, default)
    return policies


def set_outbound_policies(policies: Mapping[str, OutboundPolicy] | None) -> None:
    """Test and deployment hook: replace the file-based policies (None resets)."""
    global _override
    _override = dict(policies) if policies is not None else None


def policy_for(tenant: str) -> OutboundPolicy:
    policies = _override if _override is not None else _load(str(POLICY_PATH))
    return policies.get(tenant) or policies.get("") or OutboundPolicy()


def _redact(text: str, classes: tuple[str, ...]) -> tuple[str, dict[str, int]]:
    from graphrag.graph.pii_guard import PIIGuard

    findings = [f for f in PIIGuard(None, min_confidence=0.0).scan_text(text) if f.pii_class in classes]
    counts: dict[str, int] = {}
    for finding in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[:finding.start] + f"[{finding.pii_class}_REDACTED]" + text[finding.end:]
        counts[finding.pii_class] = counts.get(finding.pii_class, 0) + 1
    return text, counts


def minimise(text: str, *, tenant: str = "", purpose: str = "llm") -> tuple[str, OutboundAudit]:
    """The text that may leave for ``tenant``, and its hash-and-size audit."""
    if not tenant:
        from graphrag.observability.correlation import current_tenant

        tenant = current_tenant()
    policy = policy_for(tenant)
    if not policy.providers_enabled:
        log.warning("outbound.blocked", tenant=tenant, purpose=purpose)
        raise OutboundDisabledError(f"external model providers are disabled for tenant {tenant!r}")
    redactions: dict[str, int] = {}
    if policy.redact:
        text, redactions = _redact(text, policy.redact_classes)
    audit = OutboundAudit(tenant=tenant, purpose=purpose, chars=len(text),
                          sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), redactions=redactions)
    _AUDIT.append(asdict(audit))
    log.info("outbound.sent", **asdict(audit))
    return text, audit


def minimise_fields(record: Mapping[str, Any], *, tenant: str = "") -> dict[str, Any]:
    """Cut a structured record to the tenant's field allowlist (all fields when unset)."""
    if not tenant:
        from graphrag.observability.correlation import current_tenant

        tenant = current_tenant()
    allowed = policy_for(tenant).allowed_fields
    return dict(record) if allowed is None else {k: v for k, v in record.items() if k in allowed}


def recent_outbound_audits() -> list[dict[str, Any]]:
    """Recent audit records (hash, size, redaction counts; never content)."""
    return list(_AUDIT)


__all__ = [
    "DEFAULT_CLASSES", "OutboundAudit", "OutboundDisabledError", "OutboundPolicy",
    "minimise", "minimise_fields", "policy_for", "recent_outbound_audits", "set_outbound_policies",
]
