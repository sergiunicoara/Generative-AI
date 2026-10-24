"""``kg.knowledge.evaluate@1.0.0`` -- contextual knowledge evaluation.

Given a business context (customer, brand, market, product, product family,
workflow, as_of/known_as) and evidenced facts, returns which published
requirements of the domain's active knowledge release apply, how each was
validated, unresolved conflicts, authorised exceptions used, missing
evidence, and whether human review is required -- so consumers never
re-implement applicability or conflict resolution.

Trust boundaries:

* the tenant is the caller's identity tenant (the registry injects it and
  rejects a different caller-supplied tenant);
* entitlements come from the identity (subject and resolved groups); the
  business context in the request can narrow what applies but never widens
  what the caller may read.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from graphrag.context_graph.knowledge import BusinessContext, Fact
from graphrag.context_graph.knowledge_service import (
    NoActiveReleaseError, decision_payload, get_knowledge_service,
)
from graphrag.enterprise.models import AccessContext
from mcp_server.registry import CapabilityRegistry, CapabilitySpec

CAPABILITY_ID = "kg.knowledge.evaluate"
CAPABILITY_VERSION = "1.0.0"


def _error(code: str, detail: Any) -> dict[str, Any]:
    return {"error": {"code": code, "detail": detail}}


def access_from_identity(identity) -> AccessContext:
    claims: dict[str, Any] = {"sub": getattr(identity, "subject", "")}
    if getattr(identity, "groups_resolved", False):
        claims["groups"] = list(getattr(identity, "groups", ()))
    return AccessContext.from_claims(claims)


async def evaluate_knowledge_capability(
    tenant: str, domain: str, context: dict, facts: dict | None = None,
    semantic_candidates: list | None = None, expect_applicable_knowledge: bool = False,
    identity=None,
) -> dict[str, Any]:
    try:
        business_context = BusinessContext.model_validate(context)
        parsed_facts = {str(name): Fact.model_validate(value) for name, value in (facts or {}).items()}
    except ValidationError as exc:
        return _error("invalid_request", [
            {"loc": list(err["loc"]), "msg": err["msg"]} for err in exc.errors()[:20]
        ])
    if len(parsed_facts) > 200:
        return _error("invalid_request", "at most 200 facts per evaluation")
    if semantic_candidates is not None and (
        len(semantic_candidates) > 200 or not all(isinstance(c, str) for c in semantic_candidates)
    ):
        return _error("invalid_request", "semantic_candidates must be at most 200 item ids")
    service = await get_knowledge_service()
    try:
        decision = await service.evaluate(
            tenant=tenant, domain=domain, access=access_from_identity(identity),
            context=business_context, facts=parsed_facts,
            expect_applicable_knowledge=bool(expect_applicable_knowledge),
            semantic_candidates=list(semantic_candidates) if semantic_candidates else None,
        )
    except NoActiveReleaseError as exc:
        return _error("no_published_release", str(exc))
    return decision_payload(decision)


def register(registry: CapabilityRegistry) -> None:
    registry.register(CapabilitySpec(
        capability_id=CAPABILITY_ID, version=CAPABILITY_VERSION,
        title="Evaluate which contextual knowledge applies and whether it is satisfied",
        kind="read", risk="moderate", fn=evaluate_knowledge_capability,
        required_scopes=("read",), pass_identity=True,
        arg_schema={
            "tenant": {"type": str},
            "domain": {"type": str, "required": True, "pattern": r"^[a-z][a-z0-9_]{0,63}$"},
            "context": {"type": dict, "required": True},
            "facts": {"type": dict},
            "semantic_candidates": {"type": list},
            "expect_applicable_knowledge": {"type": bool},
        },
    ))


__all__ = ["CAPABILITY_ID", "CAPABILITY_VERSION", "access_from_identity", "evaluate_knowledge_capability",
           "register"]
