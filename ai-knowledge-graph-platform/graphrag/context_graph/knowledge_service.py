"""Serving contextual-knowledge decisions from the active release.

Used by the MCP capability ``kg.knowledge.evaluate`` and the
``POST /knowledge/evaluate`` route, so consumer applications never
re-implement applicability, conflict resolution or escalation.

Caching: a decision is a pure function of (tenant, caller entitlements,
domain, release identity, business context incl. temporal view, facts,
semantic candidate order, decision schema).  Every one of those is in the key,
so a cached decision can never cross tenants, access policies, business
contexts, temporal views or knowledge releases.  Publishing or rolling back a
release changes the active release id and therefore the key; the old entries
simply stop being addressed.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import os
from typing import Any

from graphrag.context_graph.knowledge import BusinessContext, Fact, content_digest
from graphrag.context_graph.knowledge_evaluation import EvaluationDecision, evaluate_knowledge
from graphrag.context_graph.knowledge_release import ReleaseCatalog
from graphrag.core.tenancy import require_tenant
from graphrag.enterprise.models import AccessContext

DECISION_CACHE_SCHEMA = "knowledge-decision-cache/v1"


class NoActiveReleaseError(LookupError):
    """No published knowledge release exists for this tenant and domain."""


def evaluation_cache_key(
    *, tenant: str, access: AccessContext, catalog: ReleaseCatalog, context: BusinessContext,
    facts: Mapping[str, Fact], expect_applicable_knowledge: bool, semantic_candidates: list[str] | None,
) -> str:
    return content_digest({
        "schema": DECISION_CACHE_SCHEMA,
        "tenant": tenant,
        "access": access.fingerprint,
        "domain": catalog.ref.domain,
        "release": [catalog.ref.release_id, catalog.ref.content_hash],
        "context": context.model_dump(mode="json"),
        "facts": {k: v.model_dump(mode="json") for k, v in sorted(facts.items())},
        "expect": expect_applicable_knowledge,
        "semantic": list(semantic_candidates or []),
    })


class KnowledgeEvaluationService:
    def __init__(self, store, *, cache_size: int = 512) -> None:
        self._store = store
        self._cache: OrderedDict[str, EvaluationDecision] = OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def store(self):
        """The governance store releases are served from (shared with curation)."""
        return self._store

    async def catalog(self, tenant: str, domain: str) -> ReleaseCatalog:
        record = await self._store.active_knowledge_release(require_tenant(tenant), domain)
        if record is None:
            raise NoActiveReleaseError(f"no published knowledge release for domain {domain!r}")
        return ReleaseCatalog.from_record(record)

    async def evaluate(
        self, *, tenant: str, domain: str, access: AccessContext, context: BusinessContext,
        facts: Mapping[str, Fact] | None = None, expect_applicable_knowledge: bool = False,
        semantic_candidates: list[str] | None = None,
    ) -> EvaluationDecision:
        tenant = require_tenant(tenant)
        catalog = await self.catalog(tenant, domain)
        facts = dict(facts or {})
        key = evaluation_cache_key(
            tenant=tenant, access=access, catalog=catalog, context=context, facts=facts,
            expect_applicable_knowledge=expect_applicable_knowledge, semantic_candidates=semantic_candidates,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        decision = evaluate_knowledge(
            items=catalog.items, exceptions=catalog.exceptions, context=context, facts=facts,
            tenant=tenant, access=access, domain=domain, release=catalog.ref,
            expect_applicable_knowledge=expect_applicable_knowledge,
            semantic_candidates=semantic_candidates,
        )
        self._cache[key] = decision
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return decision


_service: KnowledgeEvaluationService | None = None


async def get_knowledge_service() -> KnowledgeEvaluationService:
    """Process-wide service over the durable governance store.

    The store location follows ``GovernanceStore`` configuration
    (``ENERGY_GOVERNANCE_DB_URL``); ``KNOWLEDGE_GOVERNANCE_DB_URL`` overrides it
    for knowledge releases only.
    """
    global _service
    if _service is None:
        from graphrag.domains.energy.governance_store import GovernanceStore

        store = GovernanceStore(os.getenv("KNOWLEDGE_GOVERNANCE_DB_URL") or None)
        await store.open()
        _service = KnowledgeEvaluationService(store)
    return _service


def set_knowledge_service(service: KnowledgeEvaluationService | None) -> None:
    """Test hook and app-lifespan hook."""
    global _service
    _service = service


def decision_payload(decision: EvaluationDecision) -> dict[str, Any]:
    return decision.model_dump(mode="json")


__all__ = [
    "KnowledgeEvaluationService", "NoActiveReleaseError", "decision_payload", "evaluation_cache_key",
    "get_knowledge_service", "set_knowledge_service",
]
