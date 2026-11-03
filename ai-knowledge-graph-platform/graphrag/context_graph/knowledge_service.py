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
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import os
from typing import Any

from graphrag.context_graph.knowledge import BusinessContext, Fact, content_digest
from graphrag.context_graph.knowledge_decisions import (
    ReviewImpact, decision_record, expiring_exceptions, replays_exactly, review_impact,
)
from graphrag.context_graph.knowledge_evaluation import EvaluationDecision, evaluate_knowledge
from graphrag.context_graph.knowledge_release import (
    KnowledgePack, RegressionCase, ReleaseCatalog, publish_pack,
)
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
        # seconds per evaluator stage, accumulated over cache misses
        self.stage_seconds: dict[str, float] = {}

    async def catalog(self, tenant: str, domain: str) -> ReleaseCatalog:
        record = await self._store.active_knowledge_release(require_tenant(tenant), domain)
        if record is None:
            raise NoActiveReleaseError(f"no published knowledge release for domain {domain!r}")
        return ReleaseCatalog.from_record(record)

    # -- release administration (callers authorise; the store keeps the history) --

    async def publish(
        self, pack: KnowledgePack, *, published_by: str, cases: Iterable[RegressionCase],
    ) -> dict[str, Any]:
        """Gate and activate a release.  Raises ``KnowledgePublicationError``."""
        return await publish_pack(self._store, pack, published_by=published_by, cases=cases)

    async def release_history(self, tenant: str, domain: str) -> list[dict[str, Any]]:
        return await self._store.knowledge_release_history(require_tenant(tenant), domain)

    async def active_release_id(self, tenant: str, domain: str) -> str | None:
        record = await self._store.active_knowledge_release(require_tenant(tenant), domain)
        return str(record["release_id"]) if record else None

    async def rollback(self, tenant: str, domain: str, release_id: str, *, rolled_back_by: str) -> dict[str, Any]:
        return await self._store.rollback_knowledge_release(
            require_tenant(tenant), domain, release_id, rolled_back_by=rolled_back_by)

    async def evaluate(
        self, *, tenant: str, domain: str, access: AccessContext, context: BusinessContext,
        facts: Mapping[str, Fact] | None = None, expect_applicable_knowledge: bool = False,
        semantic_candidates: list[str] | None = None, record: bool = True,
    ) -> EvaluationDecision:
        """Evaluate against the active release.  With ``record`` (default) the
        decision and its inputs are appended to the tenant's decision log."""
        tenant = require_tenant(tenant)
        catalog = await self.catalog(tenant, domain)
        facts = dict(facts or {})
        key = evaluation_cache_key(
            tenant=tenant, access=access, catalog=catalog, context=context, facts=facts,
            expect_applicable_knowledge=expect_applicable_knowledge, semantic_candidates=semantic_candidates,
        )
        decision = self._cache.get(key)
        if decision is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            decision = evaluate_knowledge(
                items=catalog.items, exceptions=catalog.exceptions, context=context, facts=facts,
                tenant=tenant, access=access, domain=domain, release=catalog.ref,
                expect_applicable_knowledge=expect_applicable_knowledge,
                semantic_candidates=semantic_candidates, timings=self.stage_seconds,
            )
            self._cache[key] = decision
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        if record:
            await self._store.record_knowledge_decision(decision_record(
                decision, access=access, context=context, facts=facts,
                expect_applicable_knowledge=expect_applicable_knowledge,
                semantic_candidates=semantic_candidates,
            ))
        return decision

    def metrics(self) -> dict[str, Any]:
        """Cache hit rate and time per evaluator stage since start (process-local)."""
        lookups = self.cache_hits + self.cache_misses
        return {
            "cache": {"hits": self.cache_hits, "misses": self.cache_misses, "entries": len(self._cache),
                      "hit_rate": round(self.cache_hits / lookups, 4) if lookups else None},
            "evaluations": self.cache_misses,
            "stage_seconds": {stage: round(value, 6) for stage, value in sorted(self.stage_seconds.items())},
            "mean_ms_per_evaluation": (round(sum(self.stage_seconds.values()) * 1000 / self.cache_misses, 3)
                                       if self.cache_misses else None),
        }

    async def stale_projections(self, tenant: str, domain: str, stamps: Mapping[str, str]) -> list[str]:
        """Names of projections whose release content hash is not the active one.

        ``stamps`` maps a projection name (an RDF export, a cached answer, a
        stored decision ...) to the content hash of the release it was made
        from.  A rollback to identical content is current again: identity is
        the content, not the release id.
        """
        active = (await self.catalog(tenant, domain)).ref.content_hash
        return sorted(name for name, content_hash in stamps.items() if content_hash != active)

    # -- decision log ------------------------------------------------------------

    async def decision_log(self, tenant: str, domain: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        return await self._store.knowledge_decisions(require_tenant(tenant), domain, limit=limit)

    async def release_catalog(self, tenant: str, domain: str, release_id: str) -> ReleaseCatalog:
        record = await self._store.knowledge_release(require_tenant(tenant), domain, release_id)
        if record is None:
            raise NoActiveReleaseError(f"no knowledge release {release_id!r} for domain {domain!r}")
        return ReleaseCatalog.from_record(record)

    async def replay(self, tenant: str, domain: str, record_id: str) -> tuple[dict[str, Any], bool]:
        """A stored decision and whether its cited release reproduces it exactly."""
        record = await self._store.knowledge_decision(require_tenant(tenant), record_id)
        if record is None or record["domain"] != domain:
            raise LookupError(f"no decision record {record_id!r}")
        catalog = await self.release_catalog(tenant, domain, str(record["release_id"]))
        return record, replays_exactly(record, catalog)

    async def review_impact(
        self, tenant: str, domain: str, *, as_of: datetime | None = None, window_end: datetime | None = None,
        now: datetime | None = None, limit: int = 1000,
    ) -> list[ReviewImpact]:
        """Stored decisions re-evaluated against the active release."""
        catalog = await self.catalog(tenant, domain)
        now = now or datetime.now(timezone.utc)
        expiring = expiring_exceptions(catalog, start=now, end=window_end) if window_end else []
        records = await self.decision_log(tenant, domain, limit=limit)
        return review_impact(records, catalog, as_of=as_of or window_end, now=now, expiring=expiring)


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
