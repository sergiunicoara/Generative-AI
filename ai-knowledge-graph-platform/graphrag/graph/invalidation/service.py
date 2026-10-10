"""Targeted invalidation: event -> dependency closure -> NEEDS_REVIEW + eviction -> recompute.

Soundness rules (docs/invalidation.md):

* Only dependents are touched; unrelated artifacts and cached answers stay valid.
* Whenever targeting cannot be proven complete, fall back to the tenant-wide
  revision bump: dependency walk truncated, cache not shared across processes,
  eviction failed, or an unexpected error. Correctness wins over targeting.
* Additive events (a new or re-enabled fact) can affect answers that cited
  nothing related; their callers keep the revision bump and this service only
  marks the existing dependents.
"""
from __future__ import annotations

import structlog

from graphrag.core.config import get_settings
from graphrag.graph.invalidation import metrics
from graphrag.graph.invalidation.dependency_index import DependencyIndex
from graphrag.graph.invalidation.models import ArtifactKind, InvalidationEvent
from graphrag.graph.invalidation.recompute import RecomputeWorker
from graphrag.graph.invalidation.state_store import StateStore

log = structlog.get_logger(__name__)


def _cfg() -> dict:
    try:
        return get_settings().invalidation or {}
    except Exception:  # noqa: BLE001
        return {}


class InvalidationService:
    def __init__(self, neo4j_client, *, cache_getter=None, recompute_worker: RecomputeWorker | None = None):
        cfg = _cfg()
        self._neo4j = neo4j_client
        self._store = StateStore(neo4j_client)
        self._index = DependencyIndex(neo4j_client, max_depth=int(cfg.get("max_depth", 5)),
                                      cap=int(cfg.get("max_dependents_per_query", 5000)))
        self._cache_getter = cache_getter
        self._worker = recompute_worker or RecomputeWorker(neo4j_client)
        self._recompute_inline = bool(cfg.get("recompute_inline", True))
        self._inline_limit = int(cfg.get("recompute_inline_limit", 200))

    @property
    def store(self) -> StateStore:
        return self._store

    @property
    def worker(self) -> RecomputeWorker:
        return self._worker

    async def _cache(self):
        if self._cache_getter is not None:
            return await self._cache_getter()
        if not get_settings().retrieval.get("semantic_answer_cache_enabled", False):
            return None
        from graphrag.retrieval.query_cache import get_query_cache
        return await get_query_cache()

    async def handle(self, event: InvalidationEvent) -> dict:
        should_process, previous = await self._store.begin_event(event)
        if not should_process:
            metrics.record_event(event.kind.value, "duplicate")
            return {**(previous or {}), "duplicate": True}

        closure = await self._index.resolve(event)
        marked = {
            ArtifactKind.DECISION.value: await self._store.mark(
                event, ArtifactKind.DECISION, sorted(closure.decision_ids)),
            ArtifactKind.COMMUNITY_SNAPSHOT.value: await self._store.mark(
                event, ArtifactKind.COMMUNITY_SNAPSHOT, sorted(closure.snapshot_ids)),
            ArtifactKind.INFERRED_EDGE.value: await self._store.mark(
                event, ArtifactKind.INFERRED_EDGE, sorted(closure.inferred_edges)),
        }
        await self._store.flag_inferred(event.tenant, list(closure.inferred_edges.values()))
        await self._store.flag_snapshots(event.tenant, sorted(closure.snapshot_ids))
        for kind, n in marked.items():
            metrics.record_artifacts(kind, n)

        evicted, fallback = await self._evict(event, closure)
        if fallback and not event.additive:
            await self._neo4j.advance_corpus_revision(event.tenant, reason=f"invalidation_fallback:{fallback}")
        metrics.record_event(event.kind.value, f"fallback_{fallback}" if fallback else "targeted")

        recompute = None
        if self._recompute_inline:
            recompute = await self._worker.run_once(event.tenant, limit=self._inline_limit)

        summary = {
            "event_id": event.id,
            "kind": event.kind.value,
            "affected": closure.counts(),
            "marked": marked,
            "evicted_cached_answers": evicted,
            "fallback": fallback,
            "truncated": closure.truncated,
            "depth": closure.depth_reached,
            "recompute": recompute,
        }
        await self._store.finish_event(event, summary)
        log.info("invalidation.processed", tenant=event.tenant, kind=event.kind.value,
                 event_id=event.id, affected=closure.counts(), evicted=evicted, fallback=fallback)
        return summary

    async def _evict(self, event: InvalidationEvent, closure) -> tuple[int, str | None]:
        if closure.truncated:
            return 0, "dependency_limit"
        if not closure.entities and not closure.chunk_ids:
            return 0, None  # nothing a cached answer could have cited
        try:
            cache = await self._cache()
        except Exception:  # noqa: BLE001
            return 0, "cache_unavailable"
        if cache is None:
            return 0, None  # answer cache disabled: nothing cached to be stale
        if not getattr(cache, "shared", False):
            return 0, "unshared_cache"
        try:
            n = await cache.invalidate_for(
                event.tenant,
                entity_names=sorted({e.name for e in closure.entities.values()}),
                chunk_ids=sorted(closure.chunk_ids),
                raise_errors=True,
            )
        except Exception:  # noqa: BLE001
            return 0, "eviction_failed"
        metrics.record_evicted(n)
        return n, None


async def emit(event: InvalidationEvent, neo4j_client=None, *, inside_mutation: bool = True) -> dict:
    """Process an invalidation event; never raises.

    ``inside_mutation`` says the caller holds a CorpusMutation (cache reads are
    bypassed meanwhile, and the caller decides the revision bump). Outside one,
    an additive event bumps the revision here. Any failure falls back to the
    revision bump, so an error can cost cache hits but never serve stale answers.
    """
    from graphrag.graph.neo4j_client import get_neo4j

    neo4j = neo4j_client or get_neo4j()
    if not _cfg().get("enabled", True):
        await neo4j.advance_corpus_revision(event.tenant, reason=f"invalidation_disabled:{event.kind.value}")
        return {"event_id": event.id, "fallback": "disabled"}
    try:
        summary = await InvalidationService(neo4j).handle(event)
        if event.additive and not inside_mutation:
            await neo4j.advance_corpus_revision(event.tenant, reason=f"additive:{event.kind.value}")
        return summary
    except Exception as exc:  # noqa: BLE001
        log.warning("invalidation.failed", tenant=event.tenant, kind=event.kind.value, error=str(exc)[:200])
        metrics.record_event(event.kind.value, "error")
        try:
            await neo4j.advance_corpus_revision(event.tenant, reason=f"invalidation_error:{event.kind.value}")
        except Exception as bump_exc:  # noqa: BLE001
            log.error("invalidation.fallback_failed", tenant=event.tenant, error=str(bump_exc)[:200])
        return {"event_id": event.id, "fallback": "error", "error": type(exc).__name__}
