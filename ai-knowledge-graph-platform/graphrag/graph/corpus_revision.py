"""Fail-closed publication of retrieval-visible graph mutations."""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


class CorpusMutation:
    """Bracket a multi-step graph mutation so answer-cache reads stay safe.

    The graph has several mutation services with independent Cypher writes, so
    they cannot all share one database transaction.  This guard keeps cache
    reads disabled until the operation finishes and advances the durable
    revision on both success and exception paths.  A failed finalisation leaves
    the tenant marked updating, which is intentionally fail-closed.
    """

    def __init__(self, neo4j_client, tenant: str, reason: str, *, advance_revision: bool = True):
        self._neo4j = neo4j_client
        self.tenant = tenant
        self.reason = reason
        # False only for removals/corrections whose dependents are invalidated
        # explicitly (graphrag/graph/invalidation); see complete_corpus_update.
        self.advance_revision = advance_revision
        self.revision: int | None = None

    async def __aenter__(self) -> "CorpusMutation":
        await self._neo4j.begin_corpus_update(self.tenant, reason=self.reason)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        outcome = "completed" if exc_type is None else "failed"
        try:
            self.revision = await self._neo4j.complete_corpus_update(
                self.tenant,
                reason=self.reason,
                outcome=outcome,
                **({} if self.advance_revision else {"advance_revision": False}),
            )
        except Exception:
            log.exception(
                "corpus_revision.finalize_failed",
                tenant=self.tenant,
                reason=self.reason,
                outcome=outcome,
            )
            if exc_type is None:
                raise
        return False
