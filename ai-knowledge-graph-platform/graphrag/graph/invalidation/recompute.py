"""Targeted recomputation of artifacts in NEEDS_REVIEW.

Deterministic per kind; nothing here calls an LLM:

* ``inferred_edge``: the stored premises are re-checked; if any is retracted,
  expired, missing or touches a quarantined entity, the rule is re-run for that
  one (src, tgt) pair. Re-derived -> VALID with the new premises. Not derivable
  -> the edge is retracted (kept, ``confidence_state = RETRACTED``) and the
  artifact ends INSUFFICIENT_EVIDENCE.
* ``community_snapshot``: if all of its evidence (chunks, documents, member
  entities) is still present, current and unquarantined -> VALID. Otherwise the
  snapshot is closed (``transaction_to``; its text is kept as history) ->
  INSUFFICIENT_EVIDENCE, and the next community rebuild writes a new one.
* ``decision``: a recorded answer cannot be re-derived without re-running the
  model. If none of its evidence survives -> INSUFFICIENT_EVIDENCE; otherwise it
  goes back to NEEDS_REVIEW flagged ``requires_human_review`` (an optional
  ``answer_recomputer`` callable can re-answer and return VALID /
  VALIDATION_FAILED instead).

Unexpected errors -> RECOMPUTE_FAILED; a later invalidation re-queues it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog

from graphrag.graph.invalidation import metrics
from graphrag.graph.invalidation.models import ArtifactKind, ArtifactState
from graphrag.graph.invalidation.state_store import StateStore
from graphrag.graph.validation.graph_checks import run_read_only

log = structlog.get_logger(__name__)

AnswerRecomputer = Callable[[str, str], Awaitable[ArtifactState]]

_EDGE = """
MATCH (s:Entity {tenant: $tenant, name: $src_name, type: $src_type})
      -[r:RELATES_TO {relation: $relation}]->
      (o:Entity {tenant: $tenant, name: $tgt_name, type: $tgt_type})
WHERE r.tenant = $tenant
RETURN r.source_type AS source_type, r.inferred_by AS rule, coalesce(r.premise_keys, []) AS premises,
       r.confidence_state AS state
"""

_PREMISES_OK = """
UNWIND $premises AS p
WITH p, split(p, '|') AS parts
WITH p, parts[1] AS rel,
     split(parts[0], ':')[0] AS st, substring(parts[0], size(split(parts[0], ':')[0]) + 1) AS sn,
     split(parts[2], ':')[0] AS tt, substring(parts[2], size(split(parts[2], ':')[0]) + 1) AS tn
OPTIONAL MATCH (s:Entity {tenant: $tenant, name: sn, type: st})-[r:RELATES_TO {relation: rel}]->
               (o:Entity {tenant: $tenant, name: tn, type: tt})
WHERE r.tenant = $tenant
  AND coalesce(r.confidence_state, 'ASSERTED') <> 'RETRACTED'
  AND (r.valid_to IS NULL OR r.valid_to > datetime())
  AND coalesce(s.quarantined, false) = false AND coalesce(o.quarantined, false) = false
RETURN p AS premise, count(r) > 0 AS ok
"""

_RETRACT_INFERRED = """
MATCH (s:Entity {tenant: $tenant, name: $src_name, type: $src_type})
      -[r:RELATES_TO {relation: $relation}]->
      (o:Entity {tenant: $tenant, name: $tgt_name, type: $tgt_type})
WHERE r.tenant = $tenant AND r.source_type = 'inferred'
SET r.confidence_state = 'RETRACTED', r.retracted_at = datetime(),
    r.retraction_reason = $reason, r.review_state = 'INSUFFICIENT_EVIDENCE'
"""

_REVALIDATE_INFERRED = """
MATCH (s:Entity {tenant: $tenant, name: $src_name, type: $src_type})
      -[r:RELATES_TO {relation: $relation}]->
      (o:Entity {tenant: $tenant, name: $tgt_name, type: $tgt_type})
WHERE r.tenant = $tenant AND r.source_type = 'inferred'
SET r.premise_keys = $premises, r.confidence = $confidence, r.review_state = null,
    r.revalidated_at = datetime()
"""

_SNAPSHOT_EVIDENCE = """
MATCH (s:CommunitySummarySnapshot {tenant: $tenant, id: $id})
WITH s,
     [c IN coalesce(s.chunk_ids, []) WHERE NOT EXISTS {
        MATCH (x:Chunk {tenant: $tenant, id: c}) }] AS missing_chunks,
     [d IN coalesce(s.document_ids, []) WHERE NOT EXISTS {
        MATCH (x:Document {tenant: $tenant, id: d})
        WHERE x.superseded_by IS NULL AND coalesce(x.is_deleted, false) = false
          AND (x.valid_to IS NULL OR x.valid_to > datetime()) }] AS bad_docs,
     [e IN coalesce(s.entity_ids, []) WHERE NOT EXISTS {
        MATCH (x:Entity {tenant: $tenant, id: e}) WHERE coalesce(x.quarantined, false) = false }] AS bad_entities
RETURN s.transaction_to IS NULL AS current, size(missing_chunks) AS missing_chunks,
       size(bad_docs) AS bad_docs, size(bad_entities) AS bad_entities
"""

_CLOSE_SNAPSHOT = """
MATCH (s:CommunitySummarySnapshot {tenant: $tenant, id: $id})
WHERE s.transaction_to IS NULL
SET s.transaction_to = datetime(), s.review_state = 'INSUFFICIENT_EVIDENCE',
    s.closed_reason = $reason
"""

_CLEAR_SNAPSHOT_FLAG = """
MATCH (s:CommunitySummarySnapshot {tenant: $tenant, id: $id}) SET s.review_state = null
"""

_DECISION_EVIDENCE = """
MATCH (dec:CGDecision {tenant: $tenant, id: $id})<-[:PRODUCED_DECISION]-(run:CGAgentRun {tenant: $tenant})
      -[:USED_CONTEXT]->(m:CGContextManifest {tenant: $tenant})
WITH coalesce(m.chunk_ids, []) AS chunks, coalesce(m.document_ids, []) AS docs
RETURN size(chunks) + size(docs) AS total,
       size([c IN chunks WHERE EXISTS { MATCH (x:Chunk {tenant: $tenant, id: c}) }]) +
       size([d IN docs WHERE EXISTS {
           MATCH (x:Document {tenant: $tenant, id: d})
           WHERE x.superseded_by IS NULL AND coalesce(x.is_deleted, false) = false
             AND (x.valid_to IS NULL OR x.valid_to > datetime()) }]) AS surviving
"""


def parse_relation_key(key: str) -> dict:
    src, relation, tgt = key.split("|", 2)
    st, sn = src.split(":", 1)
    tt, tn = tgt.split(":", 1)
    return {"src_type": st, "src_name": sn, "relation": relation, "tgt_type": tt, "tgt_name": tn}


class RecomputeWorker:
    def __init__(self, neo4j_client, *, engine=None, answer_recomputer: AnswerRecomputer | None = None):
        self._neo4j = neo4j_client
        self._store = StateStore(neo4j_client)
        self._engine = engine
        self._answer_recomputer = answer_recomputer

    def _engine_or_default(self):
        if self._engine is None:
            from graphrag.graph.inference_engine import ForwardChainingEngine
            self._engine = ForwardChainingEngine(self._neo4j)
        return self._engine

    async def run_once(self, tenant: str, *, limit: int = 50, kinds: list[str] | None = None) -> dict:
        claimed = await self._store.claim(tenant, limit=limit, kinds=kinds)
        outcomes: dict[str, int] = {}
        for item in claimed:
            try:
                state, detail, human = await self._recompute(tenant, item)
            except Exception as exc:  # noqa: BLE001 - one artifact must not stop the batch
                log.warning("invalidation.recompute_failed", kind=item["kind"], error=str(exc)[:200])
                state, detail, human = ArtifactState.RECOMPUTE_FAILED, f"{type(exc).__name__}: {exc}", False
            done = await self._store.complete(tenant, item, state, detail, human=human)
            outcome = state.value if done else "superseded_by_newer_invalidation"
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            metrics.record_recompute(item["kind"], outcome)
        return {"claimed": len(claimed), "outcomes": outcomes}

    async def _recompute(self, tenant: str, item: dict) -> tuple[ArtifactState, str, bool]:
        kind = ArtifactKind(item["kind"])
        if kind is ArtifactKind.INFERRED_EDGE:
            return await self._inferred_edge(tenant, item["artifact_id"])
        if kind is ArtifactKind.COMMUNITY_SNAPSHOT:
            return await self._snapshot(tenant, item["artifact_id"])
        return await self._decision(tenant, item["artifact_id"])

    async def _inferred_edge(self, tenant: str, key: str) -> tuple[ArtifactState, str, bool]:
        ref = parse_relation_key(key)
        rows = await run_read_only(self._neo4j, _EDGE, tenant=tenant, **ref)
        if not rows or rows[0].get("source_type") != "inferred":
            return ArtifactState.INSUFFICIENT_EVIDENCE, "inferred edge no longer exists", False
        edge = rows[0]
        if edge.get("state") == "RETRACTED":
            return ArtifactState.INSUFFICIENT_EVIDENCE, "already retracted", False
        premises = list(edge.get("premises") or [])
        if premises:
            checks = await run_read_only(self._neo4j, _PREMISES_OK, tenant=tenant, premises=premises)
            if checks and all(r["ok"] for r in checks):
                await self._clear_edge_flag(tenant, ref)
                return ArtifactState.VALID, "premises still valid", False
        derivation = await self._engine_or_default().derivation_for(
            edge.get("rule") or "", tenant=tenant, src_name=ref["src_name"], src_type=ref["src_type"],
            tgt_name=ref["tgt_name"], tgt_type=ref["tgt_type"])
        if derivation:
            await self._neo4j.run(_REVALIDATE_INFERRED, tenant=tenant, premises=derivation["premises"],
                                  confidence=derivation["confidence"], **ref)
            return ArtifactState.VALID, "re-derived from current premises", False
        await self._neo4j.run(_RETRACT_INFERRED, tenant=tenant, reason="premises invalidated", **ref)
        return ArtifactState.INSUFFICIENT_EVIDENCE, "no valid derivation; edge retracted", False

    async def _clear_edge_flag(self, tenant: str, ref: dict) -> None:
        await self._neo4j.run(
            """
            MATCH (s:Entity {tenant: $tenant, name: $src_name, type: $src_type})
                  -[r:RELATES_TO {relation: $relation}]->
                  (o:Entity {tenant: $tenant, name: $tgt_name, type: $tgt_type})
            WHERE r.tenant = $tenant AND r.source_type = 'inferred'
            SET r.review_state = null, r.revalidated_at = datetime()
            """,
            tenant=tenant, **ref,
        )

    async def _snapshot(self, tenant: str, snapshot_id: str) -> tuple[ArtifactState, str, bool]:
        rows = await run_read_only(self._neo4j, _SNAPSHOT_EVIDENCE, tenant=tenant, id=snapshot_id)
        if not rows:
            return ArtifactState.INSUFFICIENT_EVIDENCE, "snapshot no longer exists", False
        r = rows[0]
        if not r["current"]:
            return ArtifactState.INSUFFICIENT_EVIDENCE, "snapshot already superseded", False
        lost = r["missing_chunks"] + r["bad_docs"] + r["bad_entities"]
        if lost == 0:
            await self._neo4j.run(_CLEAR_SNAPSHOT_FLAG, tenant=tenant, id=snapshot_id)
            return ArtifactState.VALID, "all evidence still current", False
        detail = (f"missing_chunks={r['missing_chunks']} stale_documents={r['bad_docs']} "
                  f"quarantined_or_missing_entities={r['bad_entities']}")
        await self._neo4j.run(_CLOSE_SNAPSHOT, tenant=tenant, id=snapshot_id, reason=detail)
        return ArtifactState.INSUFFICIENT_EVIDENCE, f"closed: {detail}", False

    async def _decision(self, tenant: str, decision_id: str) -> tuple[ArtifactState, str, bool]:
        rows = await run_read_only(self._neo4j, _DECISION_EVIDENCE, tenant=tenant, id=decision_id)
        total = sum(r["total"] for r in rows)
        surviving = sum(r["surviving"] for r in rows)
        if total and surviving == 0:
            return ArtifactState.INSUFFICIENT_EVIDENCE, "none of the cited evidence is still current", False
        if self._answer_recomputer is not None:
            state = await self._answer_recomputer(tenant, decision_id)
            return state, "re-answered", False
        return (ArtifactState.NEEDS_REVIEW,
                f"{surviving}/{total} cited evidence items still current; answer needs human review", True)
