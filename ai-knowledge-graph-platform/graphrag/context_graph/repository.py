"""Neo4j repository for the P0 Context Graph contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from graphrag.context_graph.models import (
    AgentRun, Case, CGEpisode, ContextManifest, Decision, DecisionOption,
    DecisionTrace, Observation, PolicyEvaluation, PolicyVersion, ToolCall,
    CGAction, CGApproval, CGCorrection, CGExceptionGrant, CGFeedback, CGOutcome,
)
from graphrag.context_graph.validation import ContextGraphValidationError, validate_trace
from graphrag.core.graph_props import props as _props


def _trace_hash(trace: DecisionTrace) -> str:
    payload = json.dumps(trace.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Decision fields that never carry evidence text (kept when a trace is redacted).
_TRACE_SAFE_KEYS = frozenset({
    "id", "tenant", "kind", "status", "decision_type", "manifest_id", "run_id", "case_id",
    "valid_from", "valid_to", "transaction_from", "transaction_to", "created_at", "integrity_hash",
    "policy_result", "selected_option_id", "schema_version",
})


class ContextGraphRepository:
    """Tenant-scoped, idempotent, append-only P0 graph persistence."""

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    async def create_case(self, case: Case) -> str:
        await self._neo4j.run(
            "MERGE (n:CGCase {tenant: $tenant, id: $id}) ON CREATE SET n += $props",
            tenant=case.tenant, id=case.id, props=_props(case),
        )
        return case.id

    async def start_agent_run(self, run: AgentRun) -> str:
        await self._neo4j.run(
            """
            MATCH (c:CGCase {tenant: $tenant, id: $case_id})
            MERGE (r:CGAgentRun {tenant: $tenant, id: $id}) ON CREATE SET r += $props
            MERGE (r)-[:ADDRESSES]->(c)
            """,
            tenant=run.tenant, case_id=run.case_id, id=run.id, props=_props(run),
        )
        return run.id

    async def complete_agent_run(self, run_id: str, tenant: str, status: str = "completed") -> None:
        await self._neo4j.run(
            """
            MATCH (r:CGAgentRun {tenant: $tenant, id: $id})
            SET r.status = $status
            """,
            tenant=tenant, id=run_id, status=status,
        )

    async def record_tool_call(self, call: ToolCall) -> str:
        await self._neo4j.run(
            """
            MATCH (r:CGAgentRun {tenant: $tenant, id: $run_id})
            MERGE (t:CGToolCall {tenant: $tenant, id: $id}) ON CREATE SET t += $props
            MERGE (r)-[:MADE_TOOL_CALL]->(t)
            """,
            tenant=call.tenant, run_id=call.run_id, id=call.id, props=_props(call),
        )
        return call.id

    async def record_observation(self, observation: Observation) -> str:
        await self._neo4j.run(
            """
            MATCH (t:CGToolCall {tenant: $tenant, id: $tool_call_id})
            MERGE (o:CGObservation {tenant: $tenant, id: $id}) ON CREATE SET o += $props
            MERGE (t)-[:PRODUCED]->(o)
            """,
            tenant=observation.tenant, tool_call_id=observation.tool_call_id,
            id=observation.id, props=_props(observation),
        )
        return observation.id

    async def record_episode(self, episode: CGEpisode) -> str:
        rows = await self._neo4j.run(
            """
            MATCH (r:CGAgentRun {tenant: $tenant, id: $run_id})
            MERGE (e:CGEpisode {tenant: $tenant, id: $id})
            ON CREATE SET e += $props
            MERGE (r)-[:RECORDED_EPISODE]->(e)
            RETURN e.id AS id
            """,
            tenant=episode.tenant,
            run_id=episode.run_id,
            id=episode.id,
            props=_props(episode),
        )
        if not rows:
            raise ContextGraphValidationError("episode references a missing or cross-tenant run")
        return episode.id

    async def _assert_kg_references(self, tenant: str, manifest: ContextManifest) -> None:
        for label, ids in (
            ("Statement", manifest.statement_ids),
            ("Chunk", manifest.chunk_ids),
            ("Document", manifest.document_ids),
        ):
            if not ids:
                continue
            rows = await self._neo4j.run(
                f"MATCH (n:{label}) WHERE n.tenant = $tenant AND n.id IN $ids "
                "RETURN count(n) AS found",
                tenant=tenant, ids=list(set(ids)),
            )
            found = rows[0].get("found", 0) if rows else 0
            if found != len(set(ids)):
                raise ContextGraphValidationError(
                    "one or more Knowledge Graph references are missing or cross-tenant"
                )

    async def persist_manifest(self, manifest: ContextManifest) -> str:
        if manifest.compute_integrity_hash() != manifest.integrity_hash:
            raise ContextGraphValidationError("manifest integrity hash does not match")
        await self._assert_kg_references(manifest.tenant, manifest)
        await self._neo4j.run(
            """
            MATCH (r:CGAgentRun {tenant: $tenant, id: $run_id})
            MERGE (m:CGContextManifest {tenant: $tenant, id: $id})
            ON CREATE SET m += $props
            MERGE (r)-[:USED_CONTEXT]->(m)
            WITH m
            UNWIND $statement_ids AS statement_id
            OPTIONAL MATCH (s:Statement {tenant: $tenant, id: statement_id})
            FOREACH (x IN CASE WHEN s IS NULL THEN [] ELSE [s] END |
              MERGE (m)-[:INCLUDED_STATEMENT]->(x))
            WITH m
            UNWIND $chunk_ids AS chunk_id
            OPTIONAL MATCH (c:Chunk {tenant: $tenant, id: chunk_id})
            FOREACH (x IN CASE WHEN c IS NULL THEN [] ELSE [c] END |
              MERGE (m)-[:INCLUDED_CHUNK]->(x))
            WITH m
            UNWIND $document_ids AS document_id
            OPTIONAL MATCH (d:Document {tenant: $tenant, id: document_id})
            FOREACH (x IN CASE WHEN d IS NULL THEN [] ELSE [d] END |
              MERGE (m)-[:INCLUDED_DOCUMENT]->(x))
            WITH m
            UNWIND $policy_version_ids AS policy_id
            MATCH (p:CGPolicyVersion {tenant: $tenant, id: policy_id})
            MERGE (m)-[:INCLUDED_POLICY]->(p)
            RETURN m.id AS manifest_id
            """,
            tenant=manifest.tenant, run_id=manifest.run_id, id=manifest.id,
            props=_props(manifest), statement_ids=manifest.statement_ids,
            chunk_ids=manifest.chunk_ids, document_ids=manifest.document_ids,
            policy_version_ids=manifest.policy_version_ids,
        )
        return manifest.id

    async def record_policy_version(self, policy: PolicyVersion) -> str:
        await self._neo4j.run(
            "MERGE (p:CGPolicyVersion {tenant: $tenant, id: $id}) ON CREATE SET p += $props",
            tenant=policy.tenant, id=policy.id, props=_props(policy),
        )
        return policy.id

    async def record_option(self, option: DecisionOption) -> str:
        await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            MERGE (o:CGOption {tenant: $tenant, id: $id}) ON CREATE SET o += $props
            MERGE (d)-[:CONSIDERED]->(o)
            FOREACH (_ IN CASE WHEN $disposition = 'selected' THEN [1] ELSE [] END |
              MERGE (d)-[:SELECTED]->(o))
            FOREACH (_ IN CASE WHEN $disposition = 'rejected' THEN [1] ELSE [] END |
              MERGE (d)-[:REJECTED]->(o))
            """,
            tenant=option.tenant, decision_id=option.decision_id, id=option.id,
            props=_props(option), disposition=option.disposition.value,
        )
        return option.id

    async def record_policy_evaluation(self, evaluation: PolicyEvaluation) -> str:
        await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id}),
                  (p:CGPolicyVersion {tenant: $tenant, id: $policy_version_id})
            MERGE (e:CGPolicyEvaluation {tenant: $tenant, id: $id}) ON CREATE SET e += $props
            MERGE (d)-[:HAS_POLICY_EVALUATION]->(e)
            MERGE (d)-[:APPLIED_POLICY]->(p)
            """,
            tenant=evaluation.tenant, decision_id=evaluation.decision_id,
            policy_version_id=evaluation.policy_version_id, id=evaluation.id,
            props=_props(evaluation),
        )
        return evaluation.id

    async def record_decision(self, decision: Decision) -> str:
        await self._neo4j.run(
            """
            MATCH (r:CGAgentRun {tenant: $tenant, id: $run_id}),
                  (m:CGContextManifest {tenant: $tenant, id: $manifest_id})
            MERGE (d:CGDecision {tenant: $tenant, id: $id}) ON CREATE SET d += $props
            MERGE (r)-[:PRODUCED_DECISION]->(d)
            """,
            tenant=decision.tenant, run_id=decision.run_id,
            manifest_id=decision.manifest_id, id=decision.id, props=_props(decision),
        )
        return decision.id

    async def record_trace(self, trace: DecisionTrace) -> str:
        """Validate everything, then persist through one atomic Cypher write."""
        validate_trace(trace)
        trace_hash = _trace_hash(trace)
        kg_refs = {
            "statement_ids": trace.manifest.statement_ids,
            "chunk_ids": trace.manifest.chunk_ids,
            "document_ids": trace.manifest.document_ids,
        }
        rows = await self._neo4j.run(
            """
            CALL {
              WITH $tenant AS tenant, $statement_ids AS ids
              UNWIND CASE WHEN size(ids) = 0 THEN [null] ELSE ids END AS id
              OPTIONAL MATCH (n:Statement {tenant: tenant, id: id})
              RETURN tenant, count(id) AS requested, count(n) AS found
            }
            WITH tenant, requested, found WHERE requested = found
            CALL {
              WITH tenant
              UNWIND CASE WHEN size($chunk_ids) = 0 THEN [null] ELSE $chunk_ids END AS id
              OPTIONAL MATCH (n:Chunk {tenant: tenant, id: id})
              RETURN count(id) AS chunk_requested, count(n) AS chunk_found
            }
            WITH tenant, requested, found, chunk_requested, chunk_found
            WHERE chunk_requested = chunk_found
            CALL {
              WITH tenant
              UNWIND CASE WHEN size($document_ids) = 0 THEN [null] ELSE $document_ids END AS id
              OPTIONAL MATCH (n:Document {tenant: tenant, id: id})
              RETURN count(id) AS document_requested, count(n) AS document_found
            }
            WITH tenant, requested, found, chunk_requested, chunk_found,
                 document_requested, document_found
            WHERE document_requested = document_found
            MERGE (d:CGDecision {tenant: tenant, id: $decision.id})
            ON CREATE SET d += $decision
            WITH tenant, d
            WHERE d.trace_hash = $trace_hash
            MERGE (c:CGCase {tenant: tenant, id: $case.id}) ON CREATE SET c += $case
            MERGE (r:CGAgentRun {tenant: tenant, id: $run.id}) ON CREATE SET r += $run
            MERGE (r)-[:ADDRESSES]->(c)
            FOREACH (item IN $policies |
              MERGE (p:CGPolicyVersion {tenant: tenant, id: item.id}) ON CREATE SET p += item)
            MERGE (m:CGContextManifest {tenant: tenant, id: $manifest.id}) ON CREATE SET m += $manifest
            MERGE (r)-[:USED_CONTEXT]->(m)
            WITH c, r, m, d, tenant
            CALL {
              WITH m, tenant
              UNWIND CASE WHEN size($statement_ids) = 0 THEN [null] ELSE $statement_ids END AS id
              OPTIONAL MATCH (s:Statement {tenant: tenant, id: id})
              FOREACH (x IN CASE WHEN s IS NULL THEN [] ELSE [s] END |
                MERGE (m)-[:INCLUDED_STATEMENT]->(x))
              RETURN count(*) AS linked_statements
            }
            WITH c, r, m, d, tenant
            CALL {
              WITH m, tenant
              UNWIND CASE WHEN size($chunk_ids) = 0 THEN [null] ELSE $chunk_ids END AS id
              OPTIONAL MATCH (x:Chunk {tenant: tenant, id: id})
              FOREACH (item IN CASE WHEN x IS NULL THEN [] ELSE [x] END |
                MERGE (m)-[:INCLUDED_CHUNK]->(item))
              RETURN count(*) AS linked_chunks
            }
            WITH c, r, m, d, tenant
            CALL {
              WITH m, tenant
              UNWIND CASE WHEN size($document_ids) = 0 THEN [null] ELSE $document_ids END AS id
              OPTIONAL MATCH (x:Document {tenant: tenant, id: id})
              FOREACH (item IN CASE WHEN x IS NULL THEN [] ELSE [x] END |
                MERGE (m)-[:INCLUDED_DOCUMENT]->(item))
              RETURN count(*) AS linked_documents
            }
            WITH c, r, m, d, tenant
            FOREACH (item IN $policies |
              MERGE (p:CGPolicyVersion {tenant: tenant, id: item.id})
              MERGE (m)-[:INCLUDED_POLICY]->(p))
            MERGE (r)-[:PRODUCED_DECISION]->(d)
            WITH c, r, m, d, tenant
            CALL {
              WITH d, tenant
              UNWIND CASE WHEN size($statement_ids) = 0 THEN [null] ELSE $statement_ids END AS id
              OPTIONAL MATCH (s:Statement {tenant: tenant, id: id})
              FOREACH (x IN CASE WHEN s IS NULL THEN [] ELSE [s] END |
                MERGE (d)-[:SUPPORTED_BY]->(x))
              RETURN count(id) AS supported_statements
            }
            FOREACH (item IN $tool_calls |
              MERGE (t:CGToolCall {tenant: tenant, id: item.id}) ON CREATE SET t += item
              MERGE (r)-[:MADE_TOOL_CALL]->(t))
            FOREACH (item IN $observations |
              MERGE (o:CGObservation {tenant: tenant, id: item.id}) ON CREATE SET o += item
              MERGE (t:CGToolCall {tenant: tenant, id: item.tool_call_id})
              MERGE (t)-[:PRODUCED]->(o))
            FOREACH (item IN $episodes |
              MERGE (ep:CGEpisode {tenant: tenant, id: item.id}) ON CREATE SET ep += item
              MERGE (r)-[:RECORDED_EPISODE]->(ep)
              MERGE (m)-[:INCLUDED_EPISODE]->(ep))
            FOREACH (item IN $options |
              MERGE (o:CGOption {tenant: tenant, id: item.id}) ON CREATE SET o += item
              MERGE (d)-[:CONSIDERED]->(o)
              FOREACH (_ IN CASE WHEN item.disposition = 'selected' THEN [1] ELSE [] END |
                MERGE (d)-[:SELECTED]->(o))
              FOREACH (_ IN CASE WHEN item.disposition = 'rejected' THEN [1] ELSE [] END |
                MERGE (d)-[:REJECTED]->(o)))
            FOREACH (item IN $evaluations |
              MERGE (e:CGPolicyEvaluation {tenant: tenant, id: item.id}) ON CREATE SET e += item
              MERGE (p:CGPolicyVersion {tenant: tenant, id: item.policy_version_id})
              MERGE (d)-[:HAS_POLICY_EVALUATION]->(e)
              MERGE (d)-[:APPLIED_POLICY]->(p))
            RETURN d.id AS decision_id, d.trace_hash AS existing_hash
            """,
            tenant=trace.case.tenant, case=_props(trace.case), run=_props(trace.run),
            manifest=_props(trace.manifest), decision={**_props(trace.decision), "trace_hash": trace_hash},
            trace_hash=trace_hash,
            policies=[_props(p) for p in trace.policy_versions],
            tool_calls=[_props(t) for t in trace.tool_calls],
            observations=[_props(o) for o in trace.observations],
            episodes=[_props(e) for e in trace.episodes],
            options=[_props(o) for o in trace.options],
            evaluations=[_props(e) for e in trace.policy_evaluations],
            **kg_refs,
        )
        if not rows:
            existing = await self._neo4j.run(
                "MATCH (d:CGDecision {tenant: $tenant, id: $decision_id}) "
                "RETURN d.trace_hash AS trace_hash",
                tenant=trace.case.tenant,
                decision_id=trace.decision.id,
            )
            if existing and existing[0].get("trace_hash") != trace_hash:
                raise ContextGraphValidationError("completed Context Graph decision is immutable")
            raise ContextGraphValidationError("one or more Knowledge Graph references are missing or cross-tenant")
        existing_hash = rows[0].get("existing_hash")
        if existing_hash and existing_hash != trace_hash:
            raise ContextGraphValidationError("completed Context Graph decision is immutable")
        return trace.decision.id

    async def _authorize_trace(self, trace: dict, tenant: str, access_context) -> dict:
        """Remove what the caller may not read from a loaded trace (plan Phase 6).

        Embeddings are never returned. With access control enabled, chunks and
        documents the caller is not authorised for are removed using the same
        fail-closed predicate as retrieval; if anything was removed, the
        decision's free text (which may restate that evidence) is redacted too.
        """
        from graphrag.core.config import get_settings
        from graphrag.enterprise.access import access_params, document_access_predicate

        for chunk in trace.get("chunks") or []:
            chunk.pop("embedding", None)
        for doc in trace.get("documents") or []:
            doc.pop("embedding", None)
        if not get_settings().access_control.get("enabled", False):
            return trace
        doc_ids = {d.get("id") for d in trace.get("documents") or [] if d.get("id")}
        doc_ids |= {c.get("document_id") for c in trace.get("chunks") or [] if c.get("document_id")}
        manifest = trace.get("manifest") or {}
        doc_ids |= set(manifest.get("document_ids") or [])
        allowed: set[str] = set()
        if doc_ids:
            rows = await self._neo4j.run(
                "UNWIND $ids AS id MATCH (d:Document {tenant: $tenant, id: id}) WHERE d.tenant = $tenant "
                + document_access_predicate("d") + " RETURN d.id AS id",
                ids=sorted(doc_ids), tenant=tenant, **access_params(access_context, enabled=True),
            )
            allowed = {r["id"] for r in rows}
        removed = doc_ids - allowed
        trace["chunks"] = [c for c in trace.get("chunks") or [] if c.get("document_id") in allowed]
        trace["documents"] = [d for d in trace.get("documents") or [] if d.get("id") in allowed]
        if manifest:
            manifest["document_ids"] = [d for d in manifest.get("document_ids") or [] if d in allowed]
            allowed_chunks = {c.get("id") for c in trace["chunks"]}
            manifest["chunk_ids"] = [c for c in manifest.get("chunk_ids") or [] if c in allowed_chunks]
        if removed:
            decision = trace.get("decision") or {}
            for key in list(decision):
                if isinstance(decision[key], str) and key not in _TRACE_SAFE_KEYS:
                    decision[key] = "[redacted: cites evidence you are not authorized to read]"
            if manifest.get("task_input"):
                manifest["task_input"] = "[redacted]"
            trace["observations"] = []
            trace["redaction"] = {"reason": "unauthorized_evidence", "removed_documents": len(removed)}
        return trace

    async def load_trace(self, decision_id: str, tenant: str, access_context=None) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            MATCH (r:CGAgentRun {tenant: $tenant})-[:PRODUCED_DECISION]->(d)
            MATCH (r)-[:ADDRESSES]->(c:CGCase {tenant: $tenant})
            OPTIONAL MATCH (r)-[:USED_CONTEXT]->(m:CGContextManifest)
            OPTIONAL MATCH (r)-[:MADE_TOOL_CALL]->(t:CGToolCall)
            OPTIONAL MATCH (t)-[:PRODUCED]->(o:CGObservation {tenant: $tenant})
            OPTIONAL MATCH (r)-[:RECORDED_EPISODE]->(ep:CGEpisode {tenant: $tenant})
            OPTIONAL MATCH (d)-[:CONSIDERED]->(op:CGOption {tenant: $tenant})
            OPTIONAL MATCH (d)-[:HAS_POLICY_EVALUATION]->(e:CGPolicyEvaluation {tenant: $tenant})
            OPTIONAL MATCH (d)-[:APPLIED_POLICY]->(p:CGPolicyVersion {tenant: $tenant})
            OPTIONAL MATCH (m)-[:INCLUDED_STATEMENT]->(s:Statement {tenant: $tenant})
            OPTIONAL MATCH (m)-[:INCLUDED_CHUNK]->(ch:Chunk {tenant: $tenant})
            OPTIONAL MATCH (m)-[:INCLUDED_DOCUMENT]->(doc:Document {tenant: $tenant})
            RETURN c {.*} AS case, r {.*} AS run, m {.*} AS manifest,
              d {.*} AS decision, collect(DISTINCT t {.*}) AS tool_calls,
              collect(DISTINCT o {.*}) AS observations,
              collect(DISTINCT ep {.*}) AS episodes,
              collect(DISTINCT op {.*}) AS options,
              collect(DISTINCT e {.*}) AS policy_evaluations,
              collect(DISTINCT p {.*}) AS policy_versions,
              collect(DISTINCT s {.*}) AS statements,
              collect(DISTINCT ch {.*}) AS chunks,
              collect(DISTINCT doc {.*}) AS documents
            """,
            tenant=tenant, decision_id=decision_id,
        )
        if not rows:
            return {}
        return await self._authorize_trace(dict(rows[0]), tenant, access_context)

    async def load_session_episodes(
        self, session_id: str, tenant: str, limit: int = 10,
    ) -> list[dict]:
        """Load durable agent/session memory in chronological order."""
        rows = await self._neo4j.run(
            """
            MATCH (e:CGEpisode {tenant: $tenant, session_id: $session_id})
            RETURN e {.*} AS episode
            ORDER BY e.created_at DESC, e.sequence DESC
            LIMIT $limit
            """,
            tenant=tenant,
            session_id=session_id,
            limit=max(1, min(limit, 100)),
        )
        return [dict(row["episode"]) for row in reversed(rows) if row.get("episode")]

    async def append_governance_event(self, event: CGApproval | CGExceptionGrant | CGCorrection) -> str:
        labels = {
            CGApproval: ("CGApproval", "GOVERNED_BY", "decision_id"),
            CGExceptionGrant: ("CGExceptionGrant", "USED_EXCEPTION", "decision_id"),
            CGCorrection: ("CGCorrection", "CORRECTED_BY", "decision_id"),
        }
        label, relation, target_field = labels[type(event)]
        props = _props(event)
        query = f"""
            MATCH (d:CGDecision {{tenant: $tenant, id: $target_id}})
            MERGE (e:{label} {{tenant: $tenant, id: $id}})
            ON CREATE SET e += $props
            MERGE (d)-[:{relation}]->(e)
            RETURN e.id AS id
        """
        if isinstance(event, CGCorrection):
            query = f"""
                MATCH (old:CGDecision {{tenant: $tenant, id: $target_id}}),
                      (new:CGDecision {{tenant: $tenant, id: $replacement_id}})
                OPTIONAL MATCH cycle=(new)-[:SUPERSEDED_BY*1..]->(old)
                WITH old, new, cycle
                WHERE cycle IS NULL
                MERGE (e:{label} {{tenant: $tenant, id: $id}})
                ON CREATE SET e += $props
                MERGE (old)-[:SUPERSEDED_BY]->(new)
                MERGE (old)-[:{relation}]->(e)
                RETURN e.id AS id
            """
        rows = await self._neo4j.run(
            query, tenant=event.tenant, target_id=getattr(event, target_field),
            replacement_id=getattr(event, "replacement_decision_id", None),
            id=event.id, props=props,
        )
        if not rows:
            raise ContextGraphValidationError(
                "governance event references a missing or cross-tenant decision, or creates a cycle"
            )
        return event.id

    async def supersession_chain(self, decision_id: str, tenant: str) -> list[dict]:
        """Load the complete append-only replacement chain from a decision."""
        rows = await self._neo4j.run(
            """
            MATCH path=(start:CGDecision {tenant: $tenant, id: $decision_id})
                       -[:SUPERSEDED_BY*0..]->(current:CGDecision {tenant: $tenant})
            WHERE NOT (current)-[:SUPERSEDED_BY]->(:CGDecision {tenant: $tenant})
            RETURN [node IN nodes(path) | node {.*}] AS decisions
            ORDER BY length(path) DESC
            LIMIT 1
            """,
            tenant=tenant,
            decision_id=decision_id,
        )
        return list(rows[0]["decisions"]) if rows else []

    async def record_action(self, action: CGAction) -> str:
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            MERGE (a:CGAction {tenant: $tenant, id: $id}) ON CREATE SET a += $props
            MERGE (d)-[:RESULTED_IN]->(a)
            RETURN a.id AS id
            """, tenant=action.tenant, decision_id=action.decision_id,
            id=action.id, props=_props(action),
        )
        if not rows:
            raise ContextGraphValidationError("action references a missing or cross-tenant decision")
        return action.id

    async def record_outcome(self, outcome: CGOutcome) -> str:
        rows = await self._neo4j.run(
            """
            MATCH (a:CGAction {tenant: $tenant, id: $action_id})
            MERGE (o:CGOutcome {tenant: $tenant, id: $id}) ON CREATE SET o += $props
            MERGE (a)-[:PRODUCED]->(o)
            RETURN o.id AS id
            """, tenant=outcome.tenant, action_id=outcome.action_id,
            id=outcome.id, props=_props(outcome),
        )
        if not rows:
            raise ContextGraphValidationError("outcome references a missing or cross-tenant action")
        return outcome.id

    async def record_feedback(self, feedback: CGFeedback) -> str:
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            OPTIONAL MATCH (d)-[:RESULTED_IN]->(:CGAction {tenant: $tenant})
                           -[:PRODUCED]->(o:CGOutcome {tenant: $tenant, id: $outcome_id})
            WITH d, o
            WHERE $outcome_id IS NULL OR o IS NOT NULL
            MERGE (f:CGFeedback {tenant: $tenant, id: $id}) ON CREATE SET f += $props
            MERGE (f)-[:EVALUATES]->(d)
            FOREACH (_ IN CASE WHEN $outcome_id IS NULL THEN [] ELSE [1] END |
              MERGE (f)-[:ASSESSES]->(o))
            RETURN f.id AS id
            """, tenant=feedback.tenant, decision_id=feedback.decision_id,
            outcome_id=feedback.outcome_id, id=feedback.id, props=_props(feedback),
        )
        if not rows:
            raise ContextGraphValidationError(
                "feedback references a missing/cross-tenant decision or an outcome outside that decision"
            )
        return feedback.id

    async def replay_trace(self, decision_id: str, tenant: str, as_of: str, access_context=None) -> dict:
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            WHERE datetime(d.transaction_from) <= datetime($as_of)
              AND (d.transaction_to IS NULL OR datetime(d.transaction_to) >= datetime($as_of))
            OPTIONAL MATCH (r:CGAgentRun {tenant: $tenant})-[:PRODUCED_DECISION]->(d)
            RETURN d {.*} AS decision, collect(DISTINCT r {.*}) AS runs
            """, tenant=tenant, decision_id=decision_id, as_of=as_of,
        )
        if not rows:
            return {}
        replay = dict(rows[0])
        # The replayed decision may restate evidence: apply the same authorization
        # as load_trace by checking the decision's manifest documents.
        full = await self.load_trace(decision_id, tenant, access_context)
        if full.get("redaction"):
            replay["decision"] = full.get("decision")
            replay["redaction"] = full["redaction"]
        return replay

    async def find_precedents(self, tenant: str, policy_version_id: str, limit: int = 10) -> list[dict]:
        rows = await self._neo4j.run(
            """
            MATCH (current:CGPolicyVersion {tenant: $tenant, id: $policy_version_id})
            MATCH (d:CGDecision {tenant: $tenant})-[:APPLIED_POLICY]->
                  (p:CGPolicyVersion {tenant: $tenant})
            WHERE d.status = 'final'
            OPTIONAL MATCH (d)-[:RESULTED_IN]->(a:CGAction {tenant: $tenant})
            OPTIONAL MATCH (a)-[:PRODUCED]->(o:CGOutcome {tenant: $tenant})
            OPTIONAL MATCH (f:CGFeedback {tenant: $tenant})-[:EVALUATES]->(d)
            OPTIONAL MATCH (f)-[:ASSESSES]->(assessed:CGOutcome {tenant: $tenant})
            WITH d, p, current,
                 max(CASE o.status WHEN 'observed' THEN 1.0 WHEN 'expected' THEN 0.5 ELSE 0.0 END) AS outcome_score,
                 coalesce(avg(CASE WHEN assessed IS NULL THEN null ELSE f.score END), 0.5) AS feedback_score,
                 count(DISTINCT assessed) AS assessed_outcomes
            WITH d, p, outcome_score, feedback_score, assessed_outcomes,
                 (0.65 * CASE WHEN p.policy_id = current.policy_id THEN 1.0 ELSE 0.0 END
                  + 0.20 * outcome_score
                  + 0.15 * feedback_score) AS score
            RETURN d {.*} AS decision, p {.*} AS policy, score, outcome_score,
                   feedback_score, assessed_outcomes
            ORDER BY score DESC, d.created_at DESC
            LIMIT $limit
            """, tenant=tenant, policy_version_id=policy_version_id, limit=limit,
        )
        return [dict(row) for row in rows]

    async def redact_trace(self, decision_id: str, tenant: str, reason_code: str, actor_id: str) -> str:
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            MERGE (r:CGRedaction {tenant: $tenant, id: $redaction_id})
            ON CREATE SET r.reason_code = $reason_code, r.actor_id = $actor_id,
                          r.created_at = datetime(), r.schema_version = 'context-graph/v1'
            MERGE (d)-[:REDACTED_BY]->(r)
            RETURN r.id AS id
            """, tenant=tenant, decision_id=decision_id,
            redaction_id=f"redaction-{decision_id}-{reason_code}",
            reason_code=reason_code, actor_id=actor_id,
        )
        if not rows:
            raise ContextGraphValidationError("cannot redact a missing or cross-tenant trace")
        return rows[0]["id"]

    async def effective_governance(
        self, decision_id: str, tenant: str, as_of: datetime | None = None,
    ) -> dict:
        """Return approval and exception state effective at a point in time."""
        effective_at = (as_of or datetime.now(timezone.utc)).isoformat()
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant, id: $decision_id})
            OPTIONAL MATCH (d)-[:GOVERNED_BY]->(a:CGApproval {tenant: $tenant})
            WHERE datetime(a.created_at) <= datetime($as_of)
            WITH d, a ORDER BY a.created_at DESC
            WITH d, head(collect(a)) AS approval
            OPTIONAL MATCH (d)-[:USED_EXCEPTION]->(e:CGExceptionGrant {tenant: $tenant})
            WHERE datetime(e.created_at) <= datetime($as_of)
            WITH approval, e ORDER BY e.created_at DESC
            WITH approval, collect(e) AS exceptions
            RETURN approval {.*} AS approval,
                   CASE WHEN approval.status = 'approved'
                          AND (approval.expires_at IS NULL OR datetime(approval.expires_at) > datetime($as_of))
                        THEN true ELSE false END AS approval_effective,
                   [e IN exceptions WHERE e.status = 'granted'
                     AND (e.expires_at IS NULL OR datetime(e.expires_at) > datetime($as_of)) | e {.*}]
                     AS active_exceptions
            """,
            tenant=tenant,
            decision_id=decision_id,
            as_of=effective_at,
        )
        if not rows:
            raise ContextGraphValidationError("governance references a missing or cross-tenant decision")
        return dict(rows[0])

    async def apply_retention_policy(
        self,
        tenant: str,
        before: datetime,
        actor_id: str,
        *,
        reason_code: str = "retention_expired",
        dry_run: bool = True,
    ) -> dict:
        """Find or append redaction markers for traces beyond retention."""
        if before.tzinfo is None:
            raise ValueError("retention boundary must be timezone-aware")
        if dry_run:
            rows = await self._neo4j.run(
                """
                MATCH (d:CGDecision {tenant: $tenant})
                WHERE datetime(d.created_at) < datetime($before)
                  AND NOT (d)-[:REDACTED_BY]->(:CGRedaction {tenant: $tenant})
                RETURN d.id AS decision_id
                ORDER BY d.created_at
                """,
                tenant=tenant,
                before=before.isoformat(),
            )
            return {"tenant": tenant, "dry_run": True, "matched": len(rows),
                    "decision_ids": [row["decision_id"] for row in rows]}
        rows = await self._neo4j.run(
            """
            MATCH (d:CGDecision {tenant: $tenant})
            WHERE datetime(d.created_at) < datetime($before)
              AND NOT (d)-[:REDACTED_BY]->(:CGRedaction {tenant: $tenant})
            WITH collect(d) AS decisions
            FOREACH (d IN decisions |
              MERGE (r:CGRedaction {tenant: $tenant, id: 'retention-' + d.id})
              ON CREATE SET r.reason_code = $reason_code, r.actor_id = $actor_id,
                            r.created_at = datetime(), r.schema_version = 'context-graph/v1'
              MERGE (d)-[:REDACTED_BY]->(r))
            RETURN size(decisions) AS marked
            """,
            tenant=tenant,
            before=before.isoformat(),
            actor_id=actor_id,
            reason_code=reason_code,
        )
        return {"tenant": tenant, "dry_run": False,
                "marked": int(rows[0]["marked"]) if rows else 0}
