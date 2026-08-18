"""Context Graph trace, governance, outcome, and proactive endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.auth.dependencies import assert_request_tenant, get_tenant, require_scope
from graphrag.context_graph.models import (
    CGAction, CGApproval, CGCorrection, CGExceptionGrant, CGFeedback, CGOutcome,
    DecisionTrace,
)
from graphrag.context_graph.proactive import ProactiveContextService, ProactiveThresholds
from graphrag.context_graph.repository import ContextGraphRepository
from graphrag.context_graph.trace_service import ContextGraphTraceService
from graphrag.context_graph.validation import validate_trace
from graphrag.graph.neo4j_client import get_neo4j
from graphrag.core.config import get_settings

router = APIRouter(prefix="/context-graph", tags=["Context Graph P0"])


def _assert_body_tenant(body_tenant: str, token_tenant: str) -> None:
    """Reject a request body whose tenant disagrees with the caller's token.

    Thin alias kept for the call sites below and their tests. The rule this
    enforces is not specific to the Context Graph — the same body/path-tenant
    hole existed on /kg/sources and /kg/cache/flush/{tenant} — so the
    implementation now lives in api/auth/dependencies.py next to get_tenant,
    where a route author looking for the tenant dependency will actually find
    it. See docs/context_graph_gap_plan.md F12.
    """
    assert_request_tenant(body_tenant, token_tenant)


def _proactive_service() -> ProactiveContextService:
    cfg = get_settings().context_graph.get("proactive", {})
    return ProactiveContextService(get_neo4j(), ProactiveThresholds(
        policy_expiry_days=int(cfg.get("policy_expiry_days", 30)),
        critical_expiry_days=int(cfg.get("critical_expiry_days", 7)),
        minimum_policy_uses=int(cfg.get("minimum_policy_uses", 1)),
    ))


class TraceRequest(BaseModel):
    trace: DecisionTrace


@router.post("/traces/validate", dependencies=[Depends(require_scope("read"))])
async def validate_context_trace(request: TraceRequest):
    validate_trace(request.trace)
    return {"valid": True, "decision_id": request.trace.decision.id,
            "integrity_hash": request.trace.manifest.integrity_hash}


@router.post("/traces", dependencies=[Depends(require_scope("write"))])
async def record_context_trace(request: TraceRequest, tenant: str = Depends(get_tenant)):
    _assert_body_tenant(request.trace.case.tenant, tenant)
    decision_id = await ContextGraphRepository(get_neo4j()).record_trace(request.trace)
    return {"decision_id": decision_id, "tenant": request.trace.case.tenant,
            "schema_version": request.trace.case.schema_version}


@router.get("/traces/{decision_id}", dependencies=[Depends(require_scope("read"))])
async def load_context_trace(decision_id: str, tenant: str = Depends(get_tenant)):
    return await ContextGraphRepository(get_neo4j()).load_trace(decision_id, tenant)


@router.get("/sessions/{session_id}/episodes", dependencies=[Depends(require_scope("read"))])
async def load_session_episodes(session_id: str, tenant: str = Depends(get_tenant), limit: int = 10):
    return await ContextGraphRepository(get_neo4j()).load_session_episodes(
        session_id, tenant, limit,
    )


class WPPTraceRequest(BaseModel):
    placement_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    statement_ids: list[str] = Field(min_length=1)
    statement_versions: list[str] = Field(min_length=1)
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_versions: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    document_versions: list[str] = Field(default_factory=list)
    selected: str = "escalate"
    policy_id: str = "data-privacy-policy"
    policy_version: str = "2024.1"


@router.post("/wpp/campaign-placement", dependencies=[Depends(require_scope("write"))])
async def record_wpp_campaign_trace(request: WPPTraceRequest, tenant: str = Depends(get_tenant)):
    service = ContextGraphTraceService(ContextGraphRepository(get_neo4j()))
    # Was **request.model_dump() alone -- WPPTraceRequest has no tenant
    # field, and record_wpp_campaign_placement's own signature defaults
    # tenant="marketing", so every call through this route silently recorded
    # its trace under the literal tenant "marketing" regardless of which
    # tenant the caller actually authenticated as. `tenant=tenant` overrides
    # that default with the token-derived value; it must come after the
    # dict-unpack so it wins if model_dump ever gains a tenant key.
    decision_id = await service.record_wpp_campaign_placement(
        **request.model_dump(), tenant=tenant,
    )
    return {"decision_id": decision_id, "tenant": tenant,
            "scenario": "wpp_campaign_placement"}


@router.post("/governance/events", dependencies=[Depends(require_scope("write"))])
async def append_governance_event(
    event: CGApproval | CGExceptionGrant | CGCorrection,
    tenant: str = Depends(get_tenant),
):
    _assert_body_tenant(event.tenant, tenant)
    event_id = await ContextGraphRepository(get_neo4j()).append_governance_event(event)
    return {"event_id": event_id, "tenant": event.tenant}


@router.post("/actions", dependencies=[Depends(require_scope("write"))])
async def record_action(action: CGAction, tenant: str = Depends(get_tenant)):
    _assert_body_tenant(action.tenant, tenant)
    return {"action_id": await ContextGraphRepository(get_neo4j()).record_action(action)}


@router.post("/outcomes", dependencies=[Depends(require_scope("write"))])
async def record_outcome(outcome: CGOutcome, tenant: str = Depends(get_tenant)):
    _assert_body_tenant(outcome.tenant, tenant)
    return {"outcome_id": await ContextGraphRepository(get_neo4j()).record_outcome(outcome)}


@router.post("/feedback", dependencies=[Depends(require_scope("write"))])
async def record_feedback(feedback: CGFeedback, tenant: str = Depends(get_tenant)):
    _assert_body_tenant(feedback.tenant, tenant)
    return {"feedback_id": await ContextGraphRepository(get_neo4j()).record_feedback(feedback)}


@router.get("/traces/{decision_id}/replay", dependencies=[Depends(require_scope("read"))])
async def replay_context_trace(decision_id: str, as_of: str, tenant: str = Depends(get_tenant)):
    return await ContextGraphRepository(get_neo4j()).replay_trace(decision_id, tenant, as_of)


@router.get("/traces/{decision_id}/governance", dependencies=[Depends(require_scope("read"))])
async def effective_context_governance(
    decision_id: str, tenant: str = Depends(get_tenant), as_of: datetime | None = None,
):
    return await ContextGraphRepository(get_neo4j()).effective_governance(
        decision_id, tenant, as_of
    )


@router.get("/traces/{decision_id}/supersession", dependencies=[Depends(require_scope("read"))])
async def context_supersession_chain(decision_id: str, tenant: str = Depends(get_tenant)):
    return await ContextGraphRepository(get_neo4j()).supersession_chain(decision_id, tenant)


class RetentionRequest(BaseModel):
    before: datetime
    actor_id: str = Field(min_length=1)
    reason_code: str = "retention_expired"
    dry_run: bool = True


@router.post("/retention/apply", dependencies=[Depends(require_scope("write"))])
async def apply_context_retention(request: RetentionRequest, tenant: str = Depends(get_tenant)):
    return await ContextGraphRepository(get_neo4j()).apply_retention_policy(
        tenant, request.before, request.actor_id,
        reason_code=request.reason_code, dry_run=request.dry_run,
    )


@router.get("/precedents", dependencies=[Depends(require_scope("read"))])
async def find_context_precedents(policy_version_id: str, tenant: str = Depends(get_tenant), limit: int = 10):
    return await ContextGraphRepository(get_neo4j()).find_precedents(tenant, policy_version_id, limit)


@router.get("/proactive/expiring-policies", dependencies=[Depends(require_scope("read"))])
async def expiring_context_policies(tenant: str = Depends(get_tenant), within_days: int | None = None):
    return [item.model_dump(mode="json") for item in await _proactive_service().expiring_policies(tenant, within_days)]
