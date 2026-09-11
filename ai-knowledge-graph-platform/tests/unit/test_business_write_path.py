"""Unit tests for the P0 safe write path: WorkOrderService + its HTTP surface.

Service-level tests use the shared `neo4j_mock` fixture (AsyncMock-over-
``.run``), with `begin_corpus_update`/`complete_corpus_update` stubbed
separately since `CorpusMutation` calls them directly rather than through
`.run()`. HTTP-level tests use the `TestClient` + minimal-FastAPI-app +
`dependency_overrides` pattern from `test_query_routes.py`.

Covers the three required outcome tests from the Wave 5 plan (success,
denied-then-approved, stale-version-rejected) plus the supporting unit
tests around each branch of `WorkOrderService.create_from_finding`.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth.dependencies import get_current_user
from api.routes import business as business_routes
from graphrag.business.commands import CommandEnvelope, CommandOutcome
from graphrag.business.service import WORKORDER_CREATE_CAPABILITY, WorkOrderService


def _envelope(**kw) -> CommandEnvelope:
    defaults = dict(
        command_id="cmd-1", capability=WORKORDER_CREATE_CAPABILITY, tenant="aerospace",
        actor_id="agent-1", actor_type="human", reason_code="remediation",
        args={"originating_finding_id": "finding-1", "title": "Rotate key"},
        expected_version=1,
    )
    defaults.update(kw)
    return CommandEnvelope(**defaults)


def _finding_row(**kw) -> dict:
    defaults = dict(id="finding-1", tenant="aerospace", status="open",
                     severity="medium", object_version=1)
    defaults.update(kw)
    return {"finding": defaults}


def _stub_corpus_mutation(neo4j_mock, revision: int = 7) -> None:
    neo4j_mock.begin_corpus_update = AsyncMock(return_value=None)
    neo4j_mock.complete_corpus_update = AsyncMock(return_value=revision)


# ── Outcome 1: success ──────────────────────────────────────────────────

class TestCreateFromFindingSuccess:
    async def test_executed_creates_work_order_and_advances_finding(self, neo4j_mock):
        _stub_corpus_mutation(neo4j_mock, revision=7)
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt: no existing receipt
            [_finding_row()],  # get_finding
            [{"finding_version": 2, "finding_from_state": "open",
              "work_order_id": "wo-1", "work_order_version": 1}],  # atomic write
            [],  # save_command_receipt
        ])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope())

        assert receipt.outcome == CommandOutcome.EXECUTED
        assert receipt.object_id == "wo-1"
        assert receipt.object_type == "WorkOrder"
        assert receipt.from_state == "open"
        assert receipt.to_state == "remediating"
        assert receipt.to_version == 2
        assert receipt.corpus_revision == 7
        assert receipt.receipt_hash  # non-empty, computed

        # Every mutating call must carry the tenant.
        for call in neo4j_mock.run.call_args_list:
            assert call.kwargs.get("tenant") == "aerospace"
        neo4j_mock.begin_corpus_update.assert_awaited_once()
        neo4j_mock.complete_corpus_update.assert_awaited_once()

    async def test_repeat_call_with_same_command_id_short_circuits(self, neo4j_mock):
        _stub_corpus_mutation(neo4j_mock)
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # first call: no existing receipt
            [_finding_row()],
            [{"finding_version": 2, "finding_from_state": "open",
              "work_order_id": "wo-1", "work_order_version": 1}],
            [],  # save receipt
        ])
        service = WorkOrderService(neo4j_mock)
        first = await service.create_from_finding(_envelope())

        # Second call with the identical command_id: get_command_receipt now
        # returns the stored receipt -- nothing else should run.
        neo4j_mock.run = AsyncMock(return_value=[{"receipt": first.model_dump(mode="json")}])
        second = await service.create_from_finding(_envelope())

        assert neo4j_mock.run.await_count == 1  # only the idempotency lookup
        assert second.command_id == first.command_id
        assert second.outcome == CommandOutcome.EXECUTED


# ── Outcome 2: denied-then-approved (CRITICAL severity escalates) ──────

class TestCreateFromFindingApprovalFlow:
    async def test_critical_severity_escalates_without_writing_work_order(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt
            [_finding_row(severity="critical")],  # get_finding
            [],  # create_approval (MERGE, no meaningful return)
        ])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope(command_id="cmd-crit"))

        assert receipt.outcome == CommandOutcome.APPROVAL_REQUIRED
        assert receipt.approval_id
        assert receipt.policy_result == "escalate"
        # Exactly the 3 expected calls ran (receipt check, finding read,
        # approval creation) -- no CorpusMutation was ever entered, so no
        # work order write reached the mock.
        assert neo4j_mock.run.await_count == 3

    async def test_approval_denied_when_decider_is_the_requester(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[
            {"approval": {"id": "appr-1", "tenant": "aerospace", "requested_by": "agent-1",
                          "status": "requested", "capability": WORKORDER_CREATE_CAPABILITY}},
        ])
        service = WorkOrderService(neo4j_mock)
        with pytest.raises(PermissionError):
            await service.decide_approval("aerospace", "appr-1", approved=True, actor_id="agent-1")
        # get_approval was the only call -- the decide write must never run.
        assert neo4j_mock.run.await_count == 1

    async def test_approval_granted_by_different_actor_then_retry_executes(self, neo4j_mock):
        service = WorkOrderService(neo4j_mock)
        # Different actor ("approver-1") than the requester ("agent-1") succeeds.
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"approval": {"id": "appr-1", "tenant": "aerospace", "requested_by": "agent-1",
                           "status": "requested", "capability": WORKORDER_CREATE_CAPABILITY}}],
            [{"approval": {"id": "appr-1", "tenant": "aerospace", "status": "approved",
                           "approved_by": "approver-1"}}],
        ])
        decided = await service.decide_approval("aerospace", "appr-1", approved=True, actor_id="approver-1")
        assert decided["status"] == "approved"

        _stub_corpus_mutation(neo4j_mock, revision=9)
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt (never persisted for approval_required)
            [_finding_row(severity="critical")],  # get_finding
            [{"approval": {"id": "appr-1", "tenant": "aerospace", "status": "approved",
                           "capability": WORKORDER_CREATE_CAPABILITY}}],  # get_approval
            [{"finding_version": 2, "finding_from_state": "open",
              "work_order_id": "wo-2", "work_order_version": 1}],  # atomic write
            [],  # save_command_receipt
        ])
        retried = await service.create_from_finding(
            _envelope(command_id="cmd-crit", approval_id="appr-1"),
        )
        assert retried.outcome == CommandOutcome.EXECUTED
        assert retried.object_id == "wo-2"

    async def test_cross_tenant_approval_reuse_rejected(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt
            [_finding_row(severity="critical")],  # get_finding
            [],  # get_approval: tenant filter excludes another tenant's approval -> not found
            [],  # save_command_receipt (DENIED is persisted)
        ])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(
            _envelope(command_id="cmd-x", approval_id="someone-elses-approval"),
        )
        assert receipt.outcome == CommandOutcome.DENIED
        assert receipt.denial_reason == "approval_not_found"


# ── Outcome 3: stale version rejected ───────────────────────────────────

class TestCreateFromFindingStaleVersion:
    async def test_stale_version_rejects_with_zero_writes_and_no_corpus_revision(self, neo4j_mock):
        _stub_corpus_mutation(neo4j_mock, revision=99)
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt
            [_finding_row(object_version=3)],  # get_finding
            [],  # atomic write: WHERE object_version=1 matches nothing (actual is 3)
            [{"status": "open", "version": 3}],  # disambiguation read
            [],  # save_command_receipt (DENIED/STALE_VERSION are persisted)
        ])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope(expected_version=1))

        assert receipt.outcome == CommandOutcome.STALE_VERSION
        assert receipt.from_version == 1
        assert receipt.to_version == 3
        assert receipt.corpus_revision is None
        assert receipt.object_id is None  # no WorkOrder was created

    async def test_retry_with_correct_version_succeeds(self, neo4j_mock):
        _stub_corpus_mutation(neo4j_mock, revision=100)
        neo4j_mock.run = AsyncMock(side_effect=[
            [],  # get_command_receipt (a fresh command_id for the corrected retry)
            [_finding_row(object_version=3)],  # get_finding
            [{"finding_version": 4, "finding_from_state": "open",
              "work_order_id": "wo-3", "work_order_version": 1}],  # atomic write succeeds
            [],  # save_command_receipt
        ])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(
            _envelope(command_id="cmd-retry", expected_version=3),
        )
        assert receipt.outcome == CommandOutcome.EXECUTED
        assert receipt.to_version == 4


# ── Other denial branches ───────────────────────────────────────────────

class TestCreateFromFindingOtherDenials:
    async def test_capability_mismatch_denied(self, neo4j_mock):
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope(capability="biz.other@1.0.0"))
        assert receipt.outcome == CommandOutcome.DENIED
        assert receipt.denial_reason == "capability_mismatch"

    async def test_missing_finding_denied(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(side_effect=[[], [], []])  # no receipt, no finding, save receipt
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope())
        assert receipt.outcome == CommandOutcome.DENIED
        assert receipt.denial_reason == "finding_not_found"

    async def test_invalid_args_denied(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(
            _envelope(args={"originating_finding_id": "", "title": ""}),
        )
        assert receipt.outcome == CommandOutcome.DENIED
        assert receipt.denial_reason == "invalid_args"

    async def test_dry_run_never_writes(self, neo4j_mock):
        neo4j_mock.run = AsyncMock(side_effect=[[], [_finding_row()]])
        service = WorkOrderService(neo4j_mock)
        receipt = await service.create_from_finding(_envelope(dry_run=True))
        assert receipt.outcome == CommandOutcome.DRY_RUN
        assert neo4j_mock.run.await_count == 2  # only the two reads, no write


# ── HTTP status-code mapping ─────────────────────────────────────────────

def _make_client(scope: str = "biz:write biz:read biz:approve") -> TestClient:
    app = FastAPI()
    app.include_router(business_routes.router)
    app.dependency_overrides[get_current_user] = lambda: {
        "scope": scope, "sub": "agent-1", "tenant": "aerospace", "type": "m2m",
    }
    return TestClient(app)


class TestBusinessRoutesStatusCodes:
    def test_executed_returns_201(self):
        client = _make_client()
        receipt = AsyncMock(return_value=_receipt_stub(CommandOutcome.EXECUTED))
        with patch("api.routes.business.WorkOrderService") as mock_cls:
            mock_cls.return_value.create_from_finding = receipt
            resp = client.post("/business/work-orders", json=_create_body())
        assert resp.status_code == 201

    def test_approval_required_returns_202(self):
        client = _make_client()
        receipt = AsyncMock(return_value=_receipt_stub(CommandOutcome.APPROVAL_REQUIRED))
        with patch("api.routes.business.WorkOrderService") as mock_cls:
            mock_cls.return_value.create_from_finding = receipt
            resp = client.post("/business/work-orders", json=_create_body())
        assert resp.status_code == 202

    def test_stale_version_returns_409(self):
        client = _make_client()
        receipt = AsyncMock(return_value=_receipt_stub(CommandOutcome.STALE_VERSION))
        with patch("api.routes.business.WorkOrderService") as mock_cls:
            mock_cls.return_value.create_from_finding = receipt
            resp = client.post("/business/work-orders", json=_create_body())
        assert resp.status_code == 409

    def test_missing_scope_denied_before_service_runs(self):
        client = _make_client(scope="biz:read")  # no biz:write
        with patch("api.routes.business.WorkOrderService") as mock_cls:
            resp = client.post("/business/work-orders", json=_create_body())
        assert resp.status_code == 403
        mock_cls.assert_not_called()

    def test_approval_decision_requires_biz_approve_scope(self):
        client = _make_client(scope="biz:write")  # no biz:approve
        resp = client.post("/business/approvals/appr-1/decide", json={"approved": True})
        assert resp.status_code == 403


def _create_body() -> dict:
    return {
        "reason_code": "remediation", "originating_finding_id": "finding-1",
        "title": "Rotate key", "expected_version": 1,
    }


def _receipt_stub(outcome: CommandOutcome):
    from graphrag.business.commands import CommandReceipt
    return CommandReceipt(
        tenant="aerospace", command_id="cmd-1", capability=WORKORDER_CREATE_CAPABILITY,
        outcome=outcome,
    ).with_receipt_hash()
