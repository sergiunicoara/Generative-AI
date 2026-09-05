import pytest

from src.domain.sales import SalesCRMWrite
from src.sales.adapter import CRMCommandError, LocalCRMEmulator


def _command(**overrides):
    values = {
        "command_id": "cmd-1", "workspace_id": "ws-a", "actor_id": "seller-1",
        "capability": "sales.opportunity.update", "object_id": "opp-1",
        "patch": {"stage": "NEGOTIATION"}, "expected_version": 1,
        "approved": True, "correlation_id": "corr-1",
    }
    values.update(overrides)
    return SalesCRMWrite(**values)


def test_local_adapter_preview_execute_and_hash_receipt():
    crm = LocalCRMEmulator()
    crm.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    preview = crm.preview(_command(dry_run=True))
    assert preview["before"] == {"stage": "PROPOSAL"}
    receipt = crm.execute(_command())
    assert receipt.outcome == "EXECUTED"
    assert len(receipt.receipt_hash) == 64
    assert receipt.verify()
    assert receipt.compensation is not None
    assert receipt.verified is True
    assert crm.get_record(workspace_id="ws-a", object_id="opp-1") == {"stage": "NEGOTIATION", "version": 2}


def test_preview_receipt_has_no_verified_flag():
    crm = LocalCRMEmulator()
    crm.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    receipt = crm.execute(_command(dry_run=True))
    assert receipt.outcome == "PREVIEW"
    assert receipt.verified is None


def test_get_record_returns_none_for_unknown_object():
    crm = LocalCRMEmulator()
    assert crm.get_record(workspace_id="ws-a", object_id="missing") is None


def test_get_record_returns_a_defensive_copy():
    crm = LocalCRMEmulator()
    crm.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    record = crm.get_record(workspace_id="ws-a", object_id="opp-1")
    record["stage"] = "TAMPERED"
    assert crm._records[("ws-a", "opp-1")]["stage"] == "PROPOSAL"


def test_stale_and_cross_tenant_commands_are_rejected():
    crm = LocalCRMEmulator()
    crm.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    crm.execute(_command())
    with pytest.raises(CRMCommandError, match="stale"):
        crm.execute(_command(command_id="cmd-2", expected_version=1))
    with pytest.raises(CRMCommandError, match="not found"):
        crm.execute(_command(command_id="cmd-3", workspace_id="ws-b"))


def test_idempotent_replay_returns_same_receipt_and_compensation_restores_state():
    crm = LocalCRMEmulator()
    crm.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    command = _command()
    first = crm.execute(command)
    replay = crm.execute(command)
    assert replay.receipt_hash == first.receipt_hash
    compensation = crm.compensate(first.compensation)
    assert compensation.outcome == "EXECUTED"
    assert crm._records[("ws-a", "opp-1")]["stage"] == "PROPOSAL"


def test_local_adapter_persists_receipts_and_state_atomically(tmp_path):
    path = tmp_path / "local-crm.json"
    first = LocalCRMEmulator(storage_path=path)
    first.seed(workspace_id="ws-a", object_id="opp-1", values={"stage": "PROPOSAL"})
    receipt = first.execute(_command())
    reloaded = LocalCRMEmulator(storage_path=path)
    replay = reloaded.execute(_command())
    assert replay.receipt_hash == receipt.receipt_hash
    assert replay.verify()
