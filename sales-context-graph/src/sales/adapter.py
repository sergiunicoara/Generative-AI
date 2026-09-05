"""Deterministic local CRM adapter used for offline demos and CI.

The adapter intentionally has no network or external-CRM side effects. Its
interface is small enough for a Salesforce/Dynamics implementation to replace
later while preserving command safety invariants.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from src.domain.sales import SalesCompensationAction, SalesCRMWrite
from src.sales.policy import PolicyCatalog, PolicyError


class CRMAdapter(Protocol):
    def preview(self, command: SalesCRMWrite) -> dict: ...
    def execute(self, command: SalesCRMWrite) -> "CRMReceipt": ...
    def compensate(self, action: SalesCompensationAction) -> "CRMReceipt": ...


class CRMCommandError(RuntimeError):
    """A rejected command that must not mutate CRM state."""


@dataclass(frozen=True)
class CRMReceipt:
    command_id: str
    workspace_id: str
    object_id: str
    outcome: str
    version: int
    diff: dict
    correlation_id: str
    recorded_at: str
    receipt_hash: str
    compensation: SalesCompensationAction | None = None
    # None = not applicable (PREVIEW/replay never mutate), True = the store was
    # read back after the write and matched. Not part of verify()'s hash
    # payload, so old persisted receipts keep validating.
    verified: bool | None = None

    def verify(self) -> bool:
        payload = {
            "command_id": self.command_id, "workspace_id": self.workspace_id,
            "object_id": self.object_id, "outcome": self.outcome, "version": self.version,
            "diff": self.diff, "correlation_id": self.correlation_id,
            "recorded_at": self.recorded_at,
        }
        return self.receipt_hash == LocalCRMEmulator._hash(payload)


class LocalCRMEmulator:
    """Synthetic, local CRM with policy, receipt and optional atomic JSON persistence."""

    def __init__(self, *, storage_path: Path | None = None, policy_catalog: PolicyCatalog | None = None) -> None:
        self._storage_path = storage_path
        self._policy_catalog = policy_catalog or PolicyCatalog()
        self._records: dict[tuple[str, str], dict] = {}
        self._commands: dict[tuple[str, str], CRMReceipt] = {}
        self.audit_events: list[dict] = []
        self._lock = threading.RLock()
        self._load()

    def seed(self, *, workspace_id: str, object_id: str, values: dict, version: int = 1) -> None:
        with self._lock:
            self._records[(workspace_id, object_id)] = {"version": version, **copy.deepcopy(values)}
            self._save()

    def _record(self, command: SalesCRMWrite) -> dict:
        try:
            return self._records[(command.workspace_id, command.object_id)]
        except KeyError as exc:
            raise CRMCommandError("CRM object not found in the command workspace") from exc

    def get_record(self, *, workspace_id: str, object_id: str) -> dict | None:
        """Public, read-only accessor. Unlike ``_record()`` this never raises
        on a miss -- callers decide for themselves whether that's an error --
        and returns a defensive copy so a caller can't mutate live state.

        Known limitation: this reads the same in-memory dict `execute()` just
        mutated in place, so a read-back from inside the same call always
        matches by construction today. It still earns its keep for interface
        parity with a future networked CRM connector, and as a regression
        guard if that live-reference behavior ever changes.
        """
        with self._lock:
            record = self._records.get((workspace_id, object_id))
            return copy.deepcopy(record) if record is not None else None

    def preview(self, command: SalesCRMWrite) -> dict:
        current = self._record(command)
        if current["version"] != command.expected_version:
            raise CRMCommandError("stale CRM version")
        before = {key: current.get(key) for key in command.patch}
        after = {key: value for key, value in command.patch.items()}
        return {"workspace_id": command.workspace_id, "object_id": command.object_id,
                "expected_version": command.expected_version, "before": before, "after": after}

    @staticmethod
    def _hash(payload: dict) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def execute(self, command: SalesCRMWrite) -> CRMReceipt:
        with self._lock:
            key = (command.workspace_id, command.command_id)
            existing = self._commands.get(key)
            if existing:
                if existing.diff.get("after") != command.patch:
                    raise CRMCommandError("command_id was already used with a different payload")
                self.audit_events.append({"event": "crm.command.replay", "command_id": command.command_id})
                self._save()
                return existing
            try:
                policy = self._policy_catalog.resolve(
                    workspace_id=command.workspace_id, policy_id=command.policy_id,
                    version=command.policy_version,
                )
                self._policy_catalog.enforce(policy=policy, patch=command.patch,
                                             approved=command.approved, dry_run=command.dry_run)
            except PolicyError as exc:
                raise CRMCommandError(str(exc)) from exc
            if command.dry_run:
                diff = self.preview(command)
                return self._receipt(command, "PREVIEW", command.expected_version, diff, None)
            diff = self.preview(command)
            record = self._record(command)
            previous = {field: record.get(field) for field in command.patch}
            record.update(command.patch)
            record["version"] += 1
            verified_record = self.get_record(workspace_id=command.workspace_id, object_id=command.object_id)
            mismatches = sorted(
                field for field, expected in command.patch.items()
                if verified_record is None or verified_record.get(field) != expected
            )
            if mismatches:
                # Abort before any success bookkeeping below -- a failed
                # verification must never look like a partial success.
                raise CRMCommandError(f"CRM write verification failed: fields did not round-trip: {mismatches}")
            compensation = SalesCompensationAction(
                compensation_id=f"compensate-{command.command_id}", workspace_id=command.workspace_id,
                original_command_id=command.command_id, object_id=command.object_id,
                restore_patch=previous,
            )
            receipt = self._receipt(command, "EXECUTED", record["version"], diff, compensation, verified=True)
            self._commands[key] = receipt
            self.audit_events.append({"event": "crm.command.executed", "command_id": command.command_id,
                                      "workspace_id": command.workspace_id, "correlation_id": command.correlation_id,
                                      "policy_id": policy.policy_id, "policy_version": policy.version})
            self._save()
            return receipt

    def compensate(self, action: SalesCompensationAction) -> CRMReceipt:
        command = SalesCRMWrite(
            command_id=action.compensation_id, workspace_id=action.workspace_id, actor_id="compensation",
            capability="sales.crm.compensate", object_id=action.object_id, patch=action.restore_patch,
            expected_version=self._records[(action.workspace_id, action.object_id)]["version"],
            approved=True, correlation_id=action.compensation_id,
        )
        receipt = self.execute(command)
        self.audit_events.append({"event": "crm.command.compensated", "original_command_id": action.original_command_id})
        self._save()
        return receipt

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        self._records = {(item["workspace_id"], item["object_id"]): item["values"] for item in payload.get("records", [])}
        self._commands = {}
        for item in payload.get("receipts", []):
            compensation = item.pop("compensation", None)
            self._commands[(item["workspace_id"], item["command_id"])] = CRMReceipt(
                **item, compensation=SalesCompensationAction(**compensation) if compensation else None,
            )
        self.audit_events = payload.get("audit_events", [])

    def _save(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "records": [{"workspace_id": workspace_id, "object_id": object_id, "values": values}
                        for (workspace_id, object_id), values in self._records.items()],
            "receipts": [{**receipt.__dict__, "compensation": receipt.compensation.model_dump(mode="json") if receipt.compensation else None}
                         for receipt in self._commands.values()],
            "audit_events": self.audit_events,
        }
        temporary = self._storage_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temporary, self._storage_path)

    def _receipt(self, command: SalesCRMWrite, outcome: str, version: int, diff: dict,
                 compensation: SalesCompensationAction | None, *, verified: bool | None = None) -> CRMReceipt:
        recorded_at = datetime.now(timezone.utc).isoformat()
        payload = {"command_id": command.command_id, "workspace_id": command.workspace_id,
                   "object_id": command.object_id, "outcome": outcome, "version": version, "diff": diff,
                   "correlation_id": command.correlation_id, "recorded_at": recorded_at}
        # CRMReceipt(**payload, ...) previously unpacked this dict, but mypy widens a
        # dict literal mixing str/int/dict values to dict[str, object], losing every
        # field's real type -- so it rejected `**payload` wholesale. `payload` (kept
        # as a plain dict, matching self._hash()'s signature) is passed to _hash()
        # unchanged; CRMReceipt itself is built straight from the well-typed locals.
        return CRMReceipt(
            command_id=command.command_id, workspace_id=command.workspace_id,
            object_id=command.object_id, outcome=outcome, version=version, diff=diff,
            correlation_id=command.correlation_id, recorded_at=recorded_at,
            receipt_hash=self._hash(payload), compensation=compensation, verified=verified,
        )
