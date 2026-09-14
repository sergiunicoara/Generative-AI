"""Durable governance state for the Energy Asset Intelligence dataset.

RDF remains the semantic source of truth, stored as content-addressed
N-Triples in :mod:`graph_blobs`.  This store persists the *governance around*
that graph: immutable publication metadata, the active-version pointer,
workflow heads and append-only transition log, and idempotency receipts.

SQLite is deliberately the default for the self-contained demo.  WAL plus a
short ``BEGIN IMMEDIATE`` critical section makes separate API processes
serialize writes safely.  ``ENERGY_GOVERNANCE_DB_URL`` may point to an async
PostgreSQL SQLAlchemy URL in a deployed environment; the public API remains
the same and Neo4j stays an explicitly rebuildable projection.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from rdflib import Graph
from rdflib.namespace import RDF
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from graphrag.domains.energy.graph_blobs import (
    dumps_prefixes, prefixes_of, read_graph, write_graph,
)
from graphrag.domains.energy.publication import (
    PrunedReference, PublicationReport, PublicationRollbackError, QuarantinedRecord,
)
from graphrag.domains.energy.vocabulary import ENERGY, REC

REVIEW_REQUIRED = "review_required"
APPROVED = "approved"
REJECTED = "rejected"
COMPLETED = "completed"
CANCELLED = "cancelled"
ALLOWED_TRANSITIONS = {
    REVIEW_REQUIRED: frozenset({APPROVED, REJECTED}),
    APPROVED: frozenset({COMPLETED, CANCELLED}),
    REJECTED: frozenset(),
    COMPLETED: frozenset(),
    CANCELLED: frozenset(),
}

# Evidence-request lifecycle: a governed follow-up task tracking a request for
# missing operational evidence. Distinct state space from the work-order
# review lifecycle above -- an evidence request is never itself a maintenance
# conclusion and never mutates published RDF.
EVIDENCE_OPEN = "open"
EVIDENCE_IN_PROGRESS = "in_progress"
EVIDENCE_FULFILLED = "fulfilled"
EVIDENCE_CANCELLED = "cancelled"
EVIDENCE_ALLOWED_TRANSITIONS = {
    EVIDENCE_OPEN: frozenset({EVIDENCE_IN_PROGRESS, EVIDENCE_CANCELLED}),
    EVIDENCE_IN_PROGRESS: frozenset({EVIDENCE_FULFILLED, EVIDENCE_CANCELLED}),
    EVIDENCE_FULFILLED: frozenset(),
    EVIDENCE_CANCELLED: frozenset(),
}

ROOT = Path(__file__).resolve().parents[3]


class GovernanceStoreError(RuntimeError):
    """Base error raised by the durable governance boundary."""


class UnknownWorkOrderError(GovernanceStoreError):
    pass


class WorkflowConflictError(GovernanceStoreError):
    """A stale object version or invalid state transition was rejected."""


class CommandReuseError(GovernanceStoreError):
    """A command id was reused for a materially different request."""


class UnknownEvidenceRequestError(GovernanceStoreError):
    pass


class EvidenceRequestConflictError(GovernanceStoreError):
    """A stale object version or invalid state transition was rejected."""


class GovernanceStore:
    """Async SQLAlchemy governance store with an append-only audit log."""

    def __init__(
        self, db_url: str | None = None, *, blob_root: Path | None = None,
    ) -> None:
        configured = db_url or os.getenv("ENERGY_GOVERNANCE_DB_URL")
        if configured:
            self.db_url = configured
        else:
            database = ROOT / "artifacts" / "energy" / "governance.sqlite"
            self.db_url = f"sqlite+aiosqlite:///{database.as_posix()}"
        self.blob_root = blob_root or ROOT / "artifacts" / "energy" / "published"
        self._is_sqlite = self.db_url.startswith("sqlite")
        if self._is_sqlite:
            database_path = self.db_url.split("///", 1)[-1]
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._engine: AsyncEngine = create_async_engine(
            self.db_url, connect_args={"timeout": 30} if self._is_sqlite else {},
            pool_pre_ping=not self._is_sqlite,
        )

    async def open(self) -> None:
        """Create schema and configure SQLite durability before use."""
        async with self._engine.begin() as connection:
            if self._is_sqlite:
                await connection.exec_driver_sql("PRAGMA journal_mode=WAL")
                await connection.exec_driver_sql("PRAGMA synchronous=FULL")
                await connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                await connection.exec_driver_sql("PRAGMA busy_timeout=30000")
            for statement in _DDL:
                await connection.execute(text(statement))

    async def close(self) -> None:
        await self._engine.dispose()

    @asynccontextmanager
    async def _write_transaction(self):
        """Acquire a writer lock only around the metadata/CAS mutation.

        RDF materialisation, SHACL fixpoint validation and blob writes happen
        before this context is entered, so an expensive publish never blocks
        unrelated workflow commands while it computes.
        """
        async with self._engine.connect() as connection:
            if self._is_sqlite:
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
                transaction = None
            else:
                transaction = await connection.begin()
            try:
                yield connection
                if self._is_sqlite:
                    await connection.commit()
                else:
                    await transaction.commit()
            except BaseException:
                if self._is_sqlite:
                    await connection.rollback()
                else:
                    await transaction.rollback()
                raise

    async def publish(self, tenant: str, report: PublicationReport, graph: Graph) -> PublicationReport:
        """Atomically make a conformant graph version active for ``tenant``.

        Blob creation is content-addressed and therefore idempotent.  If the
        active pointer already names identical bytes, the existing persisted
        report is returned rather than manufacturing a misleading version.
        """
        digest, path, _size = await asyncio.to_thread(write_graph, graph, self.blob_root)
        prefixes = dumps_prefixes(prefixes_of(graph))
        work_orders = sorted(
            str(subject).removeprefix(str(REC))
            for subject in graph.subjects(RDF.type, ENERGY.WorkOrder)
            if str(subject).startswith(str(REC))
        )
        async with self._write_transaction() as connection:
            active = await connection.execute(text("""
                SELECT p.report_json FROM energy_active_versions a
                JOIN energy_publications p ON p.version_id = a.version_id
                WHERE a.tenant = :tenant AND p.blob_hash = :blob_hash
            """), {"tenant": tenant, "blob_hash": digest})
            existing = active.scalar_one_or_none()
            if existing:
                return _report_from_json(existing)

            await connection.execute(text("""
                INSERT INTO energy_publications
                (version_id, tenant, published_at, blob_hash, blob_path, prefixes_json, report_json, rolled_back_from)
                VALUES (:version_id, :tenant, :published_at, :blob_hash, :blob_path, :prefixes_json, :report_json, :rolled_back_from)
            """), {
                "version_id": report.version_id, "tenant": tenant,
                "published_at": report.published_at, "blob_hash": digest,
                "blob_path": str(path), "prefixes_json": prefixes,
                "report_json": _report_to_json(report), "rolled_back_from": report.rolled_back_from,
            })
            await self._set_active_version(connection, tenant, report.version_id)
            for work_order_id in work_orders:
                await connection.execute(text("""
                    INSERT INTO energy_published_work_orders (version_id, work_order_id)
                    SELECT :version_id, :work_order_id
                    WHERE NOT EXISTS (
                        SELECT 1 FROM energy_published_work_orders
                        WHERE version_id = :version_id AND work_order_id = :work_order_id
                    )
                """), {"version_id": report.version_id, "work_order_id": work_order_id})
                await connection.execute(text("""
                    INSERT INTO energy_workflow_heads (tenant, work_order_id, state, object_version, updated_at)
                    SELECT :tenant, :work_order_id, :state, 0, :updated_at
                    WHERE NOT EXISTS (
                        SELECT 1 FROM energy_workflow_heads
                        WHERE tenant = :tenant AND work_order_id = :work_order_id
                    )
                """), {
                    "tenant": tenant, "work_order_id": work_order_id,
                    "state": REVIEW_REQUIRED, "updated_at": _now_iso(),
                })
        return report

    async def current(self, tenant: str) -> tuple[PublicationReport, Graph]:
        async with self._engine.connect() as connection:
            row = (await connection.execute(text("""
                SELECT p.report_json, p.blob_path, p.prefixes_json
                FROM energy_active_versions a JOIN energy_publications p ON p.version_id = a.version_id
                WHERE a.tenant = :tenant
            """), {"tenant": tenant})).mappings().one_or_none()
        if row is None:
            raise PublicationRollbackError("no version has been published yet")
        graph = await asyncio.to_thread(read_graph, Path(row["blob_path"]), json.loads(row["prefixes_json"]))
        return _report_from_json(row["report_json"]), graph

    async def history(self, tenant: str) -> list[PublicationReport]:
        async with self._engine.connect() as connection:
            rows = (await connection.execute(text("""
                SELECT report_json FROM energy_publications WHERE tenant = :tenant
                ORDER BY published_at, version_id
            """), {"tenant": tenant})).scalars().all()
        return [_report_from_json(item) for item in rows]

    async def rollback(self, tenant: str, version_id: str | None = None) -> PublicationReport:
        async with self._write_transaction() as connection:
            rows = (await connection.execute(text("""
                SELECT version_id, published_at, blob_hash, blob_path, prefixes_json, report_json
                FROM energy_publications WHERE tenant = :tenant ORDER BY published_at, version_id
            """), {"tenant": tenant})).mappings().all()
            if not rows:
                raise PublicationRollbackError("no version has been published yet")
            if version_id is None:
                if len(rows) < 2:
                    raise PublicationRollbackError("only one version has been published; there is nothing to roll back to")
                target = rows[-2]
            else:
                target = next((row for row in rows if row["version_id"] == version_id), None)
                if target is None:
                    raise PublicationRollbackError(f"no published version with id {version_id!r}")
            original = _report_from_json(target["report_json"])
            restored = PublicationReport(
                version_id=uuid4().hex, published_at=_now_iso(),
                published_triple_count=original.published_triple_count,
                candidate_record_count=original.candidate_record_count,
                quarantined_records=original.quarantined_records,
                rolled_back_from=target["version_id"],
                pruned_references=original.pruned_references,
                revalidation_passes=original.revalidation_passes, conforms=original.conforms,
            )
            await connection.execute(text("""
                INSERT INTO energy_publications
                (version_id, tenant, published_at, blob_hash, blob_path, prefixes_json, report_json, rolled_back_from)
                VALUES (:version_id, :tenant, :published_at, :blob_hash, :blob_path, :prefixes_json, :report_json, :rolled_back_from)
            """), {
                "version_id": restored.version_id, "tenant": tenant,
                "published_at": restored.published_at, "blob_hash": target["blob_hash"],
                "blob_path": target["blob_path"], "prefixes_json": target["prefixes_json"],
                "report_json": _report_to_json(restored), "rolled_back_from": target["version_id"],
            })
            source_work_orders = (await connection.execute(text("""
                SELECT work_order_id FROM energy_published_work_orders WHERE version_id = :version_id
            """), {"version_id": target["version_id"]})).scalars().all()
            for work_order_id in source_work_orders:
                await connection.execute(text("""
                    INSERT INTO energy_published_work_orders (version_id, work_order_id)
                    VALUES (:version_id, :work_order_id)
                """), {"version_id": restored.version_id, "work_order_id": work_order_id})
            await self._set_active_version(connection, tenant, restored.version_id)
        return restored

    async def current_state(self, tenant: str, work_order_id: str) -> tuple[str, int]:
        async with self._engine.connect() as connection:
            await self._require_active_work_order(connection, tenant, work_order_id)
            row = (await connection.execute(text("""
                SELECT state, object_version FROM energy_workflow_heads
                WHERE tenant = :tenant AND work_order_id = :work_order_id
            """), {"tenant": tenant, "work_order_id": work_order_id})).mappings().one()
        return str(row["state"]), int(row["object_version"])

    async def transition_history(self, tenant: str, work_order_id: str) -> list[dict[str, object]]:
        async with self._engine.connect() as connection:
            await self._require_active_work_order(connection, tenant, work_order_id)
            rows = (await connection.execute(text("""
                SELECT transition_id, work_order_id, from_state, to_state, changed_at, changed_by, reason, object_version
                FROM energy_workflow_transitions
                WHERE tenant = :tenant AND work_order_id = :work_order_id
                ORDER BY changed_at, transition_id
            """), {"tenant": tenant, "work_order_id": work_order_id})).mappings().all()
        return [dict(row) for row in rows]

    async def transition(
        self, tenant: str, work_order_id: str, *, to_state: str, changed_by: str,
        reason: str, expected_version: int, command_id: str,
    ) -> tuple[dict[str, object], str, bool]:
        """Append one transition guarded by both command receipt and CAS.

        Returns ``(transition, response_json, replayed)``.  The API emits
        ``response_json`` verbatim for a replay, so a retry receives exactly
        the original response bytes rather than a newly serialised lookalike.
        """
        if not changed_by.strip():
            raise WorkflowConflictError("changed_by is required")
        if not reason.strip():
            raise WorkflowConflictError("reason is required")
        if not command_id.strip():
            raise WorkflowConflictError("command_id is required")
        fingerprint = _fingerprint({
            "work_order_id": work_order_id, "to_state": to_state, "changed_by": changed_by,
            "reason": reason, "expected_version": expected_version,
        })
        async with self._write_transaction() as connection:
            receipt = (await connection.execute(text("""
                SELECT fingerprint, response_json FROM energy_command_receipts
                WHERE tenant = :tenant AND command_id = :command_id
            """), {"tenant": tenant, "command_id": command_id})).mappings().one_or_none()
            if receipt is not None:
                if receipt["fingerprint"] != fingerprint:
                    raise CommandReuseError("command_id was already used with different arguments")
                response_json = str(receipt["response_json"])
                return json.loads(response_json), response_json, True

            await self._require_active_work_order(connection, tenant, work_order_id)
            head = (await connection.execute(text("""
                SELECT state, object_version FROM energy_workflow_heads
                WHERE tenant = :tenant AND work_order_id = :work_order_id
            """), {"tenant": tenant, "work_order_id": work_order_id})).mappings().one()
            from_state, actual_version = str(head["state"]), int(head["object_version"])
            if actual_version != expected_version:
                raise WorkflowConflictError(
                    f"expected object_version {expected_version}, but stored version is {actual_version}"
                )
            if to_state not in ALLOWED_TRANSITIONS.get(from_state, frozenset()):
                raise WorkflowConflictError(
                    f"cannot transition {work_order_id} from {from_state!r} to {to_state!r}"
                )
            to_version = actual_version + 1
            changed_at = _now_iso()
            updated = await connection.execute(text("""
                UPDATE energy_workflow_heads SET state = :to_state, object_version = :to_version, updated_at = :changed_at
                WHERE tenant = :tenant AND work_order_id = :work_order_id AND object_version = :expected_version
            """), {
                "tenant": tenant, "work_order_id": work_order_id, "to_state": to_state,
                "to_version": to_version, "changed_at": changed_at, "expected_version": expected_version,
            })
            if updated.rowcount != 1:
                raise WorkflowConflictError("workflow changed concurrently; reload and retry")
            transition = {
                "transition_id": uuid4().hex, "work_order_id": work_order_id,
                "from_state": from_state, "to_state": to_state, "changed_at": changed_at,
                "changed_by": changed_by, "reason": reason, "object_version": to_version,
            }
            await connection.execute(text("""
                INSERT INTO energy_workflow_transitions
                (transition_id, tenant, work_order_id, from_state, to_state, changed_at, changed_by, reason, object_version)
                VALUES (:transition_id, :tenant, :work_order_id, :from_state, :to_state, :changed_at, :changed_by, :reason, :object_version)
            """), {"tenant": tenant, **transition})
            response_json = json.dumps(transition, sort_keys=True, separators=(",", ":"))
            await connection.execute(text("""
                INSERT INTO energy_command_receipts (tenant, command_id, fingerprint, response_json, created_at)
                VALUES (:tenant, :command_id, :fingerprint, :response_json, :created_at)
            """), {
                "tenant": tenant, "command_id": command_id, "fingerprint": fingerprint,
                "response_json": response_json, "created_at": changed_at,
            })
        return transition, response_json, False

    async def create_evidence_request(
        self, tenant: str, *, asset_id: str, missing_field: str, target_source_system: str,
        owner: str, priority: str, reason: str, created_by: str, command_id: str,
    ) -> tuple[dict[str, object], str, bool]:
        """Create a governed evidence-request record.

        This never touches published RDF, source-system fixtures, or the
        active-version pointer -- it is purely a tracked follow-up task.
        Returns ``(record, response_json, replayed)`` with the same
        idempotent-replay contract as :meth:`transition`.
        """
        if not owner.strip():
            raise EvidenceRequestConflictError("owner is required")
        if not reason.strip():
            raise EvidenceRequestConflictError("reason is required")
        if not command_id.strip():
            raise EvidenceRequestConflictError("command_id is required")
        fingerprint = _fingerprint({
            "asset_id": asset_id, "missing_field": missing_field,
            "target_source_system": target_source_system, "owner": owner,
            "priority": priority, "reason": reason,
        })
        async with self._write_transaction() as connection:
            receipt = (await connection.execute(text("""
                SELECT fingerprint, response_json FROM energy_command_receipts
                WHERE tenant = :tenant AND command_id = :command_id
            """), {"tenant": tenant, "command_id": command_id})).mappings().one_or_none()
            if receipt is not None:
                if receipt["fingerprint"] != fingerprint:
                    raise CommandReuseError("command_id was already used with different arguments")
                response_json = str(receipt["response_json"])
                return json.loads(response_json), response_json, True

            created_at = _now_iso()
            record = {
                "request_id": uuid4().hex, "tenant": tenant, "asset_id": asset_id,
                "missing_field": missing_field, "target_source_system": target_source_system,
                "owner": owner, "priority": priority, "state": EVIDENCE_OPEN, "reason": reason,
                "created_at": created_at, "updated_at": created_at, "created_by": created_by,
                "object_version": 0,
            }
            await connection.execute(text("""
                INSERT INTO energy_evidence_requests
                (request_id, tenant, asset_id, missing_field, target_source_system, owner,
                 priority, state, reason, created_at, updated_at, created_by, object_version)
                VALUES (:request_id, :tenant, :asset_id, :missing_field, :target_source_system,
                        :owner, :priority, :state, :reason, :created_at, :updated_at, :created_by, :object_version)
            """), record)
            response_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
            await connection.execute(text("""
                INSERT INTO energy_command_receipts (tenant, command_id, fingerprint, response_json, created_at)
                VALUES (:tenant, :command_id, :fingerprint, :response_json, :created_at)
            """), {
                "tenant": tenant, "command_id": command_id, "fingerprint": fingerprint,
                "response_json": response_json, "created_at": created_at,
            })
        return record, response_json, False

    async def evidence_requests(self, tenant: str) -> list[dict[str, object]]:
        async with self._engine.connect() as connection:
            rows = (await connection.execute(text("""
                SELECT request_id, tenant, asset_id, missing_field, target_source_system, owner,
                       priority, state, reason, created_at, updated_at, created_by, object_version
                FROM energy_evidence_requests WHERE tenant = :tenant ORDER BY created_at, request_id
            """), {"tenant": tenant})).mappings().all()
        return [dict(row) for row in rows]

    async def evidence_request(self, tenant: str, request_id: str) -> dict[str, object]:
        async with self._engine.connect() as connection:
            row = await self._require_evidence_request(connection, tenant, request_id)
        return dict(row)

    async def evidence_request_transition_history(self, tenant: str, request_id: str) -> list[dict[str, object]]:
        async with self._engine.connect() as connection:
            await self._require_evidence_request(connection, tenant, request_id)
            rows = (await connection.execute(text("""
                SELECT transition_id, request_id, from_state, to_state, changed_at, changed_by, reason, object_version
                FROM energy_evidence_request_transitions
                WHERE tenant = :tenant AND request_id = :request_id
                ORDER BY changed_at, transition_id
            """), {"tenant": tenant, "request_id": request_id})).mappings().all()
        return [dict(row) for row in rows]

    async def transition_evidence_request(
        self, tenant: str, request_id: str, *, to_state: str, changed_by: str,
        reason: str, expected_version: int, command_id: str,
    ) -> tuple[dict[str, object], str, bool]:
        """Append one evidence-request transition guarded by CAS + idempotency.

        Completing or cancelling a request only updates this governance
        record -- it never writes to the published RDF graph, so it cannot by
        construction change an ``insufficient_evidence`` answer.
        """
        if not changed_by.strip():
            raise EvidenceRequestConflictError("changed_by is required")
        if not reason.strip():
            raise EvidenceRequestConflictError("reason is required")
        if not command_id.strip():
            raise EvidenceRequestConflictError("command_id is required")
        fingerprint = _fingerprint({
            "request_id": request_id, "to_state": to_state, "changed_by": changed_by,
            "reason": reason, "expected_version": expected_version,
        })
        async with self._write_transaction() as connection:
            receipt = (await connection.execute(text("""
                SELECT fingerprint, response_json FROM energy_command_receipts
                WHERE tenant = :tenant AND command_id = :command_id
            """), {"tenant": tenant, "command_id": command_id})).mappings().one_or_none()
            if receipt is not None:
                if receipt["fingerprint"] != fingerprint:
                    raise CommandReuseError("command_id was already used with different arguments")
                response_json = str(receipt["response_json"])
                return json.loads(response_json), response_json, True

            current = await self._require_evidence_request(connection, tenant, request_id)
            from_state, actual_version = str(current["state"]), int(current["object_version"])
            if actual_version != expected_version:
                raise EvidenceRequestConflictError(
                    f"expected object_version {expected_version}, but stored version is {actual_version}"
                )
            if to_state not in EVIDENCE_ALLOWED_TRANSITIONS.get(from_state, frozenset()):
                raise EvidenceRequestConflictError(
                    f"cannot transition {request_id} from {from_state!r} to {to_state!r}"
                )
            to_version = actual_version + 1
            changed_at = _now_iso()
            updated = await connection.execute(text("""
                UPDATE energy_evidence_requests SET state = :to_state, object_version = :to_version, updated_at = :changed_at
                WHERE tenant = :tenant AND request_id = :request_id AND object_version = :expected_version
            """), {
                "tenant": tenant, "request_id": request_id, "to_state": to_state,
                "to_version": to_version, "changed_at": changed_at, "expected_version": expected_version,
            })
            if updated.rowcount != 1:
                raise EvidenceRequestConflictError("evidence request changed concurrently; reload and retry")
            transition = {
                "transition_id": uuid4().hex, "request_id": request_id,
                "from_state": from_state, "to_state": to_state, "changed_at": changed_at,
                "changed_by": changed_by, "reason": reason, "object_version": to_version,
            }
            await connection.execute(text("""
                INSERT INTO energy_evidence_request_transitions
                (transition_id, tenant, request_id, from_state, to_state, changed_at, changed_by, reason, object_version)
                VALUES (:transition_id, :tenant, :request_id, :from_state, :to_state, :changed_at, :changed_by, :reason, :object_version)
            """), {"tenant": tenant, **transition})
            response_json = json.dumps(transition, sort_keys=True, separators=(",", ":"))
            await connection.execute(text("""
                INSERT INTO energy_command_receipts (tenant, command_id, fingerprint, response_json, created_at)
                VALUES (:tenant, :command_id, :fingerprint, :response_json, :created_at)
            """), {
                "tenant": tenant, "command_id": command_id, "fingerprint": fingerprint,
                "response_json": response_json, "created_at": changed_at,
            })
        return transition, response_json, False

    async def _require_evidence_request(
        self, connection: AsyncConnection, tenant: str, request_id: str,
    ) -> dict[str, object]:
        row = (await connection.execute(text("""
            SELECT request_id, tenant, asset_id, missing_field, target_source_system, owner,
                   priority, state, reason, created_at, updated_at, created_by, object_version
            FROM energy_evidence_requests WHERE tenant = :tenant AND request_id = :request_id
        """), {"tenant": tenant, "request_id": request_id})).mappings().one_or_none()
        if row is None:
            raise UnknownEvidenceRequestError(f"unknown evidence request {request_id!r}")
        return dict(row)

    async def _set_active_version(self, connection: AsyncConnection, tenant: str, version_id: str) -> None:
        updated = await connection.execute(text("""
            UPDATE energy_active_versions SET version_id = :version_id WHERE tenant = :tenant
        """), {"tenant": tenant, "version_id": version_id})
        if updated.rowcount == 0:
            await connection.execute(text("""
                INSERT INTO energy_active_versions (tenant, version_id) VALUES (:tenant, :version_id)
            """), {"tenant": tenant, "version_id": version_id})

    async def _require_active_work_order(
        self, connection: AsyncConnection, tenant: str, work_order_id: str,
    ) -> None:
        row = await connection.execute(text("""
            SELECT 1 FROM energy_active_versions a
            JOIN energy_published_work_orders w ON w.version_id = a.version_id
            WHERE a.tenant = :tenant AND w.work_order_id = :work_order_id
        """), {"tenant": tenant, "work_order_id": work_order_id})
        if row.scalar_one_or_none() is None:
            raise UnknownWorkOrderError(f"unknown work order {work_order_id!r} in active published graph")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _report_to_json(report: PublicationReport) -> str:
    return json.dumps(asdict(report), sort_keys=True, separators=(",", ":"))


def _report_from_json(payload: str) -> PublicationReport:
    raw = json.loads(payload)
    return PublicationReport(
        version_id=raw["version_id"], published_at=raw["published_at"],
        published_triple_count=raw["published_triple_count"], candidate_record_count=raw["candidate_record_count"],
        quarantined_records=[QuarantinedRecord(**item) for item in raw.get("quarantined_records", [])],
        rolled_back_from=raw.get("rolled_back_from"),
        pruned_references=[PrunedReference(**item) for item in raw.get("pruned_references", [])],
        revalidation_passes=raw.get("revalidation_passes", 1), conforms=raw.get("conforms", True),
    )


_DDL = (
    """CREATE TABLE IF NOT EXISTS energy_publications (
        version_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, published_at TEXT NOT NULL,
        blob_hash TEXT NOT NULL, blob_path TEXT NOT NULL, prefixes_json TEXT NOT NULL,
        report_json TEXT NOT NULL, rolled_back_from TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS energy_active_versions (
        tenant TEXT PRIMARY KEY, version_id TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS energy_published_work_orders (
        version_id TEXT NOT NULL, work_order_id TEXT NOT NULL,
        PRIMARY KEY (version_id, work_order_id)
    )""",
    """CREATE TABLE IF NOT EXISTS energy_workflow_heads (
        tenant TEXT NOT NULL, work_order_id TEXT NOT NULL, state TEXT NOT NULL,
        object_version INTEGER NOT NULL, updated_at TEXT NOT NULL,
        PRIMARY KEY (tenant, work_order_id)
    )""",
    """CREATE TABLE IF NOT EXISTS energy_workflow_transitions (
        transition_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, work_order_id TEXT NOT NULL,
        from_state TEXT NOT NULL, to_state TEXT NOT NULL, changed_at TEXT NOT NULL,
        changed_by TEXT NOT NULL, reason TEXT NOT NULL, object_version INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS energy_command_receipts (
        tenant TEXT NOT NULL, command_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
        response_json TEXT NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (tenant, command_id)
    )""",
    """CREATE TABLE IF NOT EXISTS energy_evidence_requests (
        request_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, asset_id TEXT NOT NULL,
        missing_field TEXT NOT NULL, target_source_system TEXT NOT NULL,
        owner TEXT NOT NULL, priority TEXT NOT NULL, state TEXT NOT NULL, reason TEXT NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, created_by TEXT NOT NULL,
        object_version INTEGER NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_energy_evidence_requests_tenant
        ON energy_evidence_requests (tenant, created_at)""",
    """CREATE TABLE IF NOT EXISTS energy_evidence_request_transitions (
        transition_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, request_id TEXT NOT NULL,
        from_state TEXT NOT NULL, to_state TEXT NOT NULL, changed_at TEXT NOT NULL,
        changed_by TEXT NOT NULL, reason TEXT NOT NULL, object_version INTEGER NOT NULL
    )""",
)


__all__ = [
    "ALLOWED_TRANSITIONS", "APPROVED", "CANCELLED", "COMPLETED", "CommandReuseError",
    "EVIDENCE_ALLOWED_TRANSITIONS", "EVIDENCE_CANCELLED", "EVIDENCE_FULFILLED",
    "EVIDENCE_IN_PROGRESS", "EVIDENCE_OPEN", "EvidenceRequestConflictError",
    "GovernanceStore", "GovernanceStoreError", "REJECTED", "REVIEW_REQUIRED",
    "UnknownEvidenceRequestError", "UnknownWorkOrderError", "WorkflowConflictError",
]
