"""Curating and publishing a knowledge pack without code access.

The working pack for a tenant and domain lives in the durable governance
store as append-only revisions (``knowledge_pack_revisions``).  Every change
records who made it -- always the authenticated subject passed in by the
caller, never a request field -- and is guarded by a compare-and-swap on the
revision number, so two curators cannot silently overwrite each other.

* ``import_pack`` seeds (or re-seeds) the working pack from the deployed,
  version-controlled pack file ``config/knowledge_packs/<domain>.yml``.
* ``review`` approves or withdraws one item or exception, producing its next
  version through the same library helpers as scripted curation
  (``approve_item``, ``approve_exception``, ...), so the contract checks are
  identical: a reviewer cannot publish an item without sources, or an
  extracted constraint without human constraint provenance.
* ``publish`` runs the unchanged release gate (``publish_pack``) on one
  working revision with the deployed regression cases
  ``evals/knowledge_eval/<domain>_cases.json``.  Cases never come from the
  request, so a caller cannot weaken the gate it is being checked by.

A working revision is never served; only a gated release is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from graphrag.context_graph.knowledge import KnowledgeException, KnowledgeItem, PublicationStatus
from graphrag.context_graph.knowledge_release import (
    KnowledgePack, approve_exception, approve_item, load_cases, load_pack, publish_pack, replace_entry,
    review_queue, withdraw_exception, withdraw_item,
)
from graphrag.core.tenancy import require_tenant

ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = ROOT / "config" / "knowledge_packs"
CASES_DIR = ROOT / "evals" / "knowledge_eval"

EntryKind = Literal["items", "exceptions"]
ReviewAction = Literal["approve", "withdraw"]

#: Statuses an approval may start from.  Re-approving a published entry or
#: resurrecting a withdrawn one is a new draft, not a review decision.
_APPROVABLE = frozenset({PublicationStatus.DRAFT, PublicationStatus.IN_REVIEW, PublicationStatus.APPROVED})


class CurationError(Exception):
    """A curation request that cannot be applied.

    ``code`` is one of ``not_found``, ``invalid`` or ``conflict``.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def review_entry(
    pack: KnowledgePack, *, kind: EntryKind, entry_id: str, action: ReviewAction, actor: str,
    reference: str, publish: bool = True,
) -> tuple[KnowledgePack, dict[str, Any]]:
    """Apply one review decision; return the new pack and the change record."""
    if not actor:
        raise CurationError("invalid", "a review decision needs an authenticated identity")
    if not reference:
        raise CurationError("invalid", "a review decision needs a review or approval reference")
    entries = pack.items if kind == "items" else pack.exceptions
    entry = next((e for e in entries if e.id == entry_id), None)
    if entry is None:
        raise CurationError("not_found", f"no {kind[:-1]} {entry_id!r} in this pack")
    if action == "approve":
        if entry.status not in _APPROVABLE:
            raise CurationError("conflict", f"{entry.key} is {entry.status.value}; only draft, in-review "
                                            "or approved entries can be approved")
        if entry.status == PublicationStatus.APPROVED and not publish:
            raise CurationError("conflict", f"{entry.key} is already approved")
    elif entry.status == PublicationStatus.WITHDRAWN:
        raise CurationError("conflict", f"{entry.key} is already withdrawn")
    try:
        if isinstance(entry, KnowledgeItem):
            updated: KnowledgeItem | KnowledgeException = (
                approve_item(entry, reviewed_by=actor, review_ref=reference, publish=publish)
                if action == "approve" else withdraw_item(entry, reviewed_by=actor, review_ref=reference))
        else:
            updated = (
                approve_exception(entry, granted_by=actor, approval_ref=reference, publish=publish)
                if action == "approve" else withdraw_exception(entry, approval_ref=reference))
    except (ValidationError, ValueError) as exc:
        raise CurationError("invalid", f"{entry.key} cannot be {action}d: {_first_error(exc)}") from exc
    change = {
        "action": action, "kind": kind, "from": entry.key, "to": updated.key,
        "status": updated.status.value, "reference": reference,
    }
    return replace_entry(pack, updated), change


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        return str(errors[0].get("msg", exc)) if errors else str(exc)
    return str(exc)


class KnowledgeCurationService:
    def __init__(self, store, *, pack_dir: Path = PACK_DIR, cases_dir: Path = CASES_DIR) -> None:
        self._store = store
        self._pack_dir = pack_dir
        self._cases_dir = cases_dir

    async def working(self, tenant: str, domain: str) -> tuple[int, KnowledgePack]:
        record = await self._store.knowledge_pack_revision(require_tenant(tenant), domain)
        if record is None:
            raise CurationError("not_found", f"no working knowledge pack for domain {domain!r}; import it first")
        return int(record["revision"]), KnowledgePack.model_validate(record["pack"])

    async def summary(self, tenant: str, domain: str) -> dict[str, Any]:
        revision, pack = await self.working(tenant, domain)
        active = await self._store.active_knowledge_release(tenant, domain)
        return {
            "domain": domain, "revision": revision, "pack_version": pack.pack_version,
            "disclaimer": pack.disclaimer,
            "items": [_entry_row(i) for i in pack.items],
            "exceptions": [_entry_row(e) for e in pack.exceptions],
            "review_queue": review_queue(pack),
            "active_release": None if active is None else {
                key: active.get(key) for key in ("release_id", "version", "content_hash", "published_at",
                                                 "published_by", "pack_version")
            },
        }

    async def history(self, tenant: str, domain: str) -> list[dict[str, Any]]:
        return await self._store.knowledge_pack_history(require_tenant(tenant), domain)

    async def import_pack(self, tenant: str, domain: str, *, actor: str, expected_revision: int) -> dict[str, Any]:
        tenant = require_tenant(tenant)
        path = self._pack_dir / f"{domain}.yml"
        if not path.is_file():
            raise CurationError("not_found", f"no deployed knowledge pack for domain {domain!r}")
        try:
            pack = load_pack(path)
        except ValidationError as exc:
            raise CurationError("invalid", f"the deployed pack is invalid: {_first_error(exc)}") from exc
        if pack.tenant != tenant or pack.domain != domain:
            # Another tenant's pack is indistinguishable from no pack.
            raise CurationError("not_found", f"no deployed knowledge pack for domain {domain!r}")
        change = {"action": "import", "source": f"config/knowledge_packs/{domain}.yml",
                  "pack_version": pack.pack_version}
        return await self._save(tenant, domain, pack, actor=actor, change=change,
                                expected_revision=expected_revision)

    async def review(
        self, tenant: str, domain: str, *, kind: EntryKind, entry_id: str, action: ReviewAction, actor: str,
        reference: str, expected_revision: int, publish: bool = True,
    ) -> dict[str, Any]:
        revision, pack = await self.working(tenant, domain)
        if revision != expected_revision:
            raise CurationError("conflict", f"the working pack is at revision {revision}, not {expected_revision}")
        updated, change = review_entry(pack, kind=kind, entry_id=entry_id, action=action, actor=actor,
                                       reference=reference, publish=publish)
        return await self._save(tenant, domain, updated, actor=actor, change=change,
                                expected_revision=expected_revision)

    async def publish(self, tenant: str, domain: str, *, actor: str, expected_revision: int) -> dict[str, Any]:
        """Gate and publish one working revision.  Raises
        ``KnowledgePublicationError`` on a gate failure, leaving the active
        release unchanged."""
        if not actor:
            raise CurationError("invalid", "publication needs an authenticated identity")
        revision, pack = await self.working(tenant, domain)
        if revision != expected_revision:
            raise CurationError("conflict", f"the working pack is at revision {revision}, not {expected_revision}")
        cases_path = self._cases_dir / f"{domain}_cases.json"
        cases = load_cases(cases_path, domain=domain) if cases_path.is_file() else []
        return await publish_pack(self._store, pack, published_by=actor, cases=cases)

    async def _save(self, tenant, domain, pack: KnowledgePack, *, actor: str, change: dict[str, Any],
                    expected_revision: int) -> dict[str, Any]:
        from graphrag.domains.energy.governance_store import WorkflowConflictError

        try:
            record = await self._store.save_knowledge_pack_revision(
                tenant, domain, pack.model_dump(mode="json"), changed_by=actor, change=change,
                expected_revision=expected_revision)
        except WorkflowConflictError as exc:
            raise CurationError("conflict", str(exc)) from exc
        return {key: record[key] for key in ("domain", "revision", "changed_by", "change", "created_at")}


def _entry_row(entry: KnowledgeItem | KnowledgeException) -> dict[str, Any]:
    row = {"id": entry.id, "version": entry.version, "key": entry.key, "status": entry.status.value}
    if isinstance(entry, KnowledgeItem):
        row.update(title=entry.title, modality=entry.modality.value, reviewed_by=entry.reviewed_by)
    else:
        row.update(kind=entry.kind, target_item_ids=list(entry.target_item_ids), granted_by=entry.granted_by)
    return row


__all__ = ["CurationError", "KnowledgeCurationService", "review_entry"]
