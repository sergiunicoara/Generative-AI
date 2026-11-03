"""Evaluation prototype: export published knowledge as an Open Knowledge Format bundle.

Open Knowledge Format (OKF) v0.2 is Google Cloud's markdown-bundle format for
agent-readable knowledge (spec: GoogleCloudPlatform/knowledge-catalog,
``okf/SPEC.md``; checked 2026-11-03).  Its rules relevant here: one UTF-8
markdown file per concept with YAML frontmatter, ``type`` is the only required
key, ``index.md`` and ``log.md`` are reserved names, links between concepts are
plain (untyped) markdown links, and consumers must accept unknown keys.

What this prototype preserves and why:

* **Typed edges survive** in an extension key (``x-graphrag-relations``),
  allowed by the spec, alongside plain markdown links for OKF readers.
* **Trust fields are never invented.**  ``verified`` is written only from a
  recorded human review (``reviewed_by`` + ``review_ref``, not a ``system:``
  actor); ``generated`` only from recorded constraint provenance;
  ``stale_after`` only from a human-set ``review_due``.
* Only what the caller may see is exported (the evaluator's visibility rules).

It is an evaluation, not a supported interface: the repository's licence was
not confirmed, and no OKF consumer was run against the output.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import yaml

from graphrag.context_graph.knowledge import KnowledgeItem
from graphrag.context_graph.knowledge_evaluation import visible_items
from graphrag.enterprise.models import AccessContext

RESERVED = {"index.md", "log.md"}
_STATUS = {"published": "stable", "withdrawn": "deprecated"}


def _path(item_id: str) -> str:
    return f"concepts/{item_id}.md"


def _frontmatter(item: KnowledgeItem, known: set[str]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "type": "knowledge_item",
        "title": item.title,
        "description": item.statement[:500],
        "resource": f"urn:graphrag:knowledge:{item.tenant}:{item.key}",
        "tags": [item.modality.value, item.authority.value, item.subject_key],
        "status": _STATUS.get(item.status.value, "draft"),
        "sources": [{"resource": s.source_id, **({"title": s.locator} if s.locator else {})} for s in item.sources],
    }
    if item.constraint_provenance is not None:
        meta["generated"] = {"by": item.constraint_provenance.authored_by, "at": item.recorded_at.isoformat()}
    if item.reviewed_by and item.review_ref and not item.reviewed_by.startswith("system:"):
        meta["verified"] = [{"by": f"human:{item.reviewed_by}", "at": item.recorded_at.isoformat()}]
    if item.review_due is not None:
        meta["stale_after"] = item.review_due.isoformat()
    relations = [{"type": kind, "target": _path(target)}
                 for kind, targets in (("supersedes", item.supersedes), ("overrides", item.overrides))
                 for target in targets if target in known]
    if relations:
        meta["x-graphrag-relations"] = relations
    return meta


def export_okf(items: Iterable[KnowledgeItem], *, tenant: str, access: AccessContext) -> dict[str, str]:
    """Bundle files (relative path -> content) for what ``access`` may read."""
    visible = visible_items(items, tenant=tenant, access=access)
    known = {item.id for item in visible}
    files: dict[str, str] = {}
    for item in visible:
        meta = _frontmatter(item, known)
        links = "".join(f"\n- {rel['type']}: [{rel['target'].split('/')[-1][:-3]}]({rel['target'].split('/')[-1]})"
                        for rel in meta.get("x-graphrag-relations", ()))
        body = f"# {item.title}\n\n{item.statement}\n" + (f"\n## Related{links}\n" if links else "")
        files[_path(item.id)] = "---\n" + yaml.safe_dump(meta, sort_keys=True, allow_unicode=True) + "---\n\n" + body
    files["index.md"] = "# Knowledge bundle\n\n" + "".join(
        f"- [{item.title}]({_path(item.id)})\n" for item in visible)
    return files


def _parse(content: str) -> dict[str, Any] | None:
    if not content.startswith("---\n"):
        return None
    head, _, _ = content[4:].partition("\n---\n")
    return yaml.safe_load(head) or {}


def conformance_issues(files: Mapping[str, str]) -> list[str]:
    """Checks against the OKF v0.2 rules this exporter relies on."""
    issues = []
    for path, content in sorted(files.items()):
        name = path.rsplit("/", 1)[-1]
        if not path.endswith(".md"):
            issues.append(f"{path}: not a markdown file")
            continue
        if name in RESERVED:
            if path != name:
                issues.append(f"{path}: reserved name used for a concept")
            continue
        try:
            content.encode("utf-8")
        except UnicodeEncodeError:
            issues.append(f"{path}: not UTF-8")
        meta = _parse(content)
        if meta is None or not meta.get("type"):
            issues.append(f"{path}: frontmatter must carry type")
            continue
        if meta.get("status", "stable") not in ("draft", "stable", "deprecated"):
            issues.append(f"{path}: status must be draft, stable or deprecated")
        if any("resource" not in source for source in meta.get("sources", ())):
            issues.append(f"{path}: every source needs a resource")
        for event in meta.get("verified", ()):
            if not {"by", "at"} <= set(event):
                issues.append(f"{path}: verified events need by and at")
    return issues


def typed_edges(files: Mapping[str, str]) -> set[tuple[str, str, str]]:
    """(source path, relation type, target path) read back from a bundle."""
    edges = set()
    for path, content in files.items():
        meta = _parse(content) or {}
        for rel in meta.get("x-graphrag-relations", ()):
            edges.add((path, rel["type"], rel["target"]))
    return edges


__all__ = ["conformance_issues", "export_okf", "typed_edges"]
