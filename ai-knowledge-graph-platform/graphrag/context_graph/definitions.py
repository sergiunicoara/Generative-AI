"""Governed business definitions: one versioned definition per concept.

A business definition is the shared meaning of a term people and agents use
("active customer", "which price list applies when"): an owner, a
plain-language definition, a precision level, the executable form where one
exists, the sources it depends on and a review date.  Definitions travel in
the same knowledge release as the items that use them and resolve through the
same rules: the tenant from the token, published status, the single latest
version per id, and the caller's access.  A decision lists the definition
versions its applicable items used, so a definition change can be traced to
the decisions that relied on the old version (see the decision log).

Precision is explicit so the registry can record that a definition is
deliberately loose instead of forcing false precision:

* ``exact``: there is one executable form and it is authoritative;
* ``bounded``: an executable form approximates it within stated limits;
* ``loose``: intentionally informal; an executable form would be false
  precision and is refused.

Executable SQL or SPARQL text is stored and reviewed, never executed here.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from graphrag.context_graph.knowledge import EvidenceRef, Predicate, PublicationStatus
from graphrag.enterprise.models import AccessContext, DocumentAccessPolicy


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutableForm(_Strict):
    kind: Literal["sql", "sparql", "predicate"]
    text: str = Field(default="", max_length=4000)
    predicate: Predicate | None = None

    @model_validator(mode="after")
    def _one_form(self):
        if self.kind == "predicate" and (self.predicate is None or self.text):
            raise ValueError("a predicate form carries a predicate and no text")
        if self.kind != "predicate" and (not self.text.strip() or self.predicate is not None):
            raise ValueError(f"a {self.kind} form carries query text and no predicate")
        return self


class BusinessDefinition(_Strict):
    id: str = Field(pattern=r"^DEF-[A-Z0-9][A-Z0-9_\-]{0,63}$")
    version: int = Field(ge=1)
    tenant: str = Field(min_length=1, max_length=128)
    term: str = Field(min_length=1, max_length=200)
    owner: str = Field(min_length=1, max_length=256)
    definition: str = Field(min_length=1, max_length=4000)
    precision: Literal["exact", "bounded", "loose"]
    precision_note: str = Field(default="", max_length=1000)
    executable: ExecutableForm | None = None
    depends_on: tuple[str, ...] = ()
    sources: tuple[EvidenceRef, ...] = ()
    review_due: datetime | None = None
    status: PublicationStatus = PublicationStatus.DRAFT
    access: DocumentAccessPolicy = Field(default_factory=DocumentAccessPolicy)
    recorded_at: datetime

    @field_validator("recorded_at", "review_due")
    @classmethod
    def _utc(cls, value):
        if value is None:
            return None
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)

    @model_validator(mode="after")
    def _precision(self):
        if self.precision == "exact" and self.executable is None:
            raise ValueError(f"{self.id}: an exact definition needs its executable form")
        if self.precision == "loose" and self.executable is not None:
            raise ValueError(f"{self.id}: a loose definition must not carry an executable form (false precision)")
        if self.precision == "bounded" and not self.precision_note.strip():
            raise ValueError(f"{self.id}: a bounded definition states its limits in precision_note")
        if self.status == PublicationStatus.PUBLISHED and not self.sources:
            raise ValueError(f"{self.id}: a published definition cites at least one source")
        if self.id in self.depends_on:
            raise ValueError(f"{self.id}: a definition cannot depend on itself")
        return self

    @property
    def key(self) -> str:
        return f"{self.id}@v{self.version}"


def visible_definitions(
    definitions: Iterable[BusinessDefinition], *, tenant: str, access: AccessContext,
) -> list[BusinessDefinition]:
    """Latest version per id, then published and readable, as for knowledge items."""
    from graphrag.context_graph.knowledge_evaluation import can_read

    latest: dict[str, BusinessDefinition] = {}
    for definition in definitions:
        if definition.tenant != tenant:
            continue
        current = latest.get(definition.id)
        if current is None or definition.version > current.version:
            latest[definition.id] = definition
    return [d for _, d in sorted(latest.items())
            if d.status == PublicationStatus.PUBLISHED and can_read(d.access, access)]


def definition_issues(definitions: Iterable[BusinessDefinition], *, tenant: str) -> list[tuple[str, str, str]]:
    """(code, path, message) gate findings over a pack's definitions."""
    definitions = list(definitions)
    issues: list[tuple[str, str, str]] = []
    published = [d for d in definitions if d.status == PublicationStatus.PUBLISHED]
    known = {d.id for d in definitions}
    for definition in definitions:
        if definition.tenant != tenant:
            issues.append(("tenant_mismatch", definition.key,
                           f"{definition.key}: tenant {definition.tenant!r} differs from pack tenant {tenant!r}"))
    by_term: dict[str, set[str]] = {}
    for definition in published:
        by_term.setdefault(definition.term.strip().lower(), set()).add(definition.id)
        for dependency in definition.depends_on:
            if dependency not in known:
                issues.append(("definition_unknown", f"{definition.key}/depends_on",
                               f"{definition.key}: depends on unknown definition {dependency}"))
    for term, ids in sorted(by_term.items()):
        if len(ids) > 1:
            issues.append(("duplicate_definition_term", f"definitions/{term}",
                           f"term {term!r} has more than one published definition: {sorted(ids)}"))
    return issues


__all__ = ["BusinessDefinition", "ExecutableForm", "definition_issues", "visible_definitions"]
