"""Structured validation report: violations, aggregate counts, JUnit export."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from graphrag.graph.validation.rules import RULES, Severity


@dataclass(frozen=True)
class Violation:
    rule_id: str
    record_kind: str  # document | entity | relation
    record_key: str   # stable, human-readable key (e.g. "ORG:Acme", "Acme-OWNS->Bolt")
    message: str = ""

    @property
    def severity(self) -> Severity:
        return RULES[self.rule_id].severity

    def to_dict(self) -> dict:
        r = RULES[self.rule_id]
        return {
            "rule_id": self.rule_id,
            "severity": r.severity.value,
            "shacl_ref": r.shacl_ref,
            "record_kind": self.record_kind,
            "record_key": self.record_key,
            "message": self.message or r.description,
        }


@dataclass
class ValidationReport:
    tenant: str
    source: str
    schema_version: str | None = None
    records_checked: int = 0
    violations: list[Violation] = field(default_factory=list)

    @property
    def blocking(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.BLOCKING]

    @property
    def conforms(self) -> bool:
        return not self.blocking

    def counts(self) -> dict:
        return {
            "by_rule": dict(Counter(v.rule_id for v in self.violations)),
            "by_severity": dict(Counter(v.severity.value for v in self.violations)),
            "by_tenant": {self.tenant: len(self.violations)} if self.violations else {},
            "by_source": {self.source: len(self.violations)} if self.violations else {},
        }

    def extend(self, other: "ValidationReport") -> None:
        self.records_checked += other.records_checked
        self.violations.extend(other.violations)

    def to_dict(self) -> dict:
        return {
            "tenant": self.tenant,
            "source": self.source,
            "schema_version": self.schema_version,
            "conforms": self.conforms,
            "records_checked": self.records_checked,
            "counts": self.counts(),
            "violations": [v.to_dict() for v in self.violations],
        }

    def to_junit_xml(self) -> str:
        """One testcase per rule; BLOCKING violations are failures, others are
        reported in system-out so CI shows them without failing the build."""
        suite = ET.Element(
            "testsuite",
            name=f"graph-validation[{self.source}]",
            tests=str(len(RULES)),
            failures=str(len({v.rule_id for v in self.blocking})),
        )
        by_rule: dict[str, list[Violation]] = {}
        for v in self.violations:
            by_rule.setdefault(v.rule_id, []).append(v)
        for rule_id, r in sorted(RULES.items()):
            case = ET.SubElement(suite, "testcase", classname=f"graph_validation.{r.target}", name=rule_id)
            hits = by_rule.get(rule_id, [])
            if not hits:
                continue
            body = "\n".join(f"{v.record_kind} {v.record_key}: {v.message or r.description}" for v in hits)
            if r.severity is Severity.BLOCKING:
                fail = ET.SubElement(case, "failure", message=f"{len(hits)} x {r.description}",
                                     type=r.severity.value)
                fail.text = body
            else:
                ET.SubElement(case, "system-out").text = f"[{r.severity.value}] {body}"
        return ET.tostring(suite, encoding="unicode")
