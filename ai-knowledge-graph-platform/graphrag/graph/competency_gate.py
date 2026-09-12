"""Executable competency-question gate for ontology migrations.

Why this exists
----------------
``ontology/README.md`` already documents "competency questions" as a
governance concept: a question the ontology must be able to answer, tied to
a real, tracked golden-eval query rather than invented ad hoc, with a
process ask that a non-additive ontology change should state which
competency questions were re-run. Nothing enforced that ask -- it was an
honor-system line in a PR description, not a gate a migration could
actually fail. This module makes it executable: a pluggable suite runner
(``run_competency_suite``) plus a wire-in point at
``OntologyRegistry.apply_ontology_migration`` (see
``graphrag/graph/ontology_migration.py`` for the ``OntologyMigrationBlockedError``
this raises) that refuses to touch the graph when the suite fails, unless
explicitly forced.

Scope, honestly
---------------
Only one concrete, fully-executable implementation ships here:
``energy_maintenance_review_competency_questions()``, which reuses
``evals/energy_demo/sparql/maintenance_review.rq`` against an in-process
rdflib ``Graph`` via ``SPARQLBridge`` -- the one competency question in this
repo today that is deterministic and needs no LLM or live service. Other
tenants' competency questions (``ontology/README.md``'s aerospace/
automotive/marketing examples) are Cypher/golden-set-based against a live
Neo4j graph and real retrieval pipeline; wiring those in is real follow-up
work, not attempted here -- ``run_competency_suite`` is deliberately
pluggable (``query_fn: Callable[[str], Any]``) so that work is additive, not
a rewrite, when it happens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_ENERGY_MAINTENANCE_QUERY_PATH = (
    Path(__file__).resolve().parents[2] / "evals" / "energy_demo" / "sparql" / "maintenance_review.rq"
)


@dataclass(frozen=True)
class CompetencyQuestion:
    """One executable ontology competency question.

    ``query`` is the exact query text to run -- read from a version-controlled
    file wherever possible (see ``ontology/README.md``'s "not invented, tied
    to golden-eval suites" rule), not composed inline here. ``expect``
    receives whatever ``query_fn`` returns for this query and reports
    whether the result satisfies the question.
    """

    id: str
    query: str
    expect: Callable[[Any], bool]


@dataclass(frozen=True)
class CompetencyResult:
    id: str
    passed: bool
    error: str = ""


@dataclass(frozen=True)
class CompetencyGateReport:
    passed: bool
    failed_ids: list[str] = field(default_factory=list)
    results: list[CompetencyResult] = field(default_factory=list)


def run_competency_suite(
    questions: list[CompetencyQuestion],
    query_fn: Callable[[str], Any],
) -> CompetencyGateReport:
    """Run every question's query through ``query_fn`` and check its ``expect``.

    ``query_fn`` is pluggable so this isn't tied to one query language or
    backend -- ``SPARQLBridge.query`` for the Energy tenant's RDF graph
    today, a Cypher runner against Neo4j for other tenants tomorrow (not
    built here; see the module docstring).

    A question whose query raises is recorded as a failure, not propagated:
    one broken competency question must not crash the whole gate and hide
    every other result. The caller sees exactly which id(s) failed and why.
    """
    results: list[CompetencyResult] = []
    for question in questions:
        try:
            outcome = query_fn(question.query)
            results.append(CompetencyResult(id=question.id, passed=bool(question.expect(outcome))))
        except Exception as exc:  # noqa: BLE001 - isolate one question's failure from the rest
            results.append(CompetencyResult(id=question.id, passed=False, error=str(exc)))
    failed_ids = [r.id for r in results if not r.passed]
    if failed_ids:
        log.warning("competency_gate.questions_failed", failed_ids=failed_ids)
    return CompetencyGateReport(passed=not failed_ids, failed_ids=failed_ids, results=results)


def energy_maintenance_review_competency_questions(
    *, bulletin_id: str = "MFG-GBX-17-R2",
) -> list[CompetencyQuestion]:
    """The one competency question this repo can run fully automatically.

    Reuses ``evals/energy_demo/sparql/maintenance_review.rq`` unchanged, the
    same query and substitution ``graphrag/domains/energy/demo.py``'s
    ``EnergyDemoService.answer`` runs for the live "maintenance_review"
    demo question -- this is not a new, separately-maintained query.
    """
    query_text = _ENERGY_MAINTENANCE_QUERY_PATH.read_text(encoding="utf-8").replace(
        "{{BULLETIN_ID}}", bulletin_id,
    )

    def _finds_at_least_one_open_asset(rows: list[dict]) -> bool:
        return len(rows) > 0 and all("asset" in row for row in rows)

    return [
        CompetencyQuestion(
            id="energy-demo/maintenance-review-finds-open-assets",
            query=query_text,
            expect=_finds_at_least_one_open_asset,
        ),
    ]
