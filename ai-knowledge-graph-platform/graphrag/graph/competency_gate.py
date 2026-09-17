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
``energy_maintenance_review_competency_questions()`` reuses
``evals/energy_demo/sparql/maintenance_review.rq`` against an in-process
rdflib ``Graph`` via ``SPARQLBridge`` -- deterministic, needs no LLM or live
service, and is exercised in CI on every run (see
``tests/unit/test_ontology_migration_competency_gate.py``).

``aerospace_regulatory_competency_questions()``,
``automotive_iatf_competency_questions()`` and
``marketing_adtech_competency_questions()`` implement ``ontology/README.md``'s
long-documented aerospace/automotive/marketing examples as real, executable
Cypher (see ``evals/<tenant>/cypher/*.cypher``, one file per question,
grounded in that domain's ``config/ontologies/*.yml`` ``relation_rules``).
Unlike the Energy question, these have **not been run against a live Neo4j
graph in this repo** -- there is no Neo4j instance in this development
environment to run them against. They are structurally verified against the
real ontology's declared domain/range rules and unit-tested for wiring
(query text loads, placeholders substitute, ``expect`` is well-formed), but
per this repo's own evidence-level convention (see ``docs/roadmap.md``'s
"implemented and unit-tested" vs. "live-validated" distinction) they must
not be described as live-validated until someone runs
``run_competency_suite_async`` against a real Neo4j instance holding that
tenant's corpus and confirms a pass.

``run_competency_suite`` stayed synchronous and untouched for the Energy/
SPARQL path (``SPARQLBridge.query`` is sync). ``run_competency_suite_async``
is new, additive, and exists because the real Neo4j client
(``graphrag/graph/neo4j_client.py``'s ``Neo4jClient.run``) is async --
changing the existing sync signature would have been a breaking change to a
shipped, tested contract for no reason.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_EVALS_ROOT = Path(__file__).resolve().parents[2] / "evals"
_ENERGY_MAINTENANCE_QUERY_PATH = _EVALS_ROOT / "energy_demo" / "sparql" / "maintenance_review.rq"


def _load_cypher(tenant_dir: str, filename: str, substitutions: dict[str, str]) -> str:
    """Read a version-controlled ``.cypher`` file and substitute its
    ``{{PLACEHOLDER}}`` tokens with literal values -- the same pattern
    ``energy_maintenance_review_competency_questions`` already uses for
    ``{{BULLETIN_ID}}``, kept consistent rather than introducing a second,
    incompatible parameter-passing convention (``CompetencyQuestion.query``
    is a single literal string; there is no separate params channel)."""
    text = (_EVALS_ROOT / tenant_dir / "cypher" / filename).read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace("{{" + key + "}}", value)
    return text


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


async def run_competency_suite_async(
    questions: list[CompetencyQuestion],
    query_fn: Callable[[str], Awaitable[Any]],
) -> CompetencyGateReport:
    """Async counterpart to ``run_competency_suite``, for a ``query_fn`` that
    must ``await`` (a real Neo4j session, an HTTP-backed SPARQL endpoint,
    etc.). Same contract and same isolate-one-failure-from-the-rest
    behavior; kept as a separate function rather than making the original
    async, so every existing sync caller (Energy/SPARQL, already shipped and
    tested) is completely unaffected."""
    results: list[CompetencyResult] = []
    for question in questions:
        try:
            outcome = await query_fn(question.query)
            results.append(CompetencyResult(id=question.id, passed=bool(question.expect(outcome))))
        except Exception as exc:  # noqa: BLE001 - isolate one question's failure from the rest
            results.append(CompetencyResult(id=question.id, passed=False, error=str(exc)))
    failed_ids = [r.id for r in results if not r.passed]
    if failed_ids:
        log.warning("competency_gate.questions_failed", failed_ids=failed_ids)
    return CompetencyGateReport(passed=not failed_ids, failed_ids=failed_ids, results=results)


def neo4j_cypher_query_fn(client: Any) -> Callable[[str], Awaitable[list[dict]]]:
    """Adapt a ``Neo4jClient`` (``graphrag/graph/neo4j_client.py``) into the
    ``query_fn: Callable[[str], Awaitable[Any]]`` contract
    ``run_competency_suite_async`` expects. ``client`` is typed ``Any``
    rather than imported as ``Neo4jClient`` to avoid this module -- imported
    by ``ontology_registry.py``, which many lightweight unit tests construct
    without a live Neo4j driver -- pulling in the Neo4j driver dependency
    just to define this adapter; any object exposing an
    ``async def run(self, cypher: str) -> list[dict]`` matching
    ``Neo4jClient.run``'s shape works. The literal Cypher text already has
    every ``{{PLACEHOLDER}}`` substituted by the time it reaches here (see
    ``_load_cypher``), so no separate bind-parameter dict is threaded
    through -- consistent with the single-string ``query_fn`` contract every
    other competency question (Energy/SPARQL included) already uses."""
    async def _run(cypher: str) -> list[dict]:
        return await client.run(cypher)
    return _run


def aerospace_regulatory_competency_questions(
    *, directive: str = "AD-2024-01-02", tenant: str = "aerospace",
) -> list[CompetencyQuestion]:
    """The three aerospace competency questions ``ontology/README.md`` has
    documented as prose examples since before this module existed, now
    executable: SUPERSEDES/transitivity, MANDATED_BY, and APPLIES_TO domain/
    range (``config/ontologies/aerospace_regulatory.yml``). Not live-
    validated in this repo -- see the module docstring."""
    substitutions = {"TENANT": tenant, "DIRECTIVE": directive}

    def _returns_rows_with(field_name: str) -> Callable[[list[dict]], bool]:
        def expect(rows: list[dict]) -> bool:
            return len(rows) > 0 and all(field_name in row for row in rows)
        return expect

    return [
        CompetencyQuestion(
            id="aerospace-regulatory/directive-supersession",
            query=_load_cypher("aerospace_regulatory", "directive_supersession.cypher", substitutions),
            expect=_returns_rows_with("supersedingDirective"),
        ),
        CompetencyQuestion(
            id="aerospace-regulatory/mandating-authority",
            query=_load_cypher("aerospace_regulatory", "mandating_authority.cypher", substitutions),
            expect=_returns_rows_with("mandatingAuthority"),
        ),
        CompetencyQuestion(
            id="aerospace-regulatory/applicable-aircraft-type",
            query=_load_cypher("aerospace_regulatory", "applicable_aircraft_type.cypher", substitutions),
            expect=_returns_rows_with("aircraftType"),
        ),
    ]


def automotive_iatf_competency_questions(
    *, supplier: str = "PlastiAuto SRL", tenant: str = "automotive",
) -> list[CompetencyQuestion]:
    """The automotive competency questions ``ontology/README.md`` documents,
    grounded in the corpus's own C03/C05 ground-truth contradictions
    (reevaluation frequency, CRITICAL-classification consequences --
    ``config/ontologies/automotive_iatf.yml``'s header comment). Not live-
    validated in this repo -- see the module docstring."""
    substitutions = {"TENANT": tenant, "SUPPLIER": supplier}

    def _returns_rows_with(field_name: str) -> Callable[[list[dict]], bool]:
        def expect(rows: list[dict]) -> bool:
            return len(rows) > 0 and all(field_name in row for row in rows)
        return expect

    return [
        CompetencyQuestion(
            id="automotive-iatf/supplier-classification",
            query=_load_cypher("automotive_iatf", "supplier_classification.cypher", substitutions),
            expect=_returns_rows_with("classification"),
        ),
        CompetencyQuestion(
            id="automotive-iatf/reevaluation-frequency",
            query=_load_cypher("automotive_iatf", "reevaluation_frequency.cypher", substitutions),
            expect=_returns_rows_with("reevaluationFrequency"),
        ),
    ]


def marketing_adtech_competency_questions(
    *, advertiser: str = "Nova Beverages Global", regulation: str = "GDPR", tenant: str = "marketing",
) -> list[CompetencyQuestion]:
    """The marketing competency questions ``ontology/README.md`` documents:
    excluded-category negative-knowledge modeling, and a multi-hop consent
    requirement chained through a privacy regulation
    (``config/ontologies/marketing_adtech.yml``). Not live-validated in this
    repo -- see the module docstring."""
    substitutions = {"TENANT": tenant, "ADVERTISER": advertiser, "REGULATION": regulation}

    def _returns_rows_with(field_name: str) -> Callable[[list[dict]], bool]:
        def expect(rows: list[dict]) -> bool:
            return len(rows) > 0 and all(field_name in row for row in rows)
        return expect

    return [
        CompetencyQuestion(
            id="marketing-adtech/excluded-categories",
            query=_load_cypher("marketing_adtech", "excluded_categories.cypher", substitutions),
            expect=_returns_rows_with("excludedCategory"),
        ),
        CompetencyQuestion(
            id="marketing-adtech/consent-requirement",
            query=_load_cypher("marketing_adtech", "consent_requirement.cypher", substitutions),
            expect=_returns_rows_with("consentSignal"),
        ),
    ]


_DOMAIN_ONTOLOGY_PATHS: dict[str, Path] = {
    "aerospace-regulatory": Path(__file__).resolve().parents[2] / "config" / "ontologies" / "aerospace_regulatory.yml",
    "automotive-iatf": Path(__file__).resolve().parents[2] / "config" / "ontologies" / "automotive_iatf.yml",
    "marketing-adtech": Path(__file__).resolve().parents[2] / "config" / "ontologies" / "marketing_adtech.yml",
}

_CYPHER_RELATION_RE = re.compile(r"relation:\s*'([A-Z_]+)'")
_CYPHER_TYPE_RE = re.compile(r"type:\s*'([A-Z_]+)'")


def domains_with_static_ontology_checks() -> list[str]:
    """Domain ids ``check_competency_questions_against_ontology`` can
    actually check -- those with a ``config/ontologies/*.yml`` registered in
    ``_DOMAIN_ONTOLOGY_PATHS``. Energy is intentionally excluded: its schema
    drift is checked separately, by ``graphrag.semantic_model`` against
    ``ontology/models/energy-asset-intelligence.yaml``."""
    return sorted(_DOMAIN_ONTOLOGY_PATHS)


def check_competency_questions_against_ontology(domain_id: str) -> list[str]:
    """Static regression check for roadmap bullet 4 ("run these questions as
    regression gates whenever the ontology or mapping contract changes"),
    for the domains where that "run" can't yet mean live Neo4j execution
    (see the module docstring): scans each of ``domain_id``'s competency
    questions for the relation/type names their Cypher literally references
    (``relation: 'NAME'`` / ``type: 'NAME'``, the one syntactic shape every
    question in ``evals/<domain>/cypher/`` uses) and verifies each name
    still exists in that domain's *current*
    ``config/ontologies/<domain>.yml`` -- catching "someone renamed or
    removed a type/relation this competency question still expects"
    immediately, from the YAML alone, with no live graph required.

    Returns a list of problem strings; empty means every name a question
    references is still declared in the ontology. A domain with no
    registered ontology path (Energy, whose schema lives in
    ``ontology/models/*.yaml`` and is drift-checked separately by
    ``graphrag.semantic_model``) or no competency questions returns an
    empty list rather than raising.

    This is deliberately narrower than a full "model-change impact
    analysis" -- see ``graphrag.graph.ontology_migration.
    find_affected_competency_questions`` for the complementary check that
    runs the other direction (given a migration's diff, which questions
    does it affect) and documents the same "mappings/shapes/generated-
    constraints/API-contracts are not covered" boundary.
    """
    from graphrag.graph.domain_ontology import get_type_hierarchy_pairs, load_domain_ontology

    ontology_path = _DOMAIN_ONTOLOGY_PATHS.get(domain_id)
    factory = competency_questions_by_domain().get(domain_id)
    if ontology_path is None or factory is None:
        return []
    ontology = load_domain_ontology(ontology_path)
    known_types = {child for child, _parent in get_type_hierarchy_pairs(ontology)} | {
        parent for _child, parent in get_type_hierarchy_pairs(ontology)
    }
    known_relations = set((ontology.get("relation_rules") or {}).keys())

    problems: list[str] = []
    for question in factory():
        for relation_name in _CYPHER_RELATION_RE.findall(question.query):
            if relation_name not in known_relations:
                problems.append(
                    f"{question.id}: references relation '{relation_name}', "
                    f"not declared in {ontology_path.name}'s relation_rules",
                )
        for type_name in _CYPHER_TYPE_RE.findall(question.query):
            if type_name not in known_types:
                problems.append(
                    f"{question.id}: references type '{type_name}', "
                    f"not declared in {ontology_path.name}'s type_hierarchy",
                )
    return problems


def competency_questions_by_domain() -> dict[str, Callable[[], list[CompetencyQuestion]]]:
    """Registry mapping each domain-ontology tenant id (``ontology.id`` in
    ``config/ontologies/*.yml``, e.g. ``aerospace-regulatory``) to its
    competency-question factory. Used by
    ``graphrag.graph.ontology_migration``'s impact analysis to look up which
    questions could be affected by a change to a given domain's ontology,
    and by anything that wants to run "every domain's competency suite"
    without hand-maintaining a second list."""
    return {
        "energy-asset-intelligence": energy_maintenance_review_competency_questions,
        "aerospace-regulatory": aerospace_regulatory_competency_questions,
        "automotive-iatf": automotive_iatf_competency_questions,
        "marketing-adtech": marketing_adtech_competency_questions,
    }
