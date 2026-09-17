"""Competency-question gate for ontology migrations.

Covers graphrag/graph/competency_gate.py's suite runner plus its wire-in at
OntologyRegistry.apply_ontology_migration: a migration proceeds when the
suite passes, is blocked (no graph write attempted) when it fails, and an
explicit force=True override still proceeds but is logged with the failing
question ids.

TestEnergyCompetencyQuestionIsReallyExecutable is the one integration-style
case: it runs the concrete Energy-tenant implementation
(energy_maintenance_review_competency_questions) against
EnergyDemoService's own real, self-built graph via SPARQLBridge -- proving
this is genuinely wired end to end, not just plumbing exercised with fakes.

TestDomainCompetencyQuestionsWiring, TestRunCompetencySuiteAsync and
TestNeo4jCypherQueryFn cover the aerospace/automotive/marketing Cypher
questions and the async Neo4j-backed suite runner. These are unit/wiring
tests only -- there is no live Neo4j instance in this environment, so
(unlike the Energy/SPARQL case above) they cannot prove the Cypher itself
returns real rows against a real graph. See
graphrag/graph/competency_gate.py's module docstring for that evidence-level
distinction.

TestFindAffectedCompetencyQuestions covers
graphrag/graph/ontology_migration.py's model-change impact analysis: which
competency questions reference a name a migration removes or renames away
from.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.competency_gate import (
    CompetencyGateReport,
    CompetencyQuestion,
    aerospace_regulatory_competency_questions,
    automotive_iatf_competency_questions,
    check_competency_questions_against_ontology,
    competency_questions_by_domain,
    energy_maintenance_review_competency_questions,
    marketing_adtech_competency_questions,
    neo4j_cypher_query_fn,
    run_competency_suite,
    run_competency_suite_async,
)
from graphrag.graph.ontology_migration import (
    MigrationReport,
    OntologyMigrationBlockedError,
    find_affected_competency_questions,
)
from graphrag.graph.ontology_registry import OntologyRegistry
from graphrag.graph.sparql_bridge import SPARQLBridge


# ── run_competency_suite ─────────────────────────────────────────────────────

class TestRunCompetencySuite:
    def test_all_questions_passing_yields_a_passed_report(self):
        questions = [
            CompetencyQuestion(id="q1", query="SELECT 1", expect=lambda r: r == 1),
            CompetencyQuestion(id="q2", query="SELECT 2", expect=lambda r: r == 2),
        ]
        report = run_competency_suite(questions, query_fn=lambda q: int(q.rsplit(" ", 1)[-1]))
        assert report.passed is True
        assert report.failed_ids == []
        assert [r.id for r in report.results] == ["q1", "q2"]

    def test_one_failing_question_fails_the_whole_report_but_names_it(self):
        questions = [
            CompetencyQuestion(id="q1", query="SELECT 1", expect=lambda r: r == 1),
            CompetencyQuestion(id="q2", query="SELECT 2", expect=lambda r: r == 999),
        ]
        report = run_competency_suite(questions, query_fn=lambda q: int(q.rsplit(" ", 1)[-1]))
        assert report.passed is False
        assert report.failed_ids == ["q2"]

    def test_a_raising_query_is_recorded_as_a_failure_not_propagated(self):
        """One broken competency question must not crash the whole gate and
        hide every other result."""
        def _boom(_query: str) -> int:
            raise RuntimeError("backend unreachable")

        questions = [
            CompetencyQuestion(id="q-ok", query="SELECT 1", expect=lambda r: r == 1),
            CompetencyQuestion(id="q-broken", query="SELECT 2", expect=lambda r: True),
        ]
        report = run_competency_suite(
            questions,
            query_fn=lambda q: 1 if "1" in q else _boom(q),
        )
        assert report.passed is False
        assert report.failed_ids == ["q-broken"]
        broken = next(r for r in report.results if r.id == "q-broken")
        assert "backend unreachable" in broken.error


# ── OntologyRegistry.apply_ontology_migration gate ──────────────────────────

_CURRENT = {"classes": {"A": {}}, "properties": {}}
_TARGET = {"classes": {"A": {}}, "properties": {}}


@pytest.fixture
def registry():
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[{"total": 0}])
    return OntologyRegistry(neo4j, tenant="acme")


class TestMigrationGate:
    async def test_migration_proceeds_with_no_competency_report_unchanged_behavior(self, registry):
        """Omitting competency_report entirely preserves pre-gate behavior."""
        result = await registry.apply_ontology_migration(_CURRENT, _TARGET)
        assert result["compatible"] is True

    async def test_migration_proceeds_when_the_competency_report_passed(self, registry):
        passing = CompetencyGateReport(passed=True, failed_ids=[], results=[])
        result = await registry.apply_ontology_migration(_CURRENT, _TARGET, competency_report=passing)
        assert result["compatible"] is True

    async def test_migration_is_blocked_when_the_competency_report_failed(self, registry):
        failing = CompetencyGateReport(passed=False, failed_ids=["q-broken"], results=[])
        with pytest.raises(OntologyMigrationBlockedError, match="q-broken"):
            await registry.apply_ontology_migration(_CURRENT, _TARGET, competency_report=failing)
        # No graph write attempted -- only the incompatibility-check reads,
        # never apply_graph_migrations's write, happened.
        registry._neo4j.run.assert_not_awaited()

    async def test_force_true_overrides_a_failed_competency_report(self, registry):
        """The same failing report that raises OntologyMigrationBlockedError
        in the test above must instead let the migration through when
        forced -- proving `force` is a real escape hatch, not a silent
        no-op alongside the block. (The override is also logged with the
        failing question ids -- graphrag/graph/ontology_registry.py's
        "migration_forced_despite_competency_failure" event -- verified by
        inspection since structlog's own event routing, not stdlib
        `logging`, isn't reliably observable via pytest's `caplog` here.)"""
        failing = CompetencyGateReport(passed=False, failed_ids=["q-broken"], results=[])
        result = await registry.apply_ontology_migration(
            _CURRENT, _TARGET, competency_report=failing, force=True,
        )
        assert result["compatible"] is True

    async def test_an_incompatible_diff_is_still_rejected_before_the_competency_check(self, registry):
        """The plain compatibility check (unmapped removals) still runs
        first and independently of the new gate."""
        incompatible_target = {"classes": {}, "properties": {}}  # drops "A" with no migration_map
        passing = CompetencyGateReport(passed=True, failed_ids=[], results=[])
        with pytest.raises(ValueError, match="unmapped removals"):
            await registry.apply_ontology_migration(
                _CURRENT, incompatible_target, competency_report=passing,
            )


# ── The concrete Energy implementation, run for real ────────────────────────

class TestEnergyCompetencyQuestionIsReallyExecutable:
    def test_it_passes_against_the_energy_demos_own_real_graph(self):
        from graphrag.domains.energy.demo import EnergyDemoService

        service = EnergyDemoService()
        questions = energy_maintenance_review_competency_questions()
        report = run_competency_suite(
            questions, query_fn=lambda q: SPARQLBridge(service.graph).query(q),
        )
        assert report.passed is True, report.results

    def test_it_fails_honestly_against_an_empty_graph(self):
        from rdflib import Graph

        questions = energy_maintenance_review_competency_questions()
        report = run_competency_suite(
            questions, query_fn=lambda q: SPARQLBridge(Graph()).query(q),
        )
        assert report.passed is False
        assert report.failed_ids == ["energy-demo/maintenance-review-finds-open-assets"]


# ── Domain Cypher competency questions (aerospace/automotive/marketing) ────

class TestDomainCompetencyQuestionsWiring:
    """Wiring/structure only -- see the module docstring for why these
    cannot be run against a real Neo4j graph in this environment."""

    @pytest.mark.parametrize("factory", [
        aerospace_regulatory_competency_questions,
        automotive_iatf_competency_questions,
        marketing_adtech_competency_questions,
    ])
    def test_every_question_has_fully_substituted_nonempty_cypher(self, factory):
        questions = factory()
        assert questions, "factory returned no questions"
        ids = [q.id for q in questions]
        assert len(ids) == len(set(ids)), "duplicate question ids within one domain"
        for question in questions:
            assert question.query.strip()
            assert "{{" not in question.query, f"{question.id}: unsubstituted placeholder"
            assert "MATCH" in question.query
            assert "RETURN" in question.query

    def test_aerospace_questions_accept_default_directive_and_tenant(self):
        questions = aerospace_regulatory_competency_questions()
        assert any("AD-2024-01-02" in q.query for q in questions)
        assert all("aerospace" in q.query for q in questions)

    def test_custom_arguments_are_substituted_not_ignored(self):
        questions = automotive_iatf_competency_questions(supplier="Acme Parts SRL", tenant="acme-tenant")
        assert all("Acme Parts SRL" in q.query for q in questions)
        assert all("acme-tenant" in q.query for q in questions)

    def test_expect_callables_reject_empty_or_malformed_rows(self):
        for factory in (aerospace_regulatory_competency_questions, automotive_iatf_competency_questions,
                        marketing_adtech_competency_questions):
            for question in factory():
                assert question.expect([]) is False
                assert question.expect([{"unexpected_field": "x"}]) is False

    def test_competency_questions_by_domain_registers_all_four_tracks(self):
        registry = competency_questions_by_domain()
        assert set(registry) == {
            "energy-asset-intelligence", "aerospace-regulatory",
            "automotive-iatf", "marketing-adtech",
        }
        for factory in registry.values():
            assert factory()  # every registered factory returns at least one question


class TestCheckCompetencyQuestionsAgainstOntology:
    """The static drift regression gate: every relation/type name a
    domain's Cypher literally references must still be declared in that
    domain's current config/ontologies/*.yml."""

    @pytest.mark.parametrize("domain_id", ["aerospace-regulatory", "automotive-iatf", "marketing-adtech"])
    def test_the_real_shipped_questions_are_clean_against_the_real_ontology(self, domain_id):
        """Not just "this function works" -- proof the Cypher I wrote
        actually is grounded in the current YAML, not merely plausible-
        looking."""
        assert check_competency_questions_against_ontology(domain_id) == []

    def test_a_renamed_relation_in_the_ontology_is_caught(self, monkeypatch):
        import graphrag.graph.competency_gate as gate

        def _fake_load(path):
            return {"relation_rules": {}, "type_hierarchy": []}  # SUPERSEDES etc. all gone

        def _fake_pairs(ontology):
            return []

        monkeypatch.setattr(gate, "competency_questions_by_domain", lambda: {
            "aerospace-regulatory": aerospace_regulatory_competency_questions,
        })
        import graphrag.graph.domain_ontology as domain_ontology_module
        monkeypatch.setattr(domain_ontology_module, "load_domain_ontology", _fake_load)
        monkeypatch.setattr(domain_ontology_module, "get_type_hierarchy_pairs", _fake_pairs)

        problems = check_competency_questions_against_ontology("aerospace-regulatory")
        assert problems
        assert any("SUPERSEDES" in p for p in problems)

    def test_an_unregistered_domain_returns_empty_rather_than_raising(self):
        assert check_competency_questions_against_ontology("no-such-domain") == []


# ── Async suite runner + Neo4j query_fn adapter ─────────────────────────────

class TestRunCompetencySuiteAsync:
    async def test_all_passing_mirrors_the_sync_runner(self):
        questions = [CompetencyQuestion(id="q1", query="Q1", expect=lambda r: r == [{"x": 1}])]
        report = await run_competency_suite_async(questions, query_fn=AsyncMock(return_value=[{"x": 1}]))
        assert report.passed is True

    async def test_a_raising_async_query_is_isolated_not_propagated(self):
        async def _boom(_query: str):
            raise RuntimeError("neo4j unreachable")

        questions = [
            CompetencyQuestion(id="q-ok", query="Q1", expect=lambda r: True),
            CompetencyQuestion(id="q-broken", query="Q2", expect=lambda r: True),
        ]
        calls = iter([None, "boom"])

        async def query_fn(q: str):
            if next(calls) == "boom":
                return await _boom(q)
            return []

        report = await run_competency_suite_async(questions, query_fn=query_fn)
        assert report.failed_ids == ["q-broken"]
        assert "neo4j unreachable" in next(r for r in report.results if r.id == "q-broken").error


class TestNeo4jCypherQueryFn:
    async def test_adapts_a_client_run_method_into_the_query_fn_contract(self):
        class _FakeNeo4jClient:
            def __init__(self):
                self.received_cypher = None

            async def run(self, cypher: str) -> list[dict]:
                self.received_cypher = cypher
                return [{"aircraftType": "Boeing 737"}]

        client = _FakeNeo4jClient()
        query_fn = neo4j_cypher_query_fn(client)
        rows = await query_fn("MATCH (n) RETURN n")

        assert rows == [{"aircraftType": "Boeing 737"}]
        assert client.received_cypher == "MATCH (n) RETURN n"

    async def test_end_to_end_against_a_fake_client_passes_a_real_question(self):
        """The full path a live deployment would use: a real question's
        substituted Cypher, through the adapter, through the async suite
        runner -- with a fake client standing in for Neo4j since none is
        available here."""
        class _FakeNeo4jClient:
            async def run(self, cypher: str) -> list[dict]:
                assert "AIRCRAFT_TYPE" in cypher
                return [{"aircraftType": "Boeing 737"}]

        questions = aerospace_regulatory_competency_questions()
        report = await run_competency_suite_async(questions, query_fn=neo4j_cypher_query_fn(_FakeNeo4jClient()))
        applicable = next(r for r in report.results if r.id == "aerospace-regulatory/applicable-aircraft-type")
        assert applicable.passed is True


# ── Model-change impact analysis (ontology_migration.py) ────────────────────

class TestFindAffectedCompetencyQuestions:
    def test_a_removed_name_referenced_by_a_questions_query_is_flagged(self):
        report = MigrationReport(compatible=True, removed=["AIRCRAFT_TYPE"])
        affected = find_affected_competency_questions(report, domain_id="aerospace-regulatory")
        assert "aerospace-regulatory/applicable-aircraft-type" in affected

    def test_an_unrelated_removed_name_flags_nothing(self):
        report = MigrationReport(compatible=True, removed=["SOME_UNUSED_TYPE"])
        affected = find_affected_competency_questions(report, domain_id="aerospace-regulatory")
        assert affected == []

    def test_a_renamed_from_name_is_treated_the_same_as_removed(self):
        report = MigrationReport(compatible=True, renamed=[("SUPPLIER_CLASSIFICATION", "SUPPLIER_TIER")])
        affected = find_affected_competency_questions(report, domain_id="automotive-iatf")
        assert "automotive-iatf/supplier-classification" not in affected  # the query references REEVALUATION_FREQUENCY, not this type directly
        # But a name that genuinely appears in a query text is caught:
        report2 = MigrationReport(compatible=True, renamed=[("REEVALUATION_FREQUENCY", "REVIEW_CADENCE")])
        affected2 = find_affected_competency_questions(report2, domain_id="automotive-iatf")
        assert "automotive-iatf/reevaluation-frequency" in affected2

    def test_an_unregistered_domain_returns_empty_rather_than_raising(self):
        report = MigrationReport(compatible=True, removed=["ANYTHING"])
        assert find_affected_competency_questions(report, domain_id="no-such-domain") == []

    def test_a_report_with_no_removed_or_renamed_names_flags_nothing(self):
        report = MigrationReport(compatible=True, added=["NEW_TYPE"])
        assert find_affected_competency_questions(report, domain_id="aerospace-regulatory") == []
