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
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.competency_gate import (
    CompetencyGateReport,
    CompetencyQuestion,
    energy_maintenance_review_competency_questions,
    run_competency_suite,
)
from graphrag.graph.ontology_migration import OntologyMigrationBlockedError
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
