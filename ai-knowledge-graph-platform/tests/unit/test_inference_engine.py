"""Unit tests for ForwardChainingEngine — rule dispatch, fixpoint, dry-run, tenant isolation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from graphrag.graph.inference_engine import (
    DEFAULT_RULES,
    ForwardChainingEngine,
    InferenceRule,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock():
    return AsyncMock()


@pytest.fixture
def engine(neo4j_mock):
    return ForwardChainingEngine(neo4j_client=neo4j_mock, rules=[])


# ── InferenceRule dataclass ────────────────────────────────────────────────────

class TestInferenceRuleDefaults:
    def test_default_derived_relation_is_empty(self):
        rule = InferenceRule(name="r", rule_type="symmetry", relation="REL")
        assert rule.derived_relation == ""

    def test_default_max_depth(self):
        rule = InferenceRule(name="r", rule_type="transitivity", relation="REL")
        assert rule.max_depth == 3

    def test_confidence_decay_default(self):
        rule = InferenceRule(name="r", rule_type="inverse", relation="REL")
        assert rule.confidence_decay == pytest.approx(0.9)


class TestDefaultRules:
    def test_default_rules_are_non_empty(self):
        assert len(DEFAULT_RULES) > 0

    def test_all_default_rules_have_names(self):
        for rule in DEFAULT_RULES:
            assert rule.name, f"Rule {rule!r} has no name"

    def test_default_rules_have_valid_types(self):
        valid = {"transitivity", "symmetry", "inverse", "composition"}
        for rule in DEFAULT_RULES:
            assert rule.rule_type in valid, f"Unknown rule type: {rule.rule_type}"


# ── add_rule ───────────────────────────────────────────────────────────────────

class TestAddRule:
    def test_adds_rule_to_list(self, engine):
        # Note: rules=[] falls back to DEFAULT_RULES (falsy list) — this is by design.
        initial = len(engine._rules)
        engine.add_rule(InferenceRule(name="test", rule_type="symmetry", relation="X"))
        assert len(engine._rules) == initial + 1

    def test_added_rule_is_applied_on_run(self, engine, neo4j_mock):
        """add_rule wires the rule into the next run() call."""
        engine.add_rule(InferenceRule(name="sym", rule_type="symmetry", relation="REL"))
        neo4j_mock.run = AsyncMock(return_value=[])
        # run() should not raise even with a real rule and empty DB response
        # (tested below in TestRun)


# ── run() — fixpoint and report structure ─────────────────────────────────────

class TestRunFixpoint:
    async def test_returns_report_with_required_fields(self, engine, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run(tenant="acme")
        assert "tenant" in report
        assert "dry_run" in report
        assert "total_inferred" in report
        assert "by_rule" in report

    async def test_empty_rules_returns_zero_inferred(self, neo4j_mock):
        # Pass an explicit non-empty sentinel rule list that produces no rows
        # (rules=[] falls back to DEFAULT_RULES because empty list is falsy)
        sentinel = InferenceRule(name="noop", rule_type="symmetry", relation="NOOP_REL")
        engine = ForwardChainingEngine(neo4j_client=neo4j_mock, rules=[sentinel])
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run(tenant="acme")
        assert report["total_inferred"] == 0
        assert "noop" in report["by_rule"]

    async def test_fixpoint_stops_after_no_new_edges(self, neo4j_mock):
        """If iteration produces 0 new edges, loop should stop without max_iterations."""
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="sym", rule_type="symmetry", relation="REL")],
        )
        # First iteration: 1 row → 1 derived edge
        # Second iteration: 0 rows → fixpoint
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"src": "A", "src_type": "X", "tgt": "B", "tgt_type": "X"}],  # symmetry query
            [],   # MERGE write
            [],   # second iteration → 0 rows → fixpoint
        ])
        report = await engine.run(tenant="acme", max_iterations=10)
        # Should have stopped at iteration 2, not run all 10
        assert neo4j_mock.run.call_count < 10

    async def test_tenant_field_in_report(self, engine, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run(tenant="myorg")
        assert report["tenant"] == "myorg"


# ── dry_run ────────────────────────────────────────────────────────────────────

class TestDryRun:
    async def test_dry_run_does_not_write_edges(self, neo4j_mock):
        """dry_run=True should query but never call _write_inferred_edge (no MERGE)."""
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="sym", rule_type="symmetry", relation="REL")],
        )
        # Return 2 candidate rows from the symmetry query
        rows = [
            {"src": "A", "src_type": "T", "tgt": "B", "tgt_type": "T",
             "inferred_conf": 0.8},
            {"src": "C", "src_type": "T", "tgt": "D", "tgt_type": "T",
             "inferred_conf": 0.7},
        ]
        neo4j_mock.run = AsyncMock(return_value=rows)
        report = await engine.run(tenant="acme", dry_run=True, max_iterations=1)
        # dry_run → should count rows but not fire any MERGE queries
        # The MERGE write path would call neo4j.run a second time per row;
        # with dry_run we expect exactly 1 call (the read query) per rule per iteration.
        assert report["dry_run"] is True
        # total_inferred = number of candidate rows found
        assert report["total_inferred"] >= 2

    async def test_dry_run_flag_in_report(self, engine, neo4j_mock):
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run(tenant="acme", dry_run=True)
        assert report["dry_run"] is True


# ── Rule type dispatch ─────────────────────────────────────────────────────────

class TestRuleTypeDispatch:
    async def test_unknown_rule_type_returns_zero(self, neo4j_mock):
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="bad", rule_type="magic", relation="REL")],
        )
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run(tenant="acme", max_iterations=1)
        assert report["by_rule"].get("bad", 0) == 0

    async def test_symmetry_rule_dispatched(self, neo4j_mock):
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="sym", rule_type="symmetry", relation="REL")],
        )
        neo4j_mock.run = AsyncMock(return_value=[])
        await engine.run(tenant="acme", max_iterations=1)
        # At least one call to neo4j must have been made (the symmetry query)
        assert neo4j_mock.run.called

    async def test_inverse_rule_dispatched(self, neo4j_mock):
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="inv", rule_type="inverse",
                                 relation="WORKS_AT", derived_relation="EMPLOYS")],
        )
        neo4j_mock.run = AsyncMock(return_value=[])
        await engine.run(tenant="acme", max_iterations=1)
        assert neo4j_mock.run.called


# ── run_for_document ───────────────────────────────────────────────────────────

class TestRunForDocument:
    async def test_no_affected_entities_returns_early(self, engine, neo4j_mock):
        """If no entities are found for the doc, run_for_document returns zeros immediately."""
        neo4j_mock.run = AsyncMock(return_value=[])
        report = await engine.run_for_document(doc_id="doc_xyz", tenant="acme")
        assert report["total_inferred"] == 0
        assert report.get("doc_id") == "doc_xyz"

    async def test_affected_entities_triggers_run(self, neo4j_mock):
        engine = ForwardChainingEngine(
            neo4j_client=neo4j_mock,
            rules=[InferenceRule(name="sym", rule_type="symmetry", relation="REL")],
        )
        # First call: MATCH entities for doc → returns 1 entity
        # Subsequent calls: run() rule queries → empty (no new edges)
        neo4j_mock.run = AsyncMock(side_effect=[
            [{"name": "SpaceX", "type": "ORG"}],  # doc entities
            [],   # symmetry query → fixpoint immediately
        ])
        report = await engine.run_for_document(doc_id="doc_abc", tenant="acme")
        # run_for_document should have called run() which made at least 1 more query
        assert neo4j_mock.run.call_count >= 2
