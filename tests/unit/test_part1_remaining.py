from graphrag.graph.ontology_migration import plan_migration
from graphrag.observability.cost_attribution import CostEvent, aggregate_costs
from graphrag.observability.budgets import StageBudget, check_budget
from graphrag.ops.exercises import recovery_check, security_matrix
from graphrag.ops.production_exercises import (
    run_backup_recovery_exercise, run_load_exercise, run_security_exercise,
)
from graphrag.retrieval.query_planner import classify_query, retrieval_plan
from scripts.run_production_exercises import cost_exercise, recovery_exercise


def test_query_planner_routes_classes_and_fallbacks():
    assert classify_query("Which documents conflict?") == "contradiction"
    assert retrieval_plan("Compare the chain across steps")["fallback"] == "agentic"


def test_query_planner_routes_existence_check_to_negative_class():
    # Added 2026-08-17 for NEG-03 — see query_planner.py's classify_query()
    # docstring comment for the full root-cause trace. "is there"/"any
    # evidence" phrasing needs the same wider top_k as "contradiction": a
    # contrasting fact needed to answer correctly can score low against the
    # cross-encoder reranker's literal-relevance-to-query-wording model.
    plan = retrieval_plan("Is there a FAA airworthiness directive governing Airbus aircraft in this corpus?")
    assert plan["query_class"] == "negative"
    assert plan["top_k"] == 10
    assert plan["mode"] == "hybrid"

    # Regression guard: plain factoid/single-hop questions must NOT be swept
    # into the wider tier just because they happen to ask about existence of
    # a fact in passing — only the "is there"/"any evidence" phrasing should.
    assert classify_query("Who manufactures the Boeing 737 MAX?") == "factoid"
    assert classify_query("Does Airbus manufacture the Boeing 737 MAX?") == "factoid"


def test_migration_report_requires_mapping_for_removals():
    report = plan_migration({"classes": {"Old": {}}, "properties": {}},
                            {"classes": {"New": {}}, "properties": {},
                             "migration_map": {"Old": "New"}})
    assert report.compatible is True
    assert report.renamed == [("Old", "New")]


def test_cost_attribution_and_exercise_helpers():
    events = [CostEvent("acme", "map", "groq", "m", 0.1),
              CostEvent("acme", "map", "groq", "m", 0.2)]
    assert aggregate_costs(events)[0]["cost_usd"] == 0.30000000000000004
    assert security_matrix([{"name": "cross", "expected_tenant": "a", "observed_tenant": "b"}])["failed"] == 1
    assert recovery_check("abc", "abc")["match"] is True


async def test_p3_load_security_and_backup_exercises():
    async def operation(case):
        if case.get("fail"):
            raise RuntimeError("synthetic failure")

    load = await run_load_exercise(operation, [{"tenant": "a"}, {"tenant": "b", "fail": True}], 2)
    assert load["total"] == 2 and load["failed"] == 1
    assert load["p50_latency_ms"] >= 0 and load["p95_latency_ms"] >= load["p50_latency_ms"]
    assert load["p99_latency_ms"] >= load["p95_latency_ms"]
    assert load["throughput_rps"] > 0 and load["error_rate"] == 0.5
    security = run_security_exercise([
        {"name": "restricted", "expected_tenant": "a", "observed_tenant": "a",
         "restricted": True, "allowed": False},
        {"name": "destructive", "expected_tenant": "a", "observed_tenant": "a",
         "destructive": True, "approval_required": False},
    ])
    assert security["failed"] == 1
    recovery = await run_backup_recovery_exercise(lambda: _value("digest"), lambda _: _value("digest"))
    assert recovery["match"] is True


def test_executable_recovery_and_cost_exercises(tmp_path):
    backup = tmp_path / "backup.dump"
    restored = tmp_path / "restored.dump"
    backup.write_bytes(b"verified graph backup")
    restored.write_bytes(b"verified graph backup")
    assert recovery_exercise(backup, restored)["match"] is True
    report = cost_exercise([{
        "tenant": "acme", "stage": "synthesis", "provider": "groq",
        "model": "model", "cost_usd": 0.05, "latency_ms": 100,
    }])
    assert report["events"] == 1 and report["totals"][0]["cost_usd"] == 0.05


def test_budget_verdict_covers_latency_and_cost_controls():
    budgets = {"synthesis": StageBudget(latency_ms=100, cost_usd=0.01)}
    result = check_budget("synthesis", 101, 0.02, budgets)
    assert result["latency_over"] is True
    assert result["cost_over"] is True
    assert result["within_budget"] is False


async def _value(value):
    return value
