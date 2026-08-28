from graphrag.evaluation.rubrics import (
    RubricRegistry,
    RubricSpec,
    RubricResult,
    build_observations,
)


def test_default_rubrics_are_versioned_and_aggregate():
    result = RubricRegistry().evaluate({name: True for name in (
        "citation_present", "citation_resolves_to_source", "answer_supported", "tool_execution_success",
        "authorized_scope", "tenant_scope_preserved", "freshness_verified", "cost_budget_respected",
        "latency_budget_respected", "pii_policy_respected")})
    assert result.passed
    assert result.score == 1
    assert all(item.version == "1.0" for item in result.rubrics)
    assert result.config["rubric_versions"]["citation_present"] == "1.0"


def test_security_failure_overrides_aggregate():
    observations = {name: True for name in ("citation_present", "answer_supported", "freshness_verified")}
    observations["tenant_scope_preserved"] = False
    result = RubricRegistry().evaluate(observations)
    assert result.hard_failed
    assert not result.passed


def test_registry_supports_dependencies_and_penalties():
    registry = RubricRegistry([])
    registry.register(RubricSpec("base", "2.0", lambda _: RubricResult(rubric_id="base", version="2.0", passed=False, score=0, reason="no")))
    registry.register(RubricSpec("dependent", "1.0", lambda _: RubricResult(rubric_id="dependent", version="1.0", passed=True, score=1, reason="yes"), depends_on=("base",)))
    result = registry.evaluate({}, ["base", "dependent"])
    assert result.rubrics[1].reason.startswith("dependency failed")


def test_observations_record_explicit_authorization_denial_and_pii():
    observations = build_observations(
        answer="Contact alice@example.com", citations=["source"], contexts=["source"],
        tenant="acme", policy_result="escalate", policy_reason_code="no_authorized_evidence",
    )
    result = RubricRegistry().evaluate(observations)
    failures = {item.rubric_id for item in result.rubrics if not item.passed}
    assert {"authorized_scope", "pii_policy_respected"} <= failures
    assert result.hard_failed
