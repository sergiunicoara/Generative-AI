import pytest

from limina_benchmark.metrics import case_accounting, confusion_metrics, repeatability_metrics
from limina_benchmark.schemas import EvaluationCase, EvaluatorResult, TraceNode

pytestmark = pytest.mark.offline_eval


def _case(case_id: str, expected_failure: bool) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        category="healthy",
        source="historical",
        source_reference="fixture",
        trajectory=[TraceNode(node_id="u", kind="user", text="hello")],
        expected_failure=expected_failure,
    )


def _result(case_id: str, failure: bool, raw: dict) -> EvaluatorResult:
    return EvaluatorResult(
        evaluator="fixture",
        case_id=case_id,
        status="ok",
        detected_failure=failure,
        raw_result=raw,
    )


def test_confusion_metrics_are_binary_and_explicit():
    metrics = confusion_metrics(
        [_case("tp", True), _case("tn", False), _case("fp", False), _case("fn", True)],
        [_result("tp", True, {}), _result("tn", False, {}), _result("fp", True, {}), _result("fn", False, {})],
    )
    assert metrics == {"evaluated_cases": 4, "tp": 1, "tn": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 0.5, "specificity": 0.5, "false_positive_rate": 0.5, "false_negative_rate": 0.5}


def test_repeatability_does_not_claim_consistency_for_one_run():
    assert repeatability_metrics([_result("one", False, {"a": 1})])["exact_consistency"] is None


def test_repeatability_detects_exact_stable_outputs():
    metrics = repeatability_metrics([_result("one", False, {"a": 1}), _result("one", False, {"a": 1})])
    assert metrics["exact_consistency"] == 1.0
    assert metrics["classification_consistency"] == 1.0


def test_case_accounting_covers_every_case_exactly_once():
    cases = [_case("tp", True), _case("tn", False), _case("fp", False), _case("fn", True)]
    results = [
        _result("tp", True, {}),
        _result("tn", False, {}),
        _result("fp", True, {}),
        _result("fn", False, {}),
    ]
    for result in results:
        result.evaluator = "limina"
    rows, totals = case_accounting(cases, results, "limina")
    assert totals == {"TP": 1, "TN": 1, "FP": 1, "FN": 1, "unscored": 0, "scored": 4, "dataset_size": 4}
    assert len(rows) == totals["dataset_size"]
