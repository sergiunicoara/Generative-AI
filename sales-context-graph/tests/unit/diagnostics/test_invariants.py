from __future__ import annotations

import pytest

from src.diagnostics.invariants import InvariantCheck, InvariantViolation, run_invariants


def test_run_invariants_logs_and_counts_a_passing_check():
    results = run_invariants("test.stage", [InvariantCheck(name="always_true", check_fn=lambda: True)])
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].stage == "test.stage"
    assert results[0].name == "always_true"


def test_run_invariants_raises_by_default_on_a_failing_check():
    with pytest.raises(InvariantViolation, match="always_false"):
        run_invariants("test.stage", [InvariantCheck(name="always_false", check_fn=lambda: False)])


def test_run_invariants_collects_failures_without_raising_when_disabled():
    results = run_invariants(
        "test.stage",
        [InvariantCheck(name="always_false", check_fn=lambda: False)],
        raise_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].passed is False


def test_run_invariants_treats_a_raising_check_fn_as_a_failure():
    def _boom():
        raise RuntimeError("something went wrong inside the check")

    results = run_invariants(
        "test.stage", [InvariantCheck(name="raises", check_fn=_boom)], raise_on_failure=False,
    )
    assert results[0].passed is False
    assert "something went wrong inside the check" in results[0].detail


def test_run_invariants_reports_only_the_failing_checks_by_name():
    with pytest.raises(InvariantViolation) as exc_info:
        run_invariants("test.stage", [
            InvariantCheck(name="passing", check_fn=lambda: True),
            InvariantCheck(name="failing", check_fn=lambda: False),
        ])
    assert "failing" in str(exc_info.value)
    assert "passing" not in str(exc_info.value)
