"""Minimal structured invariant checking.

No existing trace/invariant/diagnostics module exists elsewhere in this
codebase -- this is deliberately small (a runner plus two call sites, not a
framework). Matches existing house style: structlog.get_logger(__name__) at
module scope (gets free PII redaction from the central configure_logging()
pipeline, same as every other module that logs this way), a RuntimeError-
derived exception matching src/graph/sales_ontology.py's own
UnknownClaimPredicate/UnknownGraphRelation convention, and a Prometheus
counter matching src/graph/repositories/claim_repository.py's existing
CLAIMS_TOTAL.labels(...) pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import structlog

from src.core.telemetry import INVARIANT_CHECKS_TOTAL

log = structlog.get_logger(__name__)


class InvariantViolation(RuntimeError):
    """Raised when one or more invariant checks fail and raise_on_failure=True."""


@dataclass(frozen=True)
class InvariantCheck:
    name: str
    check_fn: Callable[[], bool]
    detail: str = ""


@dataclass(frozen=True)
class InvariantResult:
    stage: str
    name: str
    passed: bool
    detail: str = ""


def run_invariants(
    stage: str, checks: list[InvariantCheck], *, raise_on_failure: bool = True
) -> list[InvariantResult]:
    """Run each check, log a structured pass/fail event, increment
    INVARIANT_CHECKS_TOTAL, and (by default) raise InvariantViolation naming
    every failed check if any failed. A check_fn that itself raises counts as
    a failure rather than propagating -- one bad check must not take down
    the others or hide their results.
    """
    results: list[InvariantResult] = []
    failed: list[InvariantResult] = []
    for check in checks:
        try:
            passed = bool(check.check_fn())
            detail = check.detail
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            passed = False
            detail = f"{check.detail} (check raised: {exc})".strip()
        result = InvariantResult(stage=stage, name=check.name, passed=passed, detail=detail)
        results.append(result)
        INVARIANT_CHECKS_TOTAL.labels(stage=stage, name=check.name, outcome="pass" if passed else "fail").inc()
        log.info("diagnostics.invariant_checked", stage=stage, name=check.name, passed=passed, detail=detail)
        if not passed:
            failed.append(result)
    if failed and raise_on_failure:
        raise InvariantViolation(f"{stage}: invariant(s) failed: {', '.join(r.name for r in failed)}")
    return results
