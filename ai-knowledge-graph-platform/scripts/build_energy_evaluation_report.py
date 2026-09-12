#!/usr/bin/env python
"""Build a measured local evaluation report for the Energy RDF demonstration.

The report turns the Energy POC's fixed, asserted workflow into a reviewable
evidence artifact. It evaluates answer correctness, evidence coverage,
abstention, tenant isolation, publication freshness, answer latency, and RDF
materialisation throughput. It deliberately reports only local observations
over synthetic records; it is not a production scale, availability, or customer
outcomes claim.

Usage:
    python scripts/build_energy_evaluation_report.py
    python scripts/build_energy_evaluation_report.py --output artifacts/energy-evaluation-report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.energy.demo import EnergyDemoService, TENANT  # noqa: E402
from graphrag.ops.production_exercises import run_load_exercise  # noqa: E402

_QUESTION_PATH = ROOT / "evals" / "energy_demo" / "questions.json"
_DEFAULT_OUTPUT = ROOT / "artifacts" / "energy-evaluation-report.json"
_MAINTENANCE_EVIDENCE_IDS = {"SAP-WO-9001", "SNOW-OBS-WT-01", "MFG-GBX-17-R2"}


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("questions")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} has no non-empty 'questions' list")
    return [dict(case) for case in cases]


def evaluate_answers(service: EnergyDemoService, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the five committed question fixtures against their expectations."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        response = service.answer(case["id"], tenant=TENANT, as_of=case.get("as_of"))
        expected_answer_values = [
            value for key, value in case.items() if key.startswith("expected_") and key != "expected_status"
            for value in (value if isinstance(value, list) else [value])
        ]
        expected_status = case.get("expected_status")
        answer_passed = all(str(value) in str(response.get("answer", "")) for value in expected_answer_values)
        status_passed = expected_status is None or response.get("status") == expected_status
        passed = answer_passed and status_passed
        rows.append({
            "id": case["id"],
            "expected_answer_values": [str(value) for value in expected_answer_values],
            "expected_status": expected_status,
            "status": response["status"],
            "passed": passed,
        })
    passed = sum(row["passed"] for row in rows)
    return {
        "available": True,
        "passed": passed,
        "total": len(rows),
        "pass_rate": passed / len(rows),
        "cases": rows,
    }


def evaluate_maintenance_evidence(service: EnergyDemoService) -> dict[str, Any]:
    """Verify that the advisory cites source evidence and committed SPARQL rows."""
    response = service.answer("maintenance_review", tenant=TENANT)
    evidence_ids = {str(item["source_id"]) for item in response.get("evidence", [])}
    query_rows = response.get("query_rows", [])
    query_row_matches = len(query_rows) == 1 and (
        str(query_rows[0].get("asset", "")).endswith("/WT-01")
        and str(query_rows[0].get("workOrder", "")).endswith("/WO-9001")
    )
    passed = (
        response.get("answer_source") == "version-controlled SPARQL query"
        and _MAINTENANCE_EVIDENCE_IDS.issubset(evidence_ids)
        and query_row_matches
    )
    return {
        "available": True,
        "passed": passed,
        "expected_source_ids": sorted(_MAINTENANCE_EVIDENCE_IDS),
        "observed_source_ids": sorted(evidence_ids),
        "query_row_count": len(query_rows),
        "query_row_matches": query_row_matches,
        "answer_source": response.get("answer_source"),
    }


def evaluate_abstention(service: EnergyDemoService) -> dict[str, Any]:
    response = service.answer("insufficient_evidence", tenant=TENANT)
    passed = response.get("status") == "insufficient_evidence"
    return {"available": True, "passed": passed, "status": response.get("status")}


def evaluate_tenant_isolation(service: EnergyDemoService) -> dict[str, Any]:
    response = service.answer("maintenance_review", tenant="untrusted-tenant")
    passed = response.get("status") == "not_found" and response.get("evidence") == []
    return {
        "available": True,
        "passed": passed,
        "wrong_tenant_status": response.get("status"),
        "evidence_count": len(response.get("evidence", [])),
    }


def publication_freshness(service: EnergyDemoService) -> dict[str, Any]:
    """Expose published-dataset timing without pretending fixture time is live-source freshness."""
    publication = service.publication_report()
    published_at = datetime.fromisoformat(publication.published_at).astimezone(timezone.utc)
    age_ms = max(0.0, (datetime.now(timezone.utc) - published_at).total_seconds() * 1000)
    evidence = service.answer("maintenance_review", tenant=TENANT)["evidence"]
    return {
        "available": True,
        "published_at": publication.published_at,
        "publication_age_ms_at_report": round(age_ms, 3),
        "version_id": publication.version_id,
        "published_triple_count": publication.published_triple_count,
        "candidate_record_count": publication.candidate_record_count,
        "quarantined_count": publication.quarantined_count,
        "synthetic_source_observed_at": sorted({item["observed_at"] for item in evidence}),
        "claim_policy": (
            "published_at measures local in-process publication time. Source timestamps are "
            "synthetic fixture metadata, not an SLA or live-source freshness measurement."
        ),
    }


async def measure_answer_latency(service: EnergyDemoService, cases: list[dict[str, Any]], iterations: int) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("latency iterations must be at least 1")
    operations = [
        {"id": case["id"], "as_of": case.get("as_of")}
        for _ in range(iterations)
        for case in cases
    ]

    async def answer(case: dict[str, Any]) -> None:
        service.answer(case["id"], tenant=TENANT, as_of=case.get("as_of"))

    measured = await run_load_exercise(answer, operations, concurrency=1)
    return {
        "available": True,
        "sample_count": measured["total"],
        "p50_latency_ms": round(measured["p50_latency_ms"], 3),
        "p95_latency_ms": round(measured["p95_latency_ms"], 3),
        "p99_latency_ms": round(measured["p99_latency_ms"], 3),
        "throughput_rps": round(measured["throughput_rps"], 3),
        "failed": measured["failed"],
        "claim_policy": "Measured locally against a synthetic, in-process RDF graph; not a service latency SLA.",
    }


def measure_ingestion_throughput(iterations: int) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("ingestion iterations must be at least 1")
    records = triples = 0
    started = time.perf_counter()
    for _ in range(iterations):
        service = EnergyDemoService()
        publication = service.publication_report()
        records += publication.candidate_record_count
        triples += publication.published_triple_count
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "available": True,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 6),
        "candidate_records": records,
        "published_triples": triples,
        "candidate_records_per_second": round(records / elapsed, 3),
        "published_triples_per_second": round(triples / elapsed, 3),
        "claim_policy": (
            "Measured locally while materialising the tiny synthetic Energy fixture; it is a repeatable "
            "implementation check, not an enterprise ingestion-capacity benchmark."
        ),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def build_report(
    *,
    question_path: Path = _QUESTION_PATH,
    latency_iterations: int = 5,
    ingestion_iterations: int = 3,
) -> dict[str, Any]:
    """Build the report, retaining a usable artifact if optional timing fails."""
    cases = _read_cases(question_path)
    service = EnergyDemoService()
    try:
        latency = asyncio.run(measure_answer_latency(service, cases, latency_iterations))
    except Exception as exc:  # noqa: BLE001 - report the measurement failure explicitly
        latency = _unavailable(f"local answer-latency measurement failed: {type(exc).__name__}: {exc}")
    try:
        ingestion = measure_ingestion_throughput(ingestion_iterations)
    except Exception as exc:  # noqa: BLE001 - report the measurement failure explicitly
        ingestion = _unavailable(f"local ingestion-throughput measurement failed: {type(exc).__name__}: {exc}")
    return {
        "report_schema_version": "energy-evaluation/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "energy-demo/v1",
        "synthetic": True,
        "claim_policy": (
            "This report is reproducible local evidence over a small synthetic Energy dataset. It demonstrates "
            "workflow correctness and implementation behavior only; it does not establish enterprise-scale "
            "throughput, production latency, availability, or customer outcomes."
        ),
        "answer_correctness": evaluate_answers(service, cases),
        "evidence_accuracy": evaluate_maintenance_evidence(service),
        "abstention": evaluate_abstention(service),
        "tenant_isolation": evaluate_tenant_isolation(service),
        "freshness": publication_freshness(service),
        "latency": latency,
        "ingestion_throughput": ingestion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--latency-iterations", type=int, default=5)
    parser.add_argument("--ingestion-iterations", type=int, default=3)
    args = parser.parse_args()

    report = build_report(
        latency_iterations=args.latency_iterations,
        ingestion_iterations=args.ingestion_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
