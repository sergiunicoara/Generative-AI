"""Prove one service's requirements subset is enough to run that service.

Every Docker/Fly/k8s image is built from ``requirements/${SERVICE}.txt`` (see
the Dockerfile), never from the monolithic ``requirements.txt`` that CI's other
jobs install. A package declared only in the monolith therefore passes every
other CI job and breaks the image -- which is how the API image, the dashboard
image and every worker's paid-LLM-call path shipped broken (audit-2026-10-01).

Run inside a venv that has installed *only* ``requirements/<service>.txt``::

    pip install -r requirements/query.txt
    python scripts/verify_service_imports.py query

It does two things:

1. Imports every module the service's container actually starts (entry points
   copied from docker-compose.yml / fly/*/fly.toml / deploy/kubernetes).
2. Runs a tenant-attributed ``llm_call_span`` to completion. That path does its
   quota/cost bookkeeping from a ``finally`` block after the provider has
   already been paid, so a lazily-imported dependency missing there crashes
   work that already cost money -- and no import-only check would see it.

Exit status is non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVICE_ENTRY_POINTS: dict[str, list[str]] = {
    "api": ["api.main", "mcp_server.remote"],
    "dashboard": ["graphrag.business_matrix.dashboard_server"],
    "ingestion": ["workers.ingestion_worker"],
    "query": ["workers.query_worker"],
    "evaluation": ["workers.evaluation_worker"],
    "workers": ["workers.combined_worker"],
    "backup": ["scripts.kg_backup"],
}

# Services that make LLM calls and therefore run the cost-attribution path.
SERVICES_WITH_LLM_CALLS = {"api", "ingestion", "query", "evaluation", "workers"}


async def _paid_llm_call_completes() -> None:
    from graphrag.observability.correlation import tenant_context
    from graphrag.observability.genai_telemetry import llm_call_span

    with tenant_context("verify-service-imports"):
        with llm_call_span(provider="groq", model="openai/gpt-oss-120b") as span:
            span.update(
                response_model="openai/gpt-oss-120b", input_tokens=1200, output_tokens=300,
            )
    # Let the fire-and-forget quota task run so a failure inside it surfaces.
    await asyncio.sleep(0)


def verify(service: str) -> list[str]:
    failures: list[str] = []
    for module in SERVICE_ENTRY_POINTS[service]:
        try:
            importlib.import_module(module)
            print(f"  ok   import {module}")
        except BaseException as exc:  # noqa: BLE001 - report every failure, don't stop at the first
            failures.append(f"import {module}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    if service in SERVICES_WITH_LLM_CALLS:
        try:
            asyncio.run(_paid_llm_call_completes())
            print("  ok   tenant-attributed llm_call_span completes")
        except BaseException as exc:  # noqa: BLE001
            failures.append(f"llm_call_span: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in SERVICE_ENTRY_POINTS:
        print(f"usage: {argv[0]} <{'|'.join(SERVICE_ENTRY_POINTS)}>", file=sys.stderr)
        return 2
    service = argv[1]
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ENV", "test")
    print(f"verifying service image: {service}")
    failures = verify(service)
    if failures:
        print(f"\nFAILED ({service}):")
        for line in failures:
            print(f"  - {line}")
        return 1
    print(f"OK ({service})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
