"""Adapter for reproducible GraphRAG-Benchmark route comparisons.

The public benchmark changes datasets independently of this repository.  This
adapter deliberately accepts JSONL records with the public minimum (question
plus an identifier) and preserves every unfamiliar field in ``metadata``. It
therefore avoids coupling production query code to a benchmark checkout while
making every route comparison traceable to an immutable input fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from graphrag.retrieval.hybrid_retriever import retrieval_profile_overrides


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    task_type: str = "open_ended"
    reference: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ControlledRoute:
    """A named retrieval configuration; no ambient settings are compared."""

    name: str
    profile: str = "full"
    mode: str = "hybrid"

    @property
    def overrides(self) -> dict[str, Any]:
        return retrieval_profile_overrides(self.profile)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {"name": self.name, "profile": self.profile, "mode": self.mode, "overrides": self.overrides},
            sort_keys=True, separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """Read GraphRAG-Benchmark-compatible JSONL without discarding gold data."""
    questions: list[BenchmarkQuestion] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        question = raw.get("question") or raw.get("query")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{number}: question/query must be a non-empty string")
        identifier = raw.get("id", raw.get("question_id", f"line-{number}"))
        reference = raw.get("answer", raw.get("ground_truth", raw.get("reference", "")))
        known = {"id", "question_id", "question", "query", "task_type", "type", "answer", "ground_truth", "reference"}
        questions.append(BenchmarkQuestion(
            id=str(identifier), question=question, task_type=str(raw.get("task_type", raw.get("type", "open_ended"))),
            reference=str(reference) if reference is not None else "",
            metadata={key: value for key, value in raw.items() if key not in known},
        ))
    if not questions:
        raise ValueError(f"{path}: no benchmark questions found")
    return questions


def dataset_fingerprint(questions: Iterable[BenchmarkQuestion]) -> str:
    rows = [asdict(question) for question in questions]
    return "sha256:" + hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


async def run_controlled_routes(
    questions: Iterable[BenchmarkQuestion],
    routes: Iterable[ControlledRoute],
    query: Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]],
    *,
    tenant: str,
) -> dict[str, Any]:
    """Run identical questions through explicitly fingerprinted routes.

    ``query`` receives ``(question, mode, config_overrides)``.  It permits a
    real ``HybridRetriever`` in production and a fake in unit tests; timing and
    route metadata are captured here rather than being guessed by a report.
    """
    question_rows = list(questions)
    fingerprint = dataset_fingerprint(question_rows)
    outputs: list[dict[str, Any]] = []
    for route in routes:
        for item in question_rows:
            started = time.monotonic()
            result = await query(item.question, route.mode, route.overrides)
            outputs.append({
                "id": item.id,
                "question": item.question,
                "task_type": item.task_type,
                "reference": item.reference,
                "response": str(result.get("answer", "")),
                "citations": list(result.get("citations", [])),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "route": route.name,
                "route_fingerprint": route.fingerprint,
                "mode": route.mode,
                "retrieval_mode": result.get("retrieval_mode", route.mode),
                "tenant": tenant,
            })
    return {
        "adapter": "graphrag-benchmark/v1",
        "dataset_fingerprint": fingerprint,
        "tenant": tenant,
        "routes": [{"name": route.name, "profile": route.profile, "mode": route.mode,
                    "fingerprint": route.fingerprint} for route in routes],
        "outputs": outputs,
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = [
    "BenchmarkQuestion", "ControlledRoute", "dataset_fingerprint", "load_questions",
    "run_controlled_routes", "write_report",
]
