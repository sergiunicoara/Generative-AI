import json

import pytest

from graphrag.evaluation.graphrag_benchmark import (
    BenchmarkQuestion, ControlledRoute, load_questions, run_controlled_routes,
)


def test_adapter_loads_public_shape_and_preserves_unknown_fields(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"question_id": "q1", "question": "What?", "answer": "This.", "level": 2}) + "\n")

    question = load_questions(path)[0]

    assert question.id == "q1"
    assert question.reference == "This."
    assert question.metadata == {"level": 2}


@pytest.mark.asyncio
async def test_controlled_routes_are_fingerprinted_and_receive_only_declared_overrides():
    calls = []

    async def query(question, mode, overrides):
        calls.append((question, mode, overrides))
        return {"answer": "answer", "citations": ["c1"], "retrieval_mode": mode}

    report = await run_controlled_routes(
        [BenchmarkQuestion("q1", "question")],
        [ControlledRoute("full", "full"), ControlledRoute("vector", "vector_only")],
        query, tenant="tenant-a",
    )

    assert len(report["outputs"]) == 2
    assert report["outputs"][0]["route_fingerprint"] != report["outputs"][1]["route_fingerprint"]
    assert calls[0][2] == {}
    assert calls[1][2]["bm25_enabled"] is False
