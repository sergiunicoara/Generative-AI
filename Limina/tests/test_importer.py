import json

import pytest

from limina_benchmark.importer import import_recruiter_history, load_dataset, write_dataset

pytestmark = pytest.mark.offline_eval


def test_importer_converts_and_sanitizes_historical_results(tmp_path):
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "eval_data.json").write_text(
        json.dumps([{"id": "case-1", "user_message": "Contact me at person@example.com", "expected_role": "AI Engineer"}]),
        encoding="utf-8",
    )
    (tmp_path / "eval_results.json").write_text(
        json.dumps({"results": [{"case_id": "case-1", "passed": True, "score": 5, "raw": {"chat": {"reply": "Call +40 712 345 678", "state": {"role": "AI Engineer"}}}}]}),
        encoding="utf-8",
    )

    cases = import_recruiter_history(tmp_path)

    assert len(cases) == 1
    assert cases[0].redaction_applied is True
    assert "[REDACTED_EMAIL]" in cases[0].trajectory[0].text
    assert "[REDACTED_PHONE]" in cases[0].trajectory[1].text

    destination = tmp_path / "dataset.json"
    write_dataset(cases, destination)
    assert load_dataset(destination)[0].case_id == "case-1"
