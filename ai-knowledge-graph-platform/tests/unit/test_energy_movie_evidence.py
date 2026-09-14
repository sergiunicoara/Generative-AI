"""Keep presentation mapping counts tied to the successful workflow capture."""

import json
from pathlib import Path

import pytest

from docs.presentation.energy_movie_evidence import mapping_counts


def test_current_mapping_capture():
    assert mapping_counts() == (13, 3)


@pytest.mark.parametrize("success,code,stdout", [
    (False, 0, "entity_rows=13 relation_rows=3"),
    (True, 1, "entity_rows=13 relation_rows=3"),
    (True, 0, "entity_rows=13"),
    (True, 0, "no counts available"),
])
def test_missing_or_failed_evidence_is_not_presented(tmp_path: Path, success, code, stdout):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({
        "all_succeeded": success,
        "commands": {"validate_r2rml": {"returncode": code, "stdout": stdout}},
    }))
    with pytest.raises(ValueError):
        mapping_counts(path)


def test_changed_counts_are_read_from_capture(tmp_path: Path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({
        "all_succeeded": True,
        "commands": {"validate_r2rml": {"returncode": 0, "stdout": "entity_rows=21 relation_rows=7"}},
    }))
    assert mapping_counts(path) == (21, 7)
