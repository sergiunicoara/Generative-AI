"""Strict, profile-independent citation scoring for the ablation (evaluation only).

Every profile's raw citation strings go through the same normaliser: each is mapped to a canonical
document id by exact alias match (``evals/citation_aliases.json``), anything that does not map
("Aircraft", "Propellers", entity names) is dropped, and the ids are de-duplicated. Recall and
precision are then computed on canonical ids only, so over-citing is penalised and a profile that
cites many strings is not rewarded for it.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from graphrag.retrieval import answer_citations as _pure
from graphrag.retrieval.answer_citations import (  # noqa: F401  (re-exported: the benchmark and the tests import them here)
    ACCEPTED_TAG_FORMS, _TAG, _norm, build_alias_index, citation_instruction)

ALIAS_MAP_PATH = Path(__file__).resolve().parents[2] / "evals" / "citation_aliases.json"
NORMALISER = "strict-v1"
CITATION_FAILURE = "insufficient citation recall"
DEFAULT_MIN_RECALL = 0.60  # same default as scripts/run_golden_eval.py

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _f1_of(precision_value: float | None, recall_value: float | None) -> float | None:
    if recall_value is None:
        return None
    if not precision_value or not recall_value:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def set_metrics(question: dict, ids: list[str] | None, citable: list[str] | None = None,
                f1_on: str = "raw") -> dict:
    """Metrics of one strictly normalised citation set (all None if ``ids`` is None).

    ``recall_raw`` is against every expected citation; ``recall_citable`` only against the expected ones the
    prompt let the model cite (``citable``; None means every expected citation is citable). ``f1`` uses
    ``recall_citable`` when ``f1_on == "citable"``, else ``recall_raw``."""
    expected = _expected(question)
    citable_keys = None if citable is None else {_norm(value) for value in citable}
    expected_citable = [v for v in expected if citable_keys is None or _norm(v) in citable_keys]
    if ids is None:
        return {"recall_raw": None, "recall_citable": None, "precision": None, "f1": None,
                "n_distinct": None, "n_expected_uncitable": len(expected) - len(expected_citable)}
    have = {_norm(value) for value in ids}
    recall_citable = (sum(_norm(v) in have for v in expected_citable) / len(expected_citable)
                      if expected_citable else None)
    recall_raw = recall(question, ids)
    precision_value = precision(question, ids)
    return {"recall_raw": recall_raw, "recall_citable": recall_citable, "precision": precision_value,
            "f1": _f1_of(precision_value, recall_citable if f1_on == "citable" else recall_raw),
            "n_distinct": len(ids), "n_expected_uncitable": len(expected) - len(expected_citable)}


@lru_cache(maxsize=1)
def _load(path: str = str(ALIAS_MAP_PATH)) -> tuple[dict[str, str], str]:
    raw = Path(path).read_bytes()
    return build_alias_index(json.loads(raw)["documents"]), hashlib.sha256(raw).hexdigest()


def alias_map_sha256() -> str:
    return _load()[1]


def strict_citations(raw: list[str], index: dict[str, str] | None = None) -> list[str]:
    """Canonical ids of the raw citation strings (see graphrag.retrieval.answer_citations); the committed
    alias map is the default index."""
    return _pure.strict_citations(raw, index if index is not None else _load()[0])


def parse_answer_tags(answer: str, canonical_ids: set[str] | None = None) -> dict:
    """Tags of one answer (see graphrag.retrieval.answer_citations); the committed alias map's ids by default."""
    return _pure.parse_answer_tags(
        answer, canonical_ids if canonical_ids is not None else set(_load()[0].values()))


def parse_doc_tags(answer: str) -> tuple[str, list[str]]:
    """``(answer without the tags, ids in the tags)`` (see parse_answer_tags)."""
    parsed = parse_answer_tags(answer)
    return parsed["answer"], parsed["ids"]


def _expected(question: dict) -> list[str]:
    return [str(value) for value in question.get("expected_citations", [])]


def recall(question: dict, cited: list[str]) -> float | None:
    expected = _expected(question)
    if not expected:
        return None
    have = {_norm(value) for value in cited}
    return sum(_norm(value) in have for value in expected) / len(expected)


def precision(question: dict, cited: list[str]) -> float | None:
    """Share of the cited ids that are expected; None without expected citations or without any citation."""
    expected = {_norm(value) for value in _expected(question)}
    if not expected or not cited:
        return None
    return sum(_norm(value) in expected for value in cited) / len(cited)


def f1(question: dict, cited: list[str]) -> float | None:
    """Per-question citation F1 on canonical ids: None without expected citations, 0.0 when nothing is cited."""
    recall_value = recall(question, cited)
    if recall_value is None:
        return None
    precision_value = precision(question, cited)
    if not precision_value or not recall_value:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def citation_failure(question: dict, cited: list[str], citable: list[str] | None = None) -> str | None:
    """The golden check's citation rule (run_golden_eval._check) applied to canonical ids only.

    With ``citable`` the rule applies to the expected ids that are also citable (the ids the prompt offered);
    a question whose expected ids are all uncitable then has no citation requirement."""
    expected = _expected(question)
    if citable is not None:
        offered = {_norm(value) for value in citable}
        expected = [value for value in expected if _norm(value) in offered]
    if not expected:
        return None
    have = {_norm(value) for value in cited}
    missing = [value for value in expected if _norm(value) not in have]
    threshold = question.get("min_citation_recall", question.get("citation_recall_min", DEFAULT_MIN_RECALL))
    if missing and len(missing) / len(expected) > (1 - threshold):
        return f"{CITATION_FAILURE} — missing: {missing}"
    return None
