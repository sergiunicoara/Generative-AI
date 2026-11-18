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
import re
from functools import lru_cache
from pathlib import Path

ALIAS_MAP_PATH = Path(__file__).resolve().parents[2] / "evals" / "citation_aliases.json"
NORMALISER = "strict-v1"
CITATION_FAILURE = "insufficient citation recall"
DEFAULT_MIN_RECALL = 0.60  # same default as scripts/run_golden_eval.py

def citation_instruction(ids: list[str]) -> str:
    """The one citation rule shared by every ablation profile (appended to the answer prompt's rules)."""
    listed = "\n".join(f"  - {name}" for name in ids)
    return (
        "- Citation format (required): after every statement supported by a document, cite that document as "
        "[DOC: <id>], using exactly one id from this list and nothing else:\n"
        f"{listed}\n"
        "  Cite only documents that support your answer; do not cite a document you did not use.\n"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def set_metrics(question: dict, ids: list[str] | None) -> dict:
    """Recall, precision, F1 and distinct count of one strictly normalised citation set (all None if ``ids`` is None)."""
    if ids is None:
        return {"recall": None, "precision": None, "f1": None, "n_distinct": None}
    return {"recall": recall(question, ids), "precision": precision(question, ids),
            "f1": f1(question, ids), "n_distinct": len(ids)}


_DOC_TAG = re.compile(r"\[DOC:\s*([^\]]*)\]", re.IGNORECASE)


def _norm(text: str) -> str:
    text = re.sub(r"\.txt$", "", str(text).strip(), flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-z]+", " ", text.casefold()).strip()


def build_alias_index(document_aliases: dict[str, list[str]]) -> dict[str, str]:
    """normalised alias -> canonical id; an alias that denotes two documents is a data error."""
    index: dict[str, str] = {}
    for canonical, aliases in document_aliases.items():
        for alias in {canonical, *aliases}:
            key = _norm(alias)
            if index.setdefault(key, canonical) != canonical:
                raise ValueError(f"alias {alias!r} denotes both {index[key]!r} and {canonical!r}")
    return index


@lru_cache(maxsize=1)
def _load(path: str = str(ALIAS_MAP_PATH)) -> tuple[dict[str, str], str]:
    raw = Path(path).read_bytes()
    return build_alias_index(json.loads(raw)["documents"]), hashlib.sha256(raw).hexdigest()


def alias_map_sha256() -> str:
    return _load()[1]


def strict_citations(raw: list[str], index: dict[str, str] | None = None) -> list[str]:
    """Canonical ids of the raw citation strings, unmapped strings dropped, first-seen order, no duplicates."""
    index = index if index is not None else _load()[0]
    seen: list[str] = []
    for value in raw:
        canonical = index.get(_norm(value))
        if canonical and canonical not in seen:
            seen.append(canonical)
    return seen


def parse_doc_tags(answer: str) -> tuple[str, list[str]]:
    """``(answer without [DOC: id] tags, ids in the tags)``. A tag may list ids separated by commas or semicolons."""
    ids = [part.strip() for tag in _DOC_TAG.findall(answer) for part in re.split(r"[,;]", tag) if part.strip()]
    stripped = re.sub(r"[ \t]+([.,;:])", r"\1", _DOC_TAG.sub("", answer))
    return re.sub(r"[ \t]{2,}", " ", stripped).strip(), ids


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


def citation_failure(question: dict, cited: list[str]) -> str | None:
    """The golden check's citation rule (run_golden_eval._check) applied to canonical ids only."""
    expected = _expected(question)
    if not expected:
        return None
    have = {_norm(value) for value in cited}
    missing = [value for value in expected if _norm(value) not in have]
    threshold = question.get("min_citation_recall", question.get("citation_recall_min", DEFAULT_MIN_RECALL))
    if missing and len(missing) / len(expected) > (1 - threshold):
        return f"{CITATION_FAILURE} — missing: {missing}"
    return None
