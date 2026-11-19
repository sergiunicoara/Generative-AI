"""Answer-attributed citations: the ``[DOC: id]`` rule, its parser and the strict normaliser (pure, no disk, no model).

The retriever's ``citations`` list is provenance: everything the retrieval touched. This module is the other half:
the documents the *answer* names. When the opt-in ``answer_citation_tags_enabled`` is on, the answer prompt carries one
citation rule listing the tenant's documents; the model tags what it used as ``[DOC: <id>]`` (a bare ``[<exact id>]``
is accepted too); the tags are parsed, mapped to canonical document ids by exact alias match, de-duplicated and
returned as ``cited``. The evaluation (``graphrag.evaluation.citations``) re-exports these functions and adds the
scoring, so the benchmark and production read the same tags the same way.

Nothing here reads the disk: the alias index and the canonical ids are passed in.
"""
from __future__ import annotations

import re

_TAG = re.compile(r"\[(?:DOC:\s*([^\]]*)|([^\[\]\n:]{1,80}))\]", re.IGNORECASE)
ACCEPTED_TAG_FORMS = [
    "[DOC: <id>] (prefixed; several ids may be separated by commas or semicolons)",
    "[<exact canonical id>] (bare; case-insensitive, no fuzzy matching)",
]


def canonical_id(filename: str) -> str:
    """The id a document is cited by: its file name without a trailing ``.txt``."""
    return filename[:-4] if filename.lower().endswith(".txt") else filename


def citation_instruction(ids: list[str]) -> str:
    """The one citation rule shared by every profile (appended to the answer prompt's rules)."""
    listed = "\n".join(f"  - {name}" for name in ids)
    return (
        "- Citation format (required): after every statement supported by a document, cite that document as "
        "[DOC: <id>], using exactly one id from this list and nothing else:\n"
        f"{listed}\n"
        "  Cite only documents that support your answer; do not cite a document you did not use.\n"
    )


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


def strict_citations(raw: list[str], index: dict[str, str]) -> list[str]:
    """Canonical ids of the raw citation strings, unmapped strings dropped, first-seen order, no duplicates."""
    seen: list[str] = []
    for value in raw:
        canonical = index.get(_norm(value))
        if canonical and canonical not in seen:
            seen.append(canonical)
    return seen


def parse_answer_tags(answer: str, canonical_ids: set[str]) -> dict:
    """Tags of one answer: ``{"answer": text without the tags, "ids": raw ids in order, "n_prefixed", "n_bare"}``.

    Prefixed ``[DOC: id]`` tags are always read. A bare ``[id]`` is read only when ``id`` equals a canonical
    document id (case-insensitive, nothing fuzzy); any other bracketed text is left in the answer untouched."""
    canonical = {value.casefold() for value in canonical_ids}
    ids: list[str] = []
    counts = {"prefixed": 0, "bare": 0}

    def take(match: re.Match) -> str:
        if match.group(1) is not None:
            found = [part.strip() for part in re.split(r"[,;]", match.group(1)) if part.strip()]
            ids.extend(found)
            counts["prefixed"] += len(found)
            return ""
        candidate = match.group(2).strip()
        if candidate.casefold() in canonical:
            ids.append(candidate)
            counts["bare"] += 1
            return ""
        return match.group(0)

    stripped = re.sub(r"[ \t]+([.,;:])", r"\1", _TAG.sub(take, answer))
    return {"answer": re.sub(r"[ \t]{2,}", " ", stripped).strip(), "ids": ids,
            "n_prefixed": counts["prefixed"], "n_bare": counts["bare"]}


def tag_plan(cfg: dict, document_names: list[str]) -> tuple[str, list[str]] | None:
    """``(instruction, canonical ids)`` when the option is on and usable for this tenant, else ``None``.

    ``None`` means the prompt and the answer stay exactly as without the option. It fails closed: no documents, or
    more than ``answer_citation_tags_max_documents`` (the rule lists every document, one line each, so it grows with
    the corpus), disables the option for the query instead of sending a partial or oversized list."""
    if not cfg.get("answer_citation_tags_enabled", False):
        return None
    ids = sorted({canonical_id(name) for name in document_names if name})
    if not ids or len(ids) > int(cfg.get("answer_citation_tags_max_documents", 50)):
        return None
    return citation_instruction(ids), ids


def attribute(answer: str, ids: list[str]) -> tuple[str, list[str], dict]:
    """``(answer without tags, cited canonical ids, parsed tag counts)`` for a tenant whose canonical ids are ``ids``."""
    parsed = parse_answer_tags(answer, set(ids))
    try:
        index = build_alias_index({name: [] for name in ids})
    except ValueError:            # two ids that normalise to the same key: nothing can be mapped safely
        return parsed["answer"], [], parsed
    return parsed["answer"], strict_citations(parsed["ids"], index), parsed
