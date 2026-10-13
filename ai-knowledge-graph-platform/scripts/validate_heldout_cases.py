#!/usr/bin/env python
"""Structural check of labelled golden cases against the corpus text. Offline: no LLM, no API, no cost.

    python scripts/validate_heldout_cases.py [cases.json]

For a case with citations: every required term, and at least one alternative of every any-of group, must occur in at
least one cited document (case-, accent- and '1 %'/'1%'-insensitive). For a negative case: no forbidden term may occur
anywhere in the corpus (so an answer containing it can only be fabricated). This proves a label is *answerable* from
the cited text; it does not run or score the model.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphrag.domains.automotive.audit import fold  # noqa: E402

DEFAULT_CASES = ROOT / "data" / "eval_golden" / "queries_automotive_heldout.json"
CORPUS = ROOT / "data" / "automotive"


def _norm(text: str) -> str:
    return re.sub(r"\s+%", "%", fold(text))


def load_corpus_text(corpus_dir: Path = CORPUS) -> dict[str, str]:
    return {p.stem.lower(): _norm(p.read_text(encoding="utf-8")) for p in sorted(corpus_dir.glob("*/*.txt"))}


def validate(cases: list[dict], corpus: dict[str, str]) -> list[str]:
    problems = []
    ids = [c["id"] for c in cases]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for case in cases:
        cid, cited = case["id"], case.get("expected_citations") or []
        missing_docs = [d for d in cited if d not in corpus]
        if missing_docs:
            problems.append(f"{cid}: cited documents not in corpus: {missing_docs}")
            continue
        if case.get("type") == "negative":
            whole = " ".join(corpus.values())
            for term in case.get("forbidden_terms") or []:
                if re.search(r"(?<!\w)" + re.escape(_norm(term)) + r"(?!\w)", whole):
                    problems.append(f"{cid}: forbidden term {term!r} occurs in the corpus, so it is not proof of "
                                    "fabrication")
            continue
        if not cited:
            problems.append(f"{cid}: non-negative case without citations")
            continue
        texts = [corpus[d] for d in cited]
        for term in case.get("required_answer_terms") or []:
            if not any(_norm(term) in t for t in texts):
                problems.append(f"{cid}: required term {term!r} not found in {cited}")
        for group in case.get("required_answer_any_of") or []:
            if not any(_norm(alt) in t for alt in group for t in texts):
                problems.append(f"{cid}: none of {group!r} found in {cited}")
    return problems


def main(argv: list[str]) -> int:
    path = Path(argv[0]) if argv else DEFAULT_CASES
    cases = json.loads(path.read_text(encoding="utf-8"))["questions"]
    problems = validate(cases, load_corpus_text())
    print(json.dumps({"cases": len(cases), "problems": problems}, indent=2, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
