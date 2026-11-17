"""full_context ablation profile: corpus assembly, fit check, citations, one generator call (offline)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from graphrag.evaluation import full_context as fc  # noqa: E402
from scripts import benchmark_retrieval_ablation as ablation  # noqa: E402

ROWS = [
    {"filename": "B-doc.txt", "superseded_by": None, "idx": 1, "text": "second B"},
    {"filename": "A-doc.txt", "superseded_by": "x", "idx": 0, "text": "first A"},
    {"filename": "B-doc.txt", "superseded_by": None, "idx": 0, "text": "first B"},
]


def _ordered(rows):  # the Cypher query sorts by filename, chunk_index
    return sorted(rows, key=lambda r: (r["filename"], r["idx"]))


def test_corpus_groups_chunks_by_document_in_order_and_labels_each_document():
    corpus = fc.build_corpus("t", _ordered(ROWS))
    assert corpus.names == ["A-doc", "B-doc"]
    assert corpus.contexts == ["[Document: A-doc]\nfirst A", "[Document: B-doc]\nfirst B\n\nsecond B"]
    assert corpus.context == "\n\n".join(corpus.contexts)
    assert [d["superseded"] for d in corpus.documents] == [True, False]


def test_load_corpus_fails_when_the_tenant_has_no_documents():
    client = SimpleNamespace(run=AsyncMock(return_value=[]))
    with patch.object(fc, "get_neo4j", return_value=client):
        with pytest.raises(ValueError, match="no documents"):
            asyncio.run(fc.load_corpus("empty"))


def test_report_counts_size_checks_fit_and_lists_superseded_documents():
    corpus = fc.build_corpus("t", _ordered(ROWS))
    report = fc.corpus_report(corpus, {}, window=131_072)
    assert (report["n_documents"], report["n_chunks"]) == (2, 3)
    assert report["fits"] is True and report["corpus_tokens"] > 0
    assert report["superseded_documents_included"] == ["A-doc"]
    assert len(report["context_sha256"]) == 64
    fc.require_fit(report)


def test_a_corpus_larger_than_the_window_is_refused_before_any_model_call():
    corpus = fc.build_corpus("t", _ordered(ROWS))
    report = fc.corpus_report(corpus, {}, window=10_000, output_reserve=9_990)
    assert report["fits"] is False
    with pytest.raises(ValueError, match="does not fit"):
        fc.require_fit(report)


AD, INSPECTION = "FAA-AD-2024-01-02", "G-ABCD_inspection_2024-01"


@pytest.mark.parametrize("answer, expected", [
    ("AD 2024 supersedes it (FAA-AD-2024-01-02).", [AD]),
    ("see faa-ad-2024-01-02 and G-ABCD inspection 2024-01", [AD, INSPECTION]),
    ("FAA‑AD‑2024‑01‑02 applies", [AD]),  # U+2011 non-breaking hyphens
    ("no document is named here", []),
])
def test_citations_are_the_documents_named_in_the_answer(answer, expected):
    assert fc.cited_documents(answer, [AD, INSPECTION]) == expected


def test_the_answer_is_one_generator_call_with_the_whole_corpus_and_the_standard_prompt():
    corpus = fc.build_corpus("t", _ordered(ROWS))
    llm = SimpleNamespace(generate=AsyncMock(return_value="A-doc says first A."))
    with patch.object(fc, "get_settings", return_value=SimpleNamespace(retrieval={})), \
            patch.object(fc, "get_llm", return_value=llm), \
            patch.object(fc, "get_generation_route", return_value={"primary": "groq:m"}):
        result = asyncio.run(fc.answer_full_context("What?", corpus, tenant="t"))
    llm.generate.assert_awaited_once()
    prompt = llm.generate.await_args.args[0]
    assert "first A" in prompt and "second B" in prompt and "Question: What?" in prompt
    assert "Answer using ONLY the information in the context" in prompt
    assert result.retrieval_mode == "full_context"
    assert result.contexts == corpus.contexts            # the judge sees the whole corpus
    assert result.citations == ["A-doc"] and result.model_version == "groq:m"


def test_profile_registry_accepts_full_context_and_still_rejects_unknown_profiles():
    assert ablation.retrieval_profile_overrides("full_context") == {}
    assert ablation.retrieval_profile_overrides("vector_only")["bm25_enabled"] is False
    with pytest.raises(ValueError, match="Unknown retrieval profile"):
        ablation.retrieval_profile_overrides("nonsense")


def test_run_metadata_carries_the_full_context_block_only_for_that_profile():
    from graphrag.evaluation.run_metadata import collect_run_metadata

    kwargs = dict(tenant="t", profile="p", profile_overrides={}, mode="local", config_overrides={},
                  models={}, golden={}, pinned=True)
    assert "full_context" not in collect_run_metadata(**kwargs)
    assert collect_run_metadata(**kwargs, extra={"fits": True})["full_context"] == {"fits": True}
