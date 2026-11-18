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
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    assert corpus.names == ["A-doc", "B-doc"]
    assert corpus.contexts == ["[Document: A-doc]\nfirst A", "[Document: B-doc]\nfirst B\n\nsecond B"]
    assert corpus.context == "\n\n".join(corpus.contexts)
    assert [d["superseded"] for d in corpus.documents] == [True, False]


def test_superseded_documents_are_excluded_by_default_like_retrieval_and_the_exclusion_is_recorded():
    corpus = fc.build_corpus("t", _ordered(ROWS))
    assert corpus.names == ["B-doc"] and corpus.excluded_superseded == ("A-doc",)
    assert "first A" not in corpus.context
    report = fc.corpus_report(corpus, {})
    assert report["superseded_documents_excluded"] == ["A-doc"] and report["superseded_documents_included"] == []
    assert "same text for all profiles" in report["prompt"]


def test_load_corpus_fails_when_the_tenant_has_no_documents():
    client = SimpleNamespace(run=AsyncMock(return_value=[]))
    with patch.object(fc, "get_neo4j", return_value=client):
        with pytest.raises(ValueError, match="no documents"):
            asyncio.run(fc.load_corpus("empty"))


def test_report_counts_size_checks_fit_and_lists_superseded_documents():
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    report = fc.corpus_report(corpus, {}, window=131_072)
    assert (report["n_documents"], report["n_chunks"]) == (2, 3)
    assert report["fits"] is True and report["corpus_tokens"] > 0
    assert report["superseded_documents_included"] == ["A-doc"]
    assert len(report["context_sha256"]) == 64
    fc.require_fit(report)


def test_a_corpus_larger_than_the_window_is_refused_before_any_model_call():
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    report = fc.corpus_report(corpus, {}, window=10_000, output_reserve=9_990)
    assert report["fits"] is False
    with pytest.raises(ValueError, match="does not fit"):
        fc.require_fit(report)


def _answer(corpus, generated, instruction=None):
    llm = SimpleNamespace(generate=AsyncMock(return_value=generated))
    with patch.object(fc, "get_settings", return_value=SimpleNamespace(retrieval={})), \
            patch.object(fc, "get_llm", return_value=llm), \
            patch.object(fc, "get_generation_route", return_value={"primary": "groq:m"}):
        result = asyncio.run(fc.answer_full_context("What?", corpus, tenant="t", citation_instruction=instruction))
    return result, llm


def test_the_answer_is_returned_as_generated_and_the_benchmark_owns_citation_parsing():
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    result, _ = _answer(corpus, "A supersedes B [DOC: A-doc].")
    assert result.answer == "A supersedes B [DOC: A-doc]." and result.citations == []


def test_the_shared_citation_rule_is_placed_in_the_prompt_exactly_as_for_the_retrievers():
    from graphrag.core.prompt_security import escape_prompt_data
    from graphrag.evaluation import citations
    from graphrag.retrieval.answer_policy import answer_prompt

    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    instruction = citations.citation_instruction(corpus.names)
    _, llm = _answer(corpus, "x", instruction)
    prompt = llm.generate.await_args.args[0]
    assert instruction in prompt and "[DOC: <id>]" in prompt and "  - A-doc\n  - B-doc" in prompt
    assert prompt == answer_prompt({"eval_citation_instruction": instruction}).format(
        context=escape_prompt_data(corpus.context), question="What?")


def test_without_the_rule_the_prompt_has_no_citation_instruction():
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    _, llm = _answer(corpus, "x")
    assert "[DOC:" not in llm.generate.await_args.args[0]


def test_the_answer_is_one_generator_call_with_the_whole_corpus_and_the_standard_prompt():
    corpus = fc.build_corpus("t", _ordered(ROWS), include_superseded=True)
    result, llm = _answer(corpus, "A-doc says first A.")
    llm.generate.assert_awaited_once()
    prompt = llm.generate.await_args.args[0]
    assert "first A" in prompt and "second B" in prompt and "Question: What?" in prompt
    assert "Answer using ONLY the information in the context" in prompt
    assert result.retrieval_mode == "full_context"
    assert result.contexts == corpus.contexts            # the judge sees the whole corpus
    assert result.model_version == "groq:m"


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
