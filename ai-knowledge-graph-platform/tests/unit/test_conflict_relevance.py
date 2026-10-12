"""Question-relevant conflict filtering for the sufficiency gate (opt-in, not golden-validated)."""
from __future__ import annotations

import pytest

from graphrag.retrieval.conflict_relevance import conflict_document_keys, filter_relevant_conflicts
from graphrag.retrieval.sufficiency import assess_retrieval_sufficiency

HEADCOUNT = {"conflict_id": "c1", "src": "Acme", "tgt": "Headcount", "relation": "HAS",
             "sources": "{'a': ['doc-hr-1'], 'b': ['doc-hr-2']}"}
TORQUE = {"conflict_id": "c2", "src": "Landing Gear", "tgt": "Torque 40Nm", "relation": "REQUIRES",
          "sources": "['doc-amm', 'doc-sb']"}


def _chunk(text, doc, score=1.0, **extra):
    return {"chunk_id": "x", "text": text, "document_id": doc, "_doc_name": f"{doc}.txt", "final_score": score, **extra}


def test_conflict_on_an_unrelated_topic_does_not_block():
    chunks = [_chunk("The landing gear torque is 40Nm.", "doc-amm")]
    relevant, ignored = filter_relevant_conflicts(
        [HEADCOUNT, TORQUE], question="What torque applies to the landing gear?", chunks=chunks)
    assert [c["conflict_id"] for c in relevant] == ["c2"]
    assert [c["conflict_id"] for c in ignored] == ["c1"]


def test_question_naming_an_entity_makes_the_conflict_relevant_even_without_matching_evidence():
    relevant, ignored = filter_relevant_conflicts(
        [HEADCOUNT], question="How large is Acme's headcount?", chunks=[_chunk("unrelated text", "doc-x")])
    assert relevant == [HEADCOUNT] and ignored == []


def test_evidence_mention_alone_is_not_enough_without_a_shared_source_document():
    chunks = [_chunk("Acme employs many people.", "doc-other")]
    relevant, ignored = filter_relevant_conflicts([HEADCOUNT], question="Tell me about suppliers", chunks=chunks)
    assert relevant == [] and ignored == [HEADCOUNT]


def test_shared_source_document_alone_is_not_enough_without_an_entity_in_the_text():
    chunks = [_chunk("A paragraph about cafeteria opening hours.", "doc-hr-1")]
    relevant, ignored = filter_relevant_conflicts([HEADCOUNT], question="When is lunch?", chunks=chunks)
    assert relevant == [] and ignored == [HEADCOUNT]


def test_evidence_and_source_overlap_make_it_relevant_regardless_of_extension_or_case():
    chunks = [_chunk("ACME had 120 staff.", "other", _doc_name="DOC-HR-2.TXT")]
    chunks[0]["_doc_name"] = "DOC-HR-2.TXT"
    relevant, _ = filter_relevant_conflicts([HEADCOUNT], question="Staff?", chunks=chunks)
    assert relevant == [HEADCOUNT]


def test_only_the_top_ranked_chunks_count_as_the_evidence_used():
    chunks = [_chunk("irrelevant", "doc-a", 0.9), _chunk("also irrelevant", "doc-b", 0.8),
              _chunk("Acme headcount is 120.", "doc-hr-1", 0.1)]
    relevant, ignored = filter_relevant_conflicts([HEADCOUNT], question="q", chunks=chunks, max_ranked_chunks=2)
    assert relevant == [] and ignored == [HEADCOUNT]
    relevant, _ = filter_relevant_conflicts([HEADCOUNT], question="q", chunks=chunks, max_ranked_chunks=3)
    assert relevant == [HEADCOUNT]


@pytest.mark.parametrize("sources", [None, "", "not a literal", "[]", "{}", "{'a': []}"])
def test_a_conflict_whose_sources_are_unknown_stays_relevant(sources):
    conflict = {**HEADCOUNT, "sources": sources}
    relevant, ignored = filter_relevant_conflicts([conflict], question="unrelated", chunks=[_chunk("x", "d")])
    assert relevant == [conflict] and ignored == []


def test_no_conflicts_and_no_chunks_are_handled():
    assert filter_relevant_conflicts([], question="q", chunks=[]) == ([], [])
    relevant, ignored = filter_relevant_conflicts([HEADCOUNT], question="q", chunks=[])
    assert ignored == [HEADCOUNT] and relevant == []


def test_source_parsing_accepts_dicts_lists_and_nested_values():
    assert conflict_document_keys("{'a': ['Doc-1.txt'], 'b': 'doc-2.PDF'}") == {"doc-1", "doc-2"}
    assert conflict_document_keys(["x.md", " ", 5]) == {"x"}
    assert conflict_document_keys("garbage(") == set()
    assert conflict_document_keys(None) == set()


def test_gate_outcome_changes_only_when_the_filter_removes_every_conflict():
    chunks = [_chunk("The landing gear torque is 40Nm.", "doc-amm")]
    full = assess_retrieval_sufficiency(chunks=chunks, citations=["doc-amm"], conflicts=[HEADCOUNT])
    assert full.reason_code == "unresolved_conflict" and not full.sufficient
    relevant, _ = filter_relevant_conflicts([HEADCOUNT], question="landing gear torque?", chunks=chunks)
    filtered = assess_retrieval_sufficiency(chunks=chunks, citations=["doc-amm"], conflicts=relevant)
    assert filtered.sufficient and filtered.conflict_count == 0


def test_the_filter_is_off_by_default_in_settings():
    from graphrag.core.config import get_settings

    assert get_settings().retrieval.get("conflict_relevance_filter_enabled", False) is False
