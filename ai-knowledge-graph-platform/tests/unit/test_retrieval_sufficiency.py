from graphrag.retrieval.sufficiency import assess_retrieval_sufficiency, abstention_message


def test_sufficiency_accepts_cited_global_evidence_without_local_chunks():
    result = assess_retrieval_sufficiency(
        chunks=[], citations=["community:42"], conflicts=[], min_evidence=1,
    )
    assert result.sufficient is True
    assert result.source_count == 1


def test_sufficiency_rejects_missing_evidence():
    result = assess_retrieval_sufficiency(chunks=[], citations=[], conflicts=[])
    assert result.sufficient is False
    assert result.reason_code == "insufficient_evidence"
    assert "authorized evidence" in abstention_message(result.reason_code)


def test_sufficiency_rejects_unresolved_conflicts_before_score_checks():
    result = assess_retrieval_sufficiency(
        chunks=[{"final_score": 0.99, "_doc_name": "a"}],
        citations=["a"], conflicts=[{"id": "conflict-1"}],
    )
    assert result.sufficient is False
    assert result.reason_code == "unresolved_conflict"


def test_sufficiency_honours_opt_in_score_floor():
    result = assess_retrieval_sufficiency(
        chunks=[{"final_score": 0.2}], citations=["a"], conflicts=[], min_average_score=0.5,
    )
    assert result.sufficient is False
    assert result.reason_code == "low_evidence_score"
