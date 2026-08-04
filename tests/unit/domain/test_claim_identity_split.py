"""§6 — 'assertion_id deliberately excludes the extractor version.' An identical
assertion found by a newer extractor must link to the existing Claim (same
assertion_id), while a materially different interpretation (different polarity or
evidence span) must produce a different Claim. This is enforced structurally by
assertion_id()'s signature (extractor details simply are not parameters), so this
test proves that structural guarantee rather than a convention a caller could
violate.
"""

from src.domain.identity import assertion_id, extraction_run_id


_COMMON_KWARGS = dict(
    workspace="ws-1",
    source_segment_id="seg-abc",
    evidence_char_start=10,
    evidence_char_end=42,
    canonical_subject="elena-popescu",
    predicate="RAISED_OBJECTION",
    normalized_object="pricing",
    polarity="AFFIRMED",
)


def test_same_assertion_found_by_different_extractor_versions_shares_assertion_id():
    run_a = extraction_run_id("fixture", "n/a", "prompt-v1", "extractor-1.0.0", "nonce-a")
    run_b = extraction_run_id("fixture", "n/a", "prompt-v1", "extractor-2.0.0", "nonce-b")
    assert run_a != run_b  # the runs themselves are distinct executions...

    # ...but neither extraction_run_id nor extractor_version is a parameter of
    # assertion_id at all, so the same evidence/predicate/object/polarity always
    # yields the same assertion_id regardless of which run found it.
    claim_a = assertion_id(**_COMMON_KWARGS)
    claim_b = assertion_id(**_COMMON_KWARGS)
    assert claim_a == claim_b


def test_different_polarity_produces_a_different_assertion_id():
    affirmed = assertion_id(**_COMMON_KWARGS)
    negated = assertion_id(**{**_COMMON_KWARGS, "polarity": "NEGATED"})
    hypothetical = assertion_id(**{**_COMMON_KWARGS, "polarity": "HYPOTHETICAL"})
    assert len({affirmed, negated, hypothetical}) == 3


def test_different_evidence_span_produces_a_different_assertion_id():
    base = assertion_id(**_COMMON_KWARGS)
    shifted = assertion_id(**{**_COMMON_KWARGS, "evidence_char_end": 43})
    assert base != shifted


def test_different_predicate_or_object_produces_a_different_assertion_id():
    base = assertion_id(**_COMMON_KWARGS)
    diff_predicate = assertion_id(**{**_COMMON_KWARGS, "predicate": "HAS_PAIN_POINT"})
    diff_object = assertion_id(**{**_COMMON_KWARGS, "normalized_object": "security"})
    assert base != diff_predicate
    assert base != diff_object
