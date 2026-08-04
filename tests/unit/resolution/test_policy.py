"""§8/§16 decision policy guard rails — 'similarity alone never auto-links',
'domain equality alone never auto-links', 'missing runner-up margin never
auto-links', 'a weak-base candidate cannot auto-link only through bonuses'.
"""

from __future__ import annotations

from src.domain.enums import ResolutionStatus
from src.resolution.policy import PolicyThresholds, decide, decide_deterministic
from src.resolution.scoring import ScoredCandidate


def _candidate(**overrides) -> ScoredCandidate:
    base = dict(
        entity_id="e1", entity_type="Account", name="Volkswagen Group",
        lexical=0.9, semantic=None, base=0.9, rel_bonus=0.1, final=0.95,
        relational_signals=("relational:seller_open_opportunity", "relational:participant_account"),
    )
    base.update(overrides)
    return ScoredCandidate(**base)


def test_strong_candidate_with_signals_and_margin_auto_links():
    candidate = _candidate(base=0.85, final=0.95)
    assert decide(candidate, margin=0.10) == ResolutionStatus.AUTO_LINKED


def test_no_relational_signals_never_auto_links_even_with_high_scores():
    candidate = _candidate(base=0.9, final=0.99, relational_signals=())
    assert decide(candidate, margin=0.5) != ResolutionStatus.AUTO_LINKED


def test_insufficient_margin_forces_review_not_auto_link():
    candidate = _candidate(base=0.9, final=0.95)
    assert decide(candidate, margin=0.02) == ResolutionStatus.PENDING_REVIEW


def test_weak_base_cannot_auto_link_via_bonus_alone():
    """base below threshold, even with final pushed to 0.95 by bonus and full
    signal count and large margin — auto-link must still fail."""
    candidate = _candidate(base=0.5, final=0.95, relational_signals=("a", "b", "c"))
    assert decide(candidate, margin=0.5) != ResolutionStatus.AUTO_LINKED


def test_domain_equality_alone_is_a_single_signal_and_cannot_carry_autolink():
    """A candidate whose only relational signal is domain equality, with a weak
    base, must not auto-link — domain equality contributes like any other
    single relational signal, it doesn't get special auto-link power."""
    candidate = _candidate(base=0.6, final=0.65, relational_signals=("domain_equality",))
    assert decide(candidate, margin=0.5) != ResolutionStatus.AUTO_LINKED


def test_moderate_final_without_autolink_conditions_goes_to_review():
    candidate = _candidate(base=0.5, final=0.6, relational_signals=())
    assert decide(candidate, margin=0.5) == ResolutionStatus.PENDING_REVIEW


def test_low_final_is_unresolved():
    candidate = _candidate(base=0.2, final=0.3, relational_signals=())
    assert decide(candidate, margin=0.0) == ResolutionStatus.UNRESOLVED


def test_no_candidate_is_unresolved():
    assert decide(None, margin=0.0) == ResolutionStatus.UNRESOLVED


def test_custom_thresholds_are_respected():
    candidate = _candidate(base=0.6, final=0.7, relational_signals=("a",))
    strict = PolicyThresholds(base_threshold=0.9)
    assert decide(candidate, margin=0.5, thresholds=strict) != ResolutionStatus.AUTO_LINKED


def test_deterministic_unique_match_auto_links():
    assert decide_deterministic("account-123") == ResolutionStatus.AUTO_LINKED


def test_deterministic_no_match_is_unresolved():
    assert decide_deterministic(None) == ResolutionStatus.UNRESOLVED
