from __future__ import annotations

from src.resolution.deterministic import DeterministicRule, resolve_deterministic


def test_single_candidate_auto_links():
    match = resolve_deterministic(DeterministicRule.A3_EXACT_CANONICAL_NAME, ["account-1"])
    assert match is not None
    assert match.entity_id == "account-1"
    assert match.rule == DeterministicRule.A3_EXACT_CANONICAL_NAME


def test_duplicate_exact_names_do_not_deterministic_link():
    """§16: 'duplicate exact names do not deterministic-link.'"""
    match = resolve_deterministic(DeterministicRule.A3_EXACT_CANONICAL_NAME, ["account-1", "account-2"])
    assert match is None


def test_no_candidates_do_not_deterministic_link():
    match = resolve_deterministic(DeterministicRule.A1_EXACT_EXTERNAL_ID, [])
    assert match is None


def test_repeated_identical_id_is_still_unique():
    match = resolve_deterministic(DeterministicRule.A2_EXACT_NORMALIZED_EMAIL, ["contact-1", "contact-1"])
    assert match is not None
    assert match.entity_id == "contact-1"
