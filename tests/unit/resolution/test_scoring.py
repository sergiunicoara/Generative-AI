from __future__ import annotations

from src.resolution.scoring import blend, lexical_score, rank_candidates, score_candidate


def test_lexical_score_of_identical_strings_is_one():
    assert lexical_score("Volkswagen Group", "Volkswagen Group") == 1.0


def test_lexical_score_of_close_variant_is_high_but_not_one():
    score = lexical_score("Volks Wagen", "Volkswagen Group")
    assert 0.5 < score < 1.0


def test_blend_falls_back_to_lexical_only_without_semantic():
    assert blend(0.8, None) == 0.8


def test_blend_weights_lexical_and_semantic():
    result = blend(1.0, 0.0, lexical_weight=0.6)
    assert result == 0.6


def test_rel_bonus_is_capped():
    scored = score_candidate(
        entity_id="e1", entity_type="Account", name="Volkswagen Group",
        mention_surface="Volkswagen Group",
        relational_signals=frozenset({"a", "b", "c", "d", "e", "f"}),  # 6 signals * 0.05 = 0.30, capped at 0.15
        max_rel_bonus=0.15,
    )
    assert scored.rel_bonus == 0.15


def test_final_never_exceeds_one():
    scored = score_candidate(
        entity_id="e1", entity_type="Account", name="Volkswagen Group",
        mention_surface="Volkswagen Group",
        relational_signals=frozenset({"a", "b", "c"}),
    )
    assert scored.final <= 1.0


def test_rank_candidates_computes_margin_between_top_two():
    a = score_candidate(entity_id="a", entity_type="Account", name="Volkswagen Group", mention_surface="Volkswagen Group")
    b = score_candidate(entity_id="b", entity_type="Account", name="Volkswagen Financial Services", mention_surface="Volkswagen Group")
    result = rank_candidates([b, a])
    assert result.ranked[0].entity_id == "a"
    assert result.margin == a.final - b.final


def test_rank_candidates_with_single_candidate_margin_equals_its_own_score():
    a = score_candidate(entity_id="a", entity_type="Account", name="Volkswagen Group", mention_surface="Volkswagen Group")
    result = rank_candidates([a])
    assert result.margin == a.final


def test_rank_candidates_empty_list():
    result = rank_candidates([])
    assert result.ranked == []
    assert result.margin == 0.0
