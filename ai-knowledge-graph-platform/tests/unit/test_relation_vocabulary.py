"""Unit tests for the relation-vocabulary analysis (scripts/analyze_relation_vocabulary.py).

The tool's whole value is that its *predictions* can be trusted before a
migration is run, so these tests pin the arithmetic (noisy-OR fusion, edge
collapse, manufactured multi_source conflicts, degree shifts) and the safety
guards (inverse relations are never auto-merged, review-band scores are never
auto-applied).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import analyze_relation_vocabulary as arv  # noqa: E402

EdgeRow = arv.EdgeRow


def edge(src, tgt, rel, conf=0.5, docs=()):
    return EdgeRow(src=src, tgt=tgt, relation=rel, confidence=conf, source_doc_ids=tuple(docs))


# ── Inverse guard ─────────────────────────────────────────────────────────────

class TestInverseGuard:
    """Merging a relation with its inverse would reverse direction for half the
    edges and manufacture directional_reversal conflicts."""

    def test_declared_inverse_pair_is_detected(self):
        pairs = arv.build_inverse_pairs({
            "MANDATED_BY": {"domain": ["REGULATION"], "inverse": "MANDATES"},
        })
        assert arv.is_probable_inverse("MANDATED_BY", "MANDATES", pairs)

    def test_opposite_voice_is_treated_as_inverse(self):
        # lexically near-identical, semantically opposite
        assert arv.is_probable_inverse("SUPPLIES_TO", "SUPPLIED_BY", set())
        assert arv.is_probable_inverse("OPERATES", "OPERATED_BY", set())

    def test_same_voice_is_not_an_inverse(self):
        assert not arv.is_probable_inverse("MANUFACTURES_AT", "MANUFACTURES_IN", set())

    def test_has_x_versus_is_x_of_is_an_inverse(self):
        # found live: the aerospace vocabulary carries both, and they rank as a
        # strong fuzzy+embedding match despite being converses
        assert arv.is_probable_inverse("IS_VARIANT_OF", "HAS_VARIANT", set())
        assert not arv.propose_canonical_map(
            {"IS_VARIANT_OF": 5}, {"HAS_VARIANT"}
        ).canonical_map

    def test_derived_from_is_converse(self):
        assert arv.is_probable_inverse("DERIVED_FROM", "DERIVES", set())

    def test_two_of_suffixed_names_share_voice(self):
        # both converse — the guard must not block legitimate merges
        assert not arv.is_probable_inverse("MEMBER_OF", "PART_OF", set())

    def test_identical_names_are_not_inverses(self):
        assert not arv.is_probable_inverse("SUPPLIES", "SUPPLIES", set())

    def test_inverse_is_never_auto_merged(self):
        counts = {"OPERATED_BY": 10, "OPERATES": 8}
        proposal = arv.propose_canonical_map(counts, {"OPERATES"})
        assert "OPERATED_BY" not in proposal.canonical_map

    def test_build_inverse_pairs_ignores_malformed_specs(self):
        pairs = arv.build_inverse_pairs({"A": None, "B": {"no_inverse": 1}, "C": "str"})
        assert pairs == set()


# ── Proposal cascade ──────────────────────────────────────────────────────────

class TestProposal:
    def test_normalized_variant_maps_to_canonical(self):
        # "MANUFACTURES AT" normalizes identically to "MANUFACTURES_AT"
        proposal = arv.propose_canonical_map({"MANUFACTURES AT": 3}, {"MANUFACTURES_AT"})
        assert proposal.canonical_map["MANUFACTURES AT"] == "MANUFACTURES_AT"

    def test_fuzzy_above_threshold_maps(self):
        proposal = arv.propose_canonical_map({"MANUFACTURES_IN": 5}, {"MANUFACTURES_AT"})
        assert proposal.canonical_map["MANUFACTURES_IN"] == "MANUFACTURES_AT"

    def test_review_band_is_reported_not_applied(self):
        counts = {"SUPPLIES_COMPONENTS": 4}
        proposal = arv.propose_canonical_map(
            counts, {"SUPPLIES"}, fuzzy_threshold=95.0, review_fuzzy_min=40.0
        )
        assert proposal.canonical_map == {}
        assert proposal.ambiguous[0]["relation"] == "SUPPLIES_COMPONENTS"
        assert proposal.ambiguous[0]["band"] == "canonical_vocabulary"

    def test_unrelated_name_is_left_alone(self):
        proposal = arv.propose_canonical_map({"ZZZ_QQQ_XYZ": 2}, {"MANUFACTURES"})
        assert "ZZZ_QQQ_XYZ" not in proposal.canonical_map
        assert "ZZZ_QQQ_XYZ" in proposal.unmapped

    def test_already_canonical_name_is_not_remapped(self):
        proposal = arv.propose_canonical_map({"MANUFACTURES": 9}, {"MANUFACTURES"})
        assert proposal.canonical_map == {}

    def test_leftovers_cluster_with_highest_count_as_head(self):
        # neither matches the declared vocabulary; the frequent one wins
        counts = {"COLLABORATES_WITH": 12, "COLLABORATES_WITHH": 2}
        proposal = arv.propose_canonical_map(counts, {"MANUFACTURES"})
        assert proposal.canonical_map["COLLABORATES_WITHH"] == "COLLABORATES_WITH"

    def test_embedding_merges_cross_language_when_fuzzy_cannot(self):
        # "LIVREAZA_CATRE" shares almost no characters with "DELIVERS_TO"
        assert arv.fuzzy_ratio("LIVREAZA_CATRE", "DELIVERS_TO") < 70
        embeddings = {"LIVREAZA_CATRE": [1.0, 0.0], "DELIVERS_TO": [1.0, 0.0]}
        proposal = arv.propose_canonical_map(
            {"LIVREAZA_CATRE": 3}, {"DELIVERS_TO"}, embeddings
        )
        assert proposal.canonical_map["LIVREAZA_CATRE"] == "DELIVERS_TO"

    def test_related_to_is_not_a_merge_target(self):
        terms, _ = arv.load_canonical_vocabulary("nonexistent-tenant")
        assert "RELATED_TO" not in terms


# ── Fusion arithmetic ─────────────────────────────────────────────────────────

class TestNoisyOr:
    def test_matches_the_write_path_formula(self):
        # 1 - (1-0.5)(1-0.5) = 0.75
        assert arv.noisy_or([0.5, 0.5]) == pytest.approx(0.75)

    def test_single_value_is_unchanged(self):
        assert arv.noisy_or([0.42]) == pytest.approx(0.42)

    def test_empty_is_zero(self):
        assert arv.noisy_or([]) == 0.0

    def test_accumulation_is_monotonic_and_bounded(self):
        assert arv.noisy_or([0.5] * 10) < 1.0
        assert arv.noisy_or([0.5] * 10) > arv.noisy_or([0.5] * 3)


# ── Simulation ────────────────────────────────────────────────────────────────

class TestSimulate:
    def test_parallel_edges_collapse(self):
        edges = [edge("A", "B", "SUPPLIES_TO"), edge("A", "B", "DELIVERS_TO")]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["edges_before"] == 2
        assert sim["edges_after"] == 1
        assert sim["edges_collapsed"] == 1
        assert sim["relations_before"] == 2
        assert sim["relations_after"] == 1

    def test_empty_map_is_a_no_op(self):
        edges = [edge("A", "B", "X"), edge("B", "C", "Y")]
        sim = arv.simulate(edges, {})
        assert sim["edges_before"] == sim["edges_after"] == 2
        assert sim["new_multi_source_conflicts"] == 0

    def test_merge_across_two_documents_manufactures_a_conflict(self):
        # strategy 1 fires on size(source_doc_ids) > 1
        edges = [
            edge("A", "B", "SUPPLIES_TO", docs=["doc1"]),
            edge("A", "B", "DELIVERS_TO", docs=["doc2"]),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["new_multi_source_conflicts"] == 1

    def test_merge_within_one_document_manufactures_nothing(self):
        edges = [
            edge("A", "B", "SUPPLIES_TO", docs=["doc1"]),
            edge("A", "B", "DELIVERS_TO", docs=["doc1"]),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["new_multi_source_conflicts"] == 0

    def test_already_multi_source_edge_is_not_double_counted(self):
        edges = [
            edge("A", "B", "SUPPLIES_TO", docs=["doc1", "doc2"]),   # already flagged
            edge("A", "B", "DELIVERS_TO", docs=["doc3"]),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["new_multi_source_conflicts"] == 0

    def test_conflict_rate_uses_post_merge_edge_count(self):
        edges = [
            edge("A", "B", "SUPPLIES_TO", docs=["d1"]),
            edge("A", "B", "DELIVERS_TO", docs=["d1"]),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"}, open_conflicts=1)
        # rate rises even with no new conflicts, because the denominator shrinks
        assert sim["conflicts_per_1k_edges_before"] == pytest.approx(500.0)
        assert sim["conflicts_per_1k_edges_after"] == pytest.approx(1000.0)

    def test_fused_confidence_crosses_the_gnn_threshold(self):
        # neither edge alone clears 0.7; fused they reach 0.75
        edges = [
            edge("A", "B", "SUPPLIES_TO", conf=0.5),
            edge("A", "B", "DELIVERS_TO", conf=0.5),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["edges_crossing_confidence_threshold"] == 1
        assert sim["high_conf_rate_before"] == 0.0
        assert sim["high_conf_rate_after"] == 1.0

    def test_confidence_already_above_threshold_is_not_counted_as_crossing(self):
        edges = [
            edge("A", "B", "SUPPLIES_TO", conf=0.8),
            edge("A", "B", "DELIVERS_TO", conf=0.5),
        ]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["edges_crossing_confidence_threshold"] == 0

    def test_merging_lowers_degree_and_can_release_a_quarantine_flag(self):
        # HUB is inflated purely by 30 parallel spellings to one neighbour;
        # collapsing them drops it back below mean * multiplier.
        edges = [edge("HUB", "X", f"REL_{i}") for i in range(30)]
        edges += [edge(f"N{i}", f"M{i}", "OTHER") for i in range(30)]
        mapping = {f"REL_{i}": "REL_0" for i in range(1, 30)}

        sim = arv.simulate(edges, mapping, degree_multiplier=5.0)
        assert "HUB" in sim["degree_quarantine_flagged_before"]
        assert "HUB" not in sim["degree_quarantine_flagged_after"]
        assert "HUB" in sim["degree_quarantine_no_longer_flagged"]

    def test_distinct_entity_pairs_are_not_merged(self):
        edges = [edge("A", "B", "SUPPLIES_TO"), edge("A", "C", "DELIVERS_TO")]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["edges_after"] == 2      # same relation, different targets

    def test_direction_is_part_of_edge_identity(self):
        edges = [edge("A", "B", "SUPPLIES_TO"), edge("B", "A", "DELIVERS_TO")]
        sim = arv.simulate(edges, {"DELIVERS_TO": "SUPPLIES_TO"})
        assert sim["edges_after"] == 2      # A->B and B->A stay distinct


class TestCosine:
    def test_orthogonal_is_zero(self):
        assert arv.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_identical_is_one(self):
        assert arv.cosine([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        assert arv.cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
