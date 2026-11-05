"""Lexical guards, type scoping, review routing and curated aliases in AliasRegistry.resolve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from graphrag.graph.alias_registry import (
    AliasRegistry,
    AmbiguousMatch,
    _normalize,
    load_curated_aliases,
    token_conflict,
    token_extra_identifier,
)

ROOT = Path(__file__).resolve().parents[2]


def _registry(*canonical: tuple[str, str], guards: bool = True) -> AliasRegistry:
    registry = AliasRegistry(None, tenant="t1")
    registry._lexical_guards = guards
    for name, type_ in canonical:
        registry._exact[_normalize(name)] = (name, type_)
    registry._rebuild_fuzzy_index()
    return registry


@pytest.mark.parametrize("raw, candidate, expected", [
    ("boeing 747 800", "boeing 737 800", "identifier_mismatch"),
    ("inverter module im 470", "inverter module im 450", "identifier_mismatch"),
    ("halvorsen rail services", "halvorsen grid services", "token_substitution"),
    ("meridian marine services", "meridian air services", "token_substitution"),
    ("honeywell aerospce", "honeywell aerospace", None),          # typo
    ("meridian air service", "meridian air services", None),      # plural
    ("hydraulic pump assy", "hydraulic pump assembly", None),     # abbreviation
    ("barcelona assembly siet", "barcelona assembly site", None),  # transposition
    ("pratt whitney inc", "pratt whitney", None),                 # legal suffix
    ("boeing 737", "boeing 737 800", None),                       # under-specified, not a conflict
])
def test_token_conflict(raw, candidate, expected):
    assert token_conflict(raw, candidate) == expected


def test_a_one_letter_substitution_is_not_a_typo():
    assert token_conflict("halvorsen mail services", "halvorsen rail services") == "token_substitution"


def test_extra_identifier_is_detected_either_way():
    assert token_extra_identifier("boeing 737", "boeing 737 800")
    assert token_extra_identifier("boeing 737 800", "boeing 737")
    assert not token_extra_identifier("boeing 737 800", "boeing 737 800")


def test_a_different_part_number_is_a_new_entity():
    registry = _registry(("Inverter Module IM-450", "PRODUCT"))
    assert registry.resolve("Inverter Module IM-470", entity_type="PRODUCT") is None


def test_ratio_only_behaviour_returns_when_guards_are_off():
    registry = _registry(("Inverter Module IM-450", "PRODUCT"), guards=False)
    assert registry.resolve("Inverter Module IM-470") == ("Inverter Module IM-450", "PRODUCT")


def test_an_extra_identifier_above_the_threshold_only_reaches_review():
    registry = _registry(("Boeing 737-800 Freighter", "PRODUCT"))
    result = registry.resolve("Boeing 737 Freighter", entity_type="PRODUCT")
    assert isinstance(result, AmbiguousMatch)
    assert result.candidate == ("Boeing 737-800 Freighter", "PRODUCT")


def test_typos_plurals_and_abbreviations_still_merge():
    registry = _registry(("Halvorsen Grid Services", "ORG"), ("Coolant Pump Assembly", "PRODUCT"))
    assert registry.resolve("Halvorsen Grid Servces", entity_type="ORG") == ("Halvorsen Grid Services", "ORG")
    assert registry.resolve("Coolant Pump Assy", entity_type="PRODUCT") == ("Coolant Pump Assembly", "PRODUCT")


def test_fuzzy_is_scoped_to_the_mention_type_but_exact_is_not():
    registry = _registry(("Volta Drive Unit", "PRODUCT"))
    assert registry.resolve("Volta Drive Units", entity_type="ORG") is None
    assert registry.resolve("Volta Drive Units", entity_type="PRODUCT") == ("Volta Drive Unit", "PRODUCT")
    assert registry.resolve("Volta Drive Unit", entity_type="ORG") == ("Volta Drive Unit", "PRODUCT")
    assert registry.resolve("Volta Drive Units") == ("Volta Drive Unit", "PRODUCT")  # no type, unscoped


def test_an_uncurated_acronym_goes_to_review_never_to_a_merge():
    registry = _registry(("Failure Mode and Effects Analysis", "CONCEPT"))
    result = registry.resolve("FMEA", entity_type="CONCEPT")
    assert isinstance(result, AmbiguousMatch)
    assert (result.match_type, result.candidate[0]) == ("acronym", "Failure Mode and Effects Analysis")
    assert registry.resolve("FMEA", entity_type="ORG") is None


def test_a_head_token_mention_goes_to_review_with_every_candidate():
    registry = _registry(("Northwind Electrical Components", "ORG"), ("Northwind Logistics", "ORG"))
    result = registry.resolve("Northwind", entity_type="ORG")
    assert isinstance(result, AmbiguousMatch) and result.match_type == "containment"
    names = {result.candidate[0], *(r[0] for r in result.runner_ups)}
    assert names == {"Northwind Electrical Components", "Northwind Logistics"}


def test_review_routing_is_off_with_the_guards():
    registry = _registry(("Northwind Logistics", "ORG"), guards=False)
    assert registry.resolve("Northwind") is None


def test_curated_aliases_apply_only_to_known_entities_of_this_tenant():
    registry = _registry(("International Automotive Task Force", "ORG"), ("Other Body", "ORG"))
    registry._exact[_normalize("OB")] = ("Other Body", "ORG")
    applied, skipped = registry.apply_curated_aliases([
        {"alias": "IATF", "canonical": "International Automotive Task Force", "type": "ORG"},
        {"alias": "VDA", "canonical": "Verband der Automobilindustrie", "type": "ORG"},       # unknown entity
        {"alias": "XYZ", "canonical": "Other Body", "type": "ORG", "tenant": "t2"},           # other tenant
        {"alias": "OB", "canonical": "International Automotive Task Force", "type": "ORG"},   # already elsewhere
    ])
    assert (applied, skipped) == (1, 3)
    assert registry.resolve("IATF") == ("International Automotive Task Force", "ORG")
    assert registry.resolve("OB") == ("Other Body", "ORG")
    assert registry.resolve("VDA") is None


def test_load_curated_aliases_reads_yaml_and_rejects_incomplete_entries(tmp_path):
    good = tmp_path / "aliases.yml"
    good.write_text("aliases:\n  - {alias: IATF, canonical: International Automotive Task Force, type: ORG}\n",
                    encoding="utf-8")
    assert load_curated_aliases(good)[0]["alias"] == "IATF"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"alias": "IATF", "type": "ORG"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="entry 0"):
        load_curated_aliases(bad)


@pytest.mark.parametrize("case_set", ["original", "heldout"])
def test_recorded_sets_have_no_over_merge_or_wrong_link(case_set):
    sys.path.insert(0, str(ROOT))
    from scripts.eval_entity_resolution import SETS, run

    report = run(SETS[case_set][0])
    assert report["errors"]["over_merge"] == 0
    assert report["errors"]["wrong_link"] == 0
    assert run(SETS[case_set][0], guards=False)["errors"]["over_merge"] > 0
