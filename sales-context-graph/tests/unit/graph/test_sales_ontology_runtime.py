from pathlib import Path

import pytest

from src.graph.sales_ontology import (
    InvalidRelationEndpoint,
    UnknownClaimPredicate,
    UnknownGraphRelation,
    allowed_claim_predicates,
    validate_claim_predicate,
    validate_relation,
)

_SALES_YML = Path(__file__).resolve().parents[3] / "config" / "ontologies" / "sales.yml"


def test_sales_yaml_is_not_the_leftover_adtech_template():
    """config/ontologies/sales.yml once replaced a WPP/Nova Beverages adtech
    template left over from the fork this repo started from. The parametrized
    relation_rules tests below would already fail against that template (it
    has neither RAISED_OBJECTION nor ADDRESSES_OBJECTION), but this guard
    names the specific regression directly rather than leaving it implicit."""
    text = _SALES_YML.read_text(encoding="utf-8")
    assert "WPP" not in text
    assert "Nova Beverages" not in text
    assert "ADVERTISER" not in text


def test_sales_ontology_contains_all_production_extractor_predicates():
    assert {"RAISED_OBJECTION", "HAS_BLOCKER", "HAS_ACTION_ITEM", "MENTIONS_ORG"} <= set(
        allowed_claim_predicates()
    )


def test_claim_predicate_is_normalized_and_validated():
    assert validate_claim_predicate(" raised_objection ") == "RAISED_OBJECTION"


def test_unknown_claim_predicate_is_rejected():
    with pytest.raises(UnknownClaimPredicate):
        validate_claim_predicate("IGNORE_SAFETY")


# --- relation_rules: the graph-edge vocabulary, all 5 documented entries ---

@pytest.mark.parametrize(
    "relation_type,domain,target",
    [
        ("HAS_ASSIGNMENT", "OPPORTUNITY", "STAKEHOLDER_ASSIGNMENT"),  # real edge, stakeholder_repository.py
        ("ASSIGNS", "STAKEHOLDER_ASSIGNMENT", "CONTACT"),  # real edge, stakeholder_repository.py
        ("RAISED_OBJECTION", "CONTACT", "OBJECTION"),  # not yet a materialized edge anywhere
        ("RAISED_OBJECTION", "LEAD", "OBJECTION"),  # domain has 2 allowed types; both must pass
        ("ADDRESSES_OBJECTION", "CONTENT_ASSET", "OBJECTION"),  # not yet a materialized edge anywhere
        ("CONVERTED_TO", "LEAD", "ACCOUNT"),  # not yet implemented (§5); validator is ready regardless
        ("MERGED_INTO", "ACCOUNT", "ACCOUNT"),  # not yet implemented (§5)
    ],
)
def test_all_five_relation_rules_accept_their_documented_endpoints(relation_type, domain, target):
    assert validate_relation(relation_type, domain, target) == relation_type


def test_type_hierarchy_ancestry_resolves_subtype_against_supertype_rule():
    # No relation_rules entry currently lists a supertype (ORG/PERSON/CONCEPT/
    # ASSET) in its domain/target — every rule uses concrete leaf types
    # (CONTACT, LEAD, CONTENT_ASSET, ...), so this exercises the private
    # ancestry helpers directly rather than a real rule: forward-looking
    # coverage for whenever a rule does list a supertype (e.g. "domain: [ORG]"
    # to mean "any org-shaped entity"), not dead code today. ACCOUNT is a
    # documented subtype of ORG (type_hierarchy: [ACCOUNT, ORG]).
    from src.graph.sales_ontology import _satisfies

    assert _satisfies("ACCOUNT", frozenset({"ORG"})) is True
    assert _satisfies("ACCOUNT", frozenset({"PERSON"})) is False


def test_unknown_relation_type_is_rejected():
    with pytest.raises(UnknownGraphRelation):
        validate_relation("DELETES_EVERYTHING", "ACCOUNT", "ACCOUNT")


def test_wrong_domain_label_is_rejected():
    with pytest.raises(InvalidRelationEndpoint):
        validate_relation("ASSIGNS", "OPPORTUNITY", "CONTACT")  # ASSIGNS' domain is StakeholderAssignment, not Opportunity


def test_wrong_target_label_is_rejected():
    with pytest.raises(InvalidRelationEndpoint):
        validate_relation("HAS_ASSIGNMENT", "OPPORTUNITY", "CONTACT")  # HAS_ASSIGNMENT's target is StakeholderAssignment
