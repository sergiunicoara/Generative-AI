"""config/ontologies/sales.yml replaces the leftover adtech (WPP/Nova Beverages)
template — must actually validate against the legacy domain_ontology.py schema
it's written for, and must supply §12's required ADDRESSES_OBJECTION mapping.
"""

from __future__ import annotations

from pathlib import Path

from src.graph.domain_ontology import get_relation_rules, load_domain_ontology, validate_ontology_yaml

_SALES_YML = Path(__file__).resolve().parents[3] / "config" / "ontologies" / "sales.yml"


def test_sales_yaml_file_exists_and_is_not_the_adtech_template():
    assert _SALES_YML.exists()
    text = _SALES_YML.read_text(encoding="utf-8")
    assert "WPP" not in text
    assert "Nova Beverages" not in text
    assert "ADVERTISER" not in text


def test_sales_ontology_validates_cleanly():
    report = validate_ontology_yaml(_SALES_YML)
    assert report["valid"] is True
    assert report["errors"] == []


def test_addresses_objection_mapping_is_present_and_curated():
    ontology = load_domain_ontology(_SALES_YML)
    rules = get_relation_rules(ontology)
    assert "ADDRESSES_OBJECTION" in rules
    assert "OBJECTION" in [t.upper() for t in rules["ADDRESSES_OBJECTION"]["target"]]
    assert "CONTENT_ASSET" in [d.upper() for d in rules["ADDRESSES_OBJECTION"]["domain"]]
