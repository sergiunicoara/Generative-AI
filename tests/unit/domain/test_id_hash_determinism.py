"""§6 stable source identities: same inputs -> same ID across repeated calls (the
"second identical ingest changes zero graph counts" property, §15, starts here at
the pure-function level), and adjacent-field-boundary changes must not collide —
proving the \\x1f-separator join in src/domain/identity.py actually prevents the
naive-concatenation collision it's designed to prevent.
"""

from src.domain.identity import (
    assertion_id,
    conversation_id,
    crm_entity_id,
    extraction_run_id,
    mention_id,
    segment_id,
)


def test_crm_entity_id_is_deterministic():
    a = crm_entity_id("ws-1", "salesforce", "Account", "001xx")
    b = crm_entity_id("ws-1", "salesforce", "Account", "001xx")
    assert a == b
    assert len(a) == 64  # full sha256 hex digest, not truncated


def test_crm_entity_id_changes_with_any_single_field():
    base = crm_entity_id("ws-1", "salesforce", "Account", "001xx")
    assert crm_entity_id("ws-2", "salesforce", "Account", "001xx") != base
    assert crm_entity_id("ws-1", "gong", "Account", "001xx") != base
    assert crm_entity_id("ws-1", "salesforce", "Contact", "001xx") != base
    assert crm_entity_id("ws-1", "salesforce", "Account", "002xx") != base


def test_crm_entity_id_does_not_collide_across_field_boundaries():
    """Naive '+'-concatenation would make ("ab","c",...) collide with ("a","bc",...).
    The \\x1f-separated join must not."""
    left = crm_entity_id("ab", "c", "Account", "001")
    right = crm_entity_id("a", "bc", "Account", "001")
    assert left != right


def test_conversation_id_is_deterministic_and_scoped_by_workspace():
    a = conversation_id("ws-1", "gong", "call-abc")
    b = conversation_id("ws-1", "gong", "call-abc")
    c = conversation_id("ws-2", "gong", "call-abc")
    assert a == b
    assert a != c


def test_segment_id_is_deterministic_and_scoped_by_index():
    conv = conversation_id("ws-1", "gong", "call-abc")
    a = segment_id(conv, 0)
    b = segment_id(conv, 0)
    c = segment_id(conv, 1)
    assert a == b
    assert a != c


def test_mention_id_is_deterministic_and_scoped_by_span():
    seg = segment_id(conversation_id("ws-1", "gong", "call-abc"), 0)
    a = mention_id(seg, 10, 20, "volkswagen", "ORG")
    b = mention_id(seg, 10, 20, "volkswagen", "ORG")
    c = mention_id(seg, 10, 21, "volkswagen", "ORG")
    assert a == b
    assert a != c


def test_assertion_id_is_deterministic():
    seg = segment_id(conversation_id("ws-1", "gong", "call-abc"), 0)
    a = assertion_id("ws-1", seg, 10, 40, "elena-popescu", "RAISED_OBJECTION", "pricing", "AFFIRMED")
    b = assertion_id("ws-1", seg, 10, 40, "elena-popescu", "RAISED_OBJECTION", "pricing", "AFFIRMED")
    assert a == b


def test_extraction_run_id_is_deterministic_and_changes_with_extractor_version():
    a = extraction_run_id("fixture", "n/a", "v1", "extractor-1.0.0", "nonce-1")
    b = extraction_run_id("fixture", "n/a", "v1", "extractor-1.0.0", "nonce-1")
    c = extraction_run_id("fixture", "n/a", "v1", "extractor-2.0.0", "nonce-1")
    assert a == b
    assert a != c
