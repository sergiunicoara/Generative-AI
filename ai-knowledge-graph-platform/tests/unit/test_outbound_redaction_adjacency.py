"""Outbound redaction must not lose a selected PII class to an unselected one.

``outbound._redact`` scans with ``min_confidence=0.0`` and filters to the
tenant's classes *after* ``PIIGuard._deoverlap`` has picked one finding per span.
The credit-card pattern (a run of 13-19 digits with optional spaces or hyphens)
is not a default outbound class, but it starts earlier than a phone number or SSN
that follows a date or any other digit run, so it won the span, was then filtered
out, and took the phone number or SSN with it: the value went to the external
provider unredacted while the audit recorded no redaction.
"""
from __future__ import annotations

import pytest

from graphrag.core.outbound import DEFAULT_CLASSES, _redact
from graphrag.graph.pii_guard import PIIGuard

PHONE = "555-123-4567"
SSN = "123-45-6789"
PREFIXES = ["2026-11-06 ", "Case 4821 ", "ticket 123456 ", "2026 11 06 ", "4111-1111-1111-1111 "]


@pytest.mark.parametrize("prefix", PREFIXES)
@pytest.mark.parametrize("value, pii_class", [(PHONE, "PHONE"), (SSN, "SSN")])
def test_value_after_a_date_or_number_is_still_redacted(prefix, value, pii_class):
    out, counts = _redact(f"{prefix}{value} about the invoice", DEFAULT_CLASSES)
    assert value not in out
    assert counts.get(pii_class) == 1


@pytest.mark.parametrize("text, value", [
    ("Alice, 2026 11 06 " + SSN, SSN),
    ("Called on 2026-11-06 " + PHONE + " twice", PHONE),
    ("12345 " + PHONE + ", " + SSN, PHONE),
    ("12345 " + PHONE + ", " + SSN, SSN),
])
def test_neighbouring_values_are_all_redacted(text, value):
    out, _ = _redact(text, DEFAULT_CLASSES)
    assert value not in out


def test_unselected_classes_are_still_left_alone():
    """The class filter is respected: with only SSN selected a phone number stays."""
    out, counts = _redact(f"call {PHONE} ssn {SSN}", ("SSN",))
    assert PHONE in out and SSN not in out and counts == {"SSN": 1}


def test_a_selected_card_number_is_still_redacted():
    out, counts = _redact("card 4111 1111 1111 1111 ok", ("CC",))
    assert "4111" not in out and counts == {"CC": 1}


def test_guard_default_behaviour_is_unchanged():
    """PIIGuard's own scan keeps its 0.80 threshold: a card-like run is not reported."""
    assert PIIGuard(None).redact("2026-11-06 " + PHONE) == "2026-11-06 [PHONE_REDACTED]"
