"""Unit tests for graphrag.graph.pii_guard — PIIGuard text scanning and redaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag.graph.pii_guard import PIIGuard, PIIFinding


# ── helpers ────────────────────────────────────────────────────────────────────

def _guard(min_confidence: float = 0.80) -> PIIGuard:
    neo4j = MagicMock()
    neo4j.run = AsyncMock(return_value=[])
    return PIIGuard(neo4j, min_confidence=min_confidence)


# ── scan_text ──────────────────────────────────────────────────────────────────

def test_scan_text_detects_ssn():
    g = _guard()
    findings = g.scan_text("His SSN is 123-45-6789 on file.")
    ssn = [f for f in findings if f.pii_class == "SSN"]
    assert len(ssn) == 1
    assert ssn[0].value == "123-45-6789"
    assert ssn[0].confidence >= 0.99


def test_scan_text_detects_email():
    g = _guard()
    findings = g.scan_text("Contact alice@example.com for details.")
    emails = [f for f in findings if f.pii_class == "EMAIL"]
    assert len(emails) == 1
    assert "alice@example.com" in emails[0].value


def test_scan_text_detects_phone():
    g = _guard()
    findings = g.scan_text("Call us at 555-123-4567 anytime.")
    phones = [f for f in findings if f.pii_class == "PHONE"]
    assert len(phones) >= 1


def test_scan_text_detects_ipv4():
    g = _guard()
    findings = g.scan_text("Server at 192.168.1.100 is down.")
    ips = [f for f in findings if f.pii_class == "IP_V4"]
    assert len(ips) == 1
    assert ips[0].value == "192.168.1.100"


def test_scan_text_clean_text_returns_empty():
    g = _guard()
    findings = g.scan_text("The company was founded in 2002.")
    assert findings == []


def test_scan_text_multiple_pii_classes():
    g = _guard()
    text = "Email john@test.org or call 800-555-0199 or send SSN 987-65-4321."
    findings = g.scan_text(text)
    classes = {f.pii_class for f in findings}
    assert "EMAIL" in classes
    assert "PHONE" in classes
    assert "SSN" in classes


def test_scan_text_respects_min_confidence():
    """With high min_confidence, low-confidence patterns are suppressed.

    NATIONAL_ID pattern ([A-Z]{2}\\d{6,10}, conf=0.60) matches exactly 6
    digits, which is outside PASSPORT ([A-Z]{1,2}\\d{7,9}, conf=0.85) range.
    So AB123456 is purely NATIONAL_ID territory.
    """
    g_strict = PIIGuard(MagicMock(), min_confidence=0.80)   # 0.80 > 0.60 → blocks NATIONAL_ID
    g_loose  = PIIGuard(MagicMock(), min_confidence=0.60)   # 0.60 <= 0.60 → allows NATIONAL_ID
    text = "National ID: AB123456"   # 6 digits → only NATIONAL_ID, not PASSPORT
    strict_classes = {f.pii_class for f in g_strict.scan_text(text)}
    loose_classes  = {f.pii_class for f in g_loose.scan_text(text)}
    assert "NATIONAL_ID" not in strict_classes
    assert "NATIONAL_ID" in loose_classes


# ── redact ─────────────────────────────────────────────────────────────────────

def test_redact_replaces_ssn():
    g = _guard()
    result = g.redact("SSN: 123-45-6789 on record.")
    assert "123-45-6789" not in result
    assert "[SSN_REDACTED]" in result


def test_redact_replaces_email():
    g = _guard()
    result = g.redact("Send to user@domain.org please.")
    assert "user@domain.org" not in result
    assert "[EMAIL_REDACTED]" in result


def test_redact_clean_text_unchanged():
    g = _guard()
    text = "The quarterly report shows 15 % growth."
    assert g.redact(text) == text


def test_redact_multiple_pii_all_replaced():
    g = _guard()
    text = "SSN: 111-22-3333, email: x@y.com."
    result = g.redact(text)
    assert "111-22-3333" not in result
    assert "x@y.com" not in result
    assert "REDACTED" in result


# ── has_pii ───────────────────────────────────────────────────────────────────

def test_has_pii_true_with_ssn():
    assert _guard().has_pii("SSN 999-88-7777") is True


def test_has_pii_false_for_clean_text():
    assert _guard().has_pii("Revenue was $1.2M last quarter.") is False


# ── finding structure ─────────────────────────────────────────────────────────

def test_finding_has_correct_span():
    g = _guard()
    text = "SSN: 123-45-6789"
    findings = g.scan_text(text)
    ssn = next(f for f in findings if f.pii_class == "SSN")
    assert text[ssn.start:ssn.end] == ssn.value


# ── tag_entity_pii (async, uses neo4j mock) ────────────────────────────────────

async def test_tag_entity_pii_calls_neo4j():
    neo4j = MagicMock()
    neo4j.run = AsyncMock(return_value=[])
    g = PIIGuard(neo4j)
    await g.tag_entity_pii("Alice Smith", "PERSON", tenant="acme")
    assert neo4j.run.called
    call_str = str(neo4j.run.call_args)
    assert "Alice Smith" in call_str or "acme" in call_str
