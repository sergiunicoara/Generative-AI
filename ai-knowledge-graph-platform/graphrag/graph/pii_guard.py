"""PII guard — detection, tagging, and redaction of personally identifiable information.

Problem solved
--------------
LLM extractors emit PII from source documents directly into the graph:
  - PERSON entity names — acceptable (the entity IS a person)
  - Embedded SSN, passport numbers, phone numbers in entity descriptions — not acceptable
  - Email addresses in chunk text — borderline (public vs. private)
  - Direct identifiers in relation attributes — accidental

Without a PII layer, the graph becomes a PII store without explicit consent
or data minimisation, creating GDPR/CCPA liability.

Architecture
------------
PIIGuard uses two complementary detection strategies:

1. Regex detection — pattern-matched PII classes:
   - SSN: \d{3}-\d{2}-\d{4}
   - Passport: [A-Z]{1,2}\d{7,9}
   - Phone: various formats
   - Email: RFC 5321 subset
   - Credit card: Luhn-checkable 13-19 digit numbers
   - IP address: IPv4 / IPv6

2. Entity-type detection — entity.type = PERSON → entity is PII by default.
   The description field may contain additional PII (address, DOB, etc.).

Operations:
  scan_chunk(text) → list of PII findings with position and class
  redact(text) → text with PII replaced by [CLASS_REDACTED]
  tag_entity_pii(name, type, tenant) → add pii_sensitive=true flag to entity
  scan_document(doc_id, tenant) → full PII scan report for a document

Integration:
  - Called from GraphWriter.write_chunks() to scan chunk text before ingestion.
  - Called from GraphWriter.write_entities() to tag PERSON entities.
  - Not a blocking filter — PII findings are logged as warnings; callers decide
    whether to redact or allow.  Use `auto_redact=true` in settings to redact
    all findings automatically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import structlog

log = structlog.get_logger(__name__)


# ── PII patterns ───────────────────────────────────────────────────────────────

@dataclass
class PIIFinding:
    pii_class:  str     # SSN | PASSPORT | PHONE | EMAIL | CC | IP | PERSON_NAME
    start:      int     # character offset in text
    end:        int
    value:      str     # the matched string (for logging — redact before storing)
    confidence: float   # 0.0–1.0


_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("SSN",      re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                             0.99),
    ("PASSPORT", re.compile(r"\b[A-Z]{1,2}\d{7,9}\b"),                             0.85),
    ("EMAIL",    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0.98),
    ("PHONE",    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"), 0.90),
    ("PHONE_INTL", re.compile(r"\+\d{1,3}[\s-]\d[\d\s\-]{6,14}\d"),               0.85),
    ("IP_V4",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),                       0.80),
    ("IP_V6",    re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"),    0.90),
    # Credit card: 13–19 contiguous digits (not preceded by another digit)
    ("CC",       re.compile(r"(?<!\d)(\d[ -]?){13,19}(?!\d)"),                     0.75),
    # National ID / driver's license fallback — alphanumeric 8-12 chars
    ("NATIONAL_ID", re.compile(r"\b[A-Z]{2}\d{6,10}\b"),                           0.60),
]

# Entity types considered PII by nature
PII_ENTITY_TYPES = {"PERSON"}


class PIIGuard:
    """
    Detect and redact PII in chunk text and entity descriptions.

    Usage::

        guard = PIIGuard(neo4j_client)

        # Scan a piece of text
        findings = guard.scan_text("Contact John at 555-123-4567 or ssn 123-45-6789")
        # → [PIIFinding(pii_class="PHONE", ...), PIIFinding(pii_class="SSN", ...)]

        # Redact PII in text
        clean = guard.redact("SSN: 123-45-6789")
        # → "SSN: [SSN_REDACTED]"

        # Tag an entity as PII-sensitive in Neo4j
        await guard.tag_entity_pii("Alice Smith", "PERSON", tenant="acme")

        # Scan a whole document
        report = await guard.scan_document("doc_abc", tenant="acme")
    """

    def __init__(self, neo4j_client, min_confidence: float = 0.80):
        self._neo4j = neo4j_client
        self._min_confidence = min_confidence

    # ── Text scanning ──────────────────────────────────────────────────────────

    def scan_text(self, text: str) -> list[PIIFinding]:
        """
        Return all PII findings above the confidence threshold.

        Results are de-overlapped: if two patterns match the same span,
        the higher-confidence one wins.
        """
        findings: list[PIIFinding] = []
        for pii_class, pattern, confidence in _PATTERNS:
            if confidence < self._min_confidence:
                continue
            for match in pattern.finditer(text):
                findings.append(PIIFinding(
                    pii_class=pii_class,
                    start=match.start(),
                    end=match.end(),
                    value=match.group(),
                    confidence=confidence,
                ))
        return self._deoverlap(findings)

    def redact(self, text: str) -> str:
        """Replace all PII patterns with [CLASS_REDACTED] placeholders."""
        findings = sorted(self.scan_text(text), key=lambda f: f.start, reverse=True)
        result = text
        for f in findings:
            result = result[:f.start] + f"[{f.pii_class}_REDACTED]" + result[f.end:]
        return result

    def has_pii(self, text: str) -> bool:
        """Return True if any PII is detected in text."""
        return bool(self.scan_text(text))

    # ── Entity tagging ─────────────────────────────────────────────────────────

    async def tag_entity_pii(
        self,
        entity_name: str,
        entity_type: str,
        tenant: str = "default",
        reason: str = "",
    ) -> None:
        """
        Mark an entity as PII-sensitive in Neo4j.

        Tagged entities:
        - Are flagged in retrieval to omit full details in public responses.
        - Are prioritized for erasure in GDPR deletion requests.
        - Are listed in the PII inventory.
        """
        is_pii_type = entity_type in PII_ENTITY_TYPES
        desc_rows = await self._neo4j.run(
            "MATCH (e:Entity {name: $name, type: $type, tenant: $tenant}) "
            "RETURN e.description AS desc",
            name=entity_name,
            type=entity_type,
            tenant=tenant,
        )
        desc_pii = False
        if desc_rows:
            desc = desc_rows[0].get("desc") or ""
            desc_pii = self.has_pii(desc)

        await self._neo4j.run(
            """
            MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})
            SET e.pii_sensitive    = true,
                e.pii_reason       = $reason,
                e.pii_tagged_at    = datetime()
            """,
            name=entity_name,
            type=entity_type,
            tenant=tenant,
            reason=reason or (
                "entity_type_pii" if is_pii_type else "description_pii"
            ),
        )
        log.info(
            "pii_guard.entity_tagged",
            entity=entity_name,
            type=entity_type,
            tenant=tenant,
        )

    async def auto_tag_persons(self, tenant: str = "default") -> int:
        """
        Tag all PERSON entities in a tenant as PII-sensitive.
        Called post-ingestion when auto_pii_tagging is enabled in config.
        Returns count of entities tagged.
        """
        rows = await self._neo4j.run(
            """
            MATCH (e:Entity {type: 'PERSON', tenant: $tenant})
            WHERE NOT e.pii_sensitive = true
            SET e.pii_sensitive = true,
                e.pii_reason    = 'entity_type_pii',
                e.pii_tagged_at = datetime()
            RETURN count(e) AS n
            """,
            tenant=tenant,
        )
        count = rows[0].get("n", 0) if rows else 0
        log.info("pii_guard.auto_tag_persons", count=count, tenant=tenant)
        return count

    # ── Document scan ──────────────────────────────────────────────────────────

    async def scan_document(
        self,
        doc_id: str,
        tenant: str = "default",
    ) -> dict:
        """
        Scan all chunks of a document for PII and return a structured report.

        Does NOT redact or modify data — purely diagnostic.
        """
        rows = await self._neo4j.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, tenant: $tenant})
            RETURN c.id AS chunk_id, c.text AS text
            """,
            doc_id=doc_id,
            tenant=tenant,
        )

        total_findings = 0
        chunk_reports: list[dict] = []
        by_class: dict[str, int] = {}

        for row in rows:
            text = row.get("text") or ""
            findings = self.scan_text(text)
            if findings:
                total_findings += len(findings)
                chunk_reports.append({
                    "chunk_id":    row["chunk_id"],
                    "finding_count": len(findings),
                    "findings":    [
                        {
                            "pii_class":  f.pii_class,
                            "confidence": f.confidence,
                            "offset":     f.start,
                            # Value is masked in the report — only class and position
                            "value_masked": f"{'*' * min(len(f.value), 8)}",
                        }
                        for f in findings
                    ],
                })
                for f in findings:
                    by_class[f.pii_class] = by_class.get(f.pii_class, 0) + 1

        report = {
            "doc_id":          doc_id,
            "tenant":          tenant,
            "chunks_scanned":  len(rows),
            "total_findings":  total_findings,
            "by_class":        by_class,
            "chunks_with_pii": chunk_reports,
            "action_required": total_findings > 0,
        }
        if total_findings > 0:
            log.warning(
                "pii_guard.pii_found",
                doc_id=doc_id,
                total=total_findings,
                by_class=by_class,
            )
        return report

    # ── PII inventory ──────────────────────────────────────────────────────────

    async def list_pii_entities(
        self,
        tenant: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return all entities tagged as PII-sensitive."""
        return await self._neo4j.run(
            """
            MATCH (e:Entity {pii_sensitive: true})
            WHERE ($tenant = 'default' OR e.tenant = $tenant)
            RETURN e.name         AS name,
                   e.type         AS type,
                   e.pii_reason   AS reason,
                   e.pii_tagged_at AS tagged_at,
                   e.tenant        AS tenant
            ORDER BY e.pii_tagged_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _deoverlap(findings: list[PIIFinding]) -> list[PIIFinding]:
        """Remove overlapping findings, keeping the highest-confidence one."""
        if not findings:
            return []
        sorted_f = sorted(findings, key=lambda f: (f.start, -f.confidence))
        result: list[PIIFinding] = []
        last_end = -1
        for f in sorted_f:
            if f.start >= last_end:
                result.append(f)
                last_end = f.end
        return result
