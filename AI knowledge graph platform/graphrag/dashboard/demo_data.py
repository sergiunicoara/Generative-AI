"""Representative sample payloads for dashboard demo mode.

These are used **only** when ``GRAPHRAG_DASHBOARD_DEMO`` is set AND the live API
is unreachable, so that the dashboard can be shown populated (gauges, charts,
tables) without a running Neo4j / ingestion pipeline. In normal operation the
flag is unset and the tabs render real data or a real error panel — production
behaviour is never altered.

The numbers mirror the healthy thresholds documented in
docs/performance-metrics-inventory.md so the demo is realistic, not arbitrary.
"""

from __future__ import annotations

# ── Graph Health ────────────────────────────────────────────────────────────

HEALTH_SNAPSHOTS = {
    "snapshots": [
        {
            "entity_count": 24850, "edge_count": 89432,
            "alias_coverage": 0.92, "high_conf_rate": 0.81,
            "contradiction_rate": 0.9, "orphan_rate": 0.08,
            "community_coherence": 0.67, "recorded_at": "2026-05-31T21:00:00Z",
        },
        {
            "entity_count": 24180, "edge_count": 86790,
            "alias_coverage": 0.90, "high_conf_rate": 0.80,
            "contradiction_rate": 1.4, "orphan_rate": 0.09,
            "community_coherence": 0.64, "recorded_at": "2026-05-30T21:00:00Z",
        },
        {
            "entity_count": 23410, "edge_count": 83120,
            "alias_coverage": 0.89, "high_conf_rate": 0.78,
            "contradiction_rate": 1.1, "orphan_rate": 0.11,
            "community_coherence": 0.62, "recorded_at": "2026-05-29T21:00:00Z",
        },
        {
            "entity_count": 22600, "edge_count": 79050,
            "alias_coverage": 0.88, "high_conf_rate": 0.77,
            "contradiction_rate": 2.0, "orphan_rate": 0.12,
            "community_coherence": 0.60, "recorded_at": "2026-05-28T21:00:00Z",
        },
    ]
}

HEALTH_ALERTS = {
    "alerts": [
        {"metric": "orphan_rate", "value": 0.12, "threshold": 0.10,
         "direction": "above", "tenant": "aerospace", "fired_at": "2026-05-28T21:00:00Z"},
    ]
}

# ── Calibration ─────────────────────────────────────────────────────────────

CALIBRATION_SNAPSHOTS = {
    "snapshots": [
        {
            "brier_score": 0.18, "model_version": "llama-3.3-70b",
            "recorded_at": "2026-05-31T21:00:00Z",
            "calibration_bins": [
                {"predicted": 0.1, "actual": 0.08}, {"predicted": 0.3, "actual": 0.32},
                {"predicted": 0.5, "actual": 0.49}, {"predicted": 0.7, "actual": 0.71},
                {"predicted": 0.9, "actual": 0.88},
            ],
        },
        {"brier_score": 0.21, "model_version": "llama-3.3-70b", "recorded_at": "2026-05-30T21:00:00Z",
         "calibration_bins": []},
        {"brier_score": 0.24, "model_version": "llama-3.3-70b", "recorded_at": "2026-05-29T21:00:00Z",
         "calibration_bins": []},
        {"brier_score": 0.27, "model_version": "llama-3.3-70b", "recorded_at": "2026-05-28T21:00:00Z",
         "calibration_bins": []},
    ]
}

# ── Conflicts ───────────────────────────────────────────────────────────────

CONFLICTS = {
    "conflicts": [
        {"conflict_id": "c-8f21a", "type": "exclusive_state", "src": "Aircraft G-ABCD",
         "tgt": "IS_AIRWORTHY / IS_UNAIRWORTHY", "relation": "state", "tenant": "aerospace"},
        {"conflict_id": "c-3d77c", "type": "directional_reversal", "src": "FAA-AD-2024-01-02",
         "tgt": "FAA-AD-2022-03-07", "relation": "SUPERSEDES", "tenant": "aerospace"},
        {"conflict_id": "c-19b04", "type": "functional_violation", "src": "Boeing 737-800",
         "tgt": "Type Certificate A16WE", "relation": "CERTIFIED_BY", "tenant": "aerospace"},
    ]
}

# ── Communities ─────────────────────────────────────────────────────────────

COMMUNITY_SUMMARY = {"change_fraction": 0.14, "changed_entities": 312}

COMMUNITY_HISTORY = {
    "history": [
        {"snapshot_id": "s-2051", "entity_count": 24850, "edge_count": 89432,
         "recorded_at": "2026-05-31T21:00:00Z", "is_rebuild": True},
        {"snapshot_id": "s-2050", "entity_count": 24180, "edge_count": 86790,
         "recorded_at": "2026-05-30T21:00:00Z", "is_rebuild": False},
        {"snapshot_id": "s-2049", "entity_count": 23410, "edge_count": 83120,
         "recorded_at": "2026-05-29T21:00:00Z", "is_rebuild": False},
    ]
}

# ── GDPR ────────────────────────────────────────────────────────────────────

GDPR_AUDIT = {
    "log": [
        {"audit_id": "a-7741", "action": "forget_entity", "entity_name": "J. Doe",
         "entity_type": "PERSON", "tenant": "aerospace", "requested_by": "dpo@client",
         "executed_at": "2026-05-31T14:22:00Z"},
        {"audit_id": "a-7732", "action": "pii_redaction", "entity_name": "Contact Record 4821",
         "entity_type": "PII", "tenant": "aerospace", "requested_by": "system",
         "executed_at": "2026-05-30T09:10:00Z"},
    ]
}
