"""Unit tests for the degree-anomaly auto-quarantine fix.

Before this fix, `auto_quarantine_anomalies` always passed entity_type=
"UNKNOWN" to `quarantine_entity`, whose MATCH is keyed on (name, type,
tenant) -- since no real entity has type "UNKNOWN", the MATCH silently
matched nothing, yet the caller still incremented its success count and
logged as if the entity had been quarantined. The degree-anomaly detector
also never selected `e.type` in the first place, so the real type wasn't
even available to pass through.

See docs/archive/audits/audit-2026-09-23.md, "Not fixed" #5.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


class TestDegreeAnomalyDetectorReturnsEntityType:
    @pytest.mark.asyncio
    async def test_check_degree_anomalies_selects_and_returns_entity_type(self):
        from graphrag.graph.ingestion_validator import IngestionValidator

        validator = IngestionValidator.__new__(IngestionValidator)
        validator._neo4j = AsyncMock()
        validator._neo4j.run = AsyncMock(return_value=[
            {"entity": "Boeing 737", "entity_type": "Aircraft", "degree": 500, "mean_degree": 10.0},
        ])

        issues = await validator._check_degree_anomalies(tenant="aerospace", doc_id=None)

        cypher = validator._neo4j.run.call_args[0][0]
        assert "e.type AS entity_type" in cypher
        assert len(issues) == 1
        assert issues[0]["entity_type"] == "Aircraft"
        assert issues[0]["entity"] == "Boeing 737"


class TestAutoQuarantineUsesRealEntityType:
    @pytest.mark.asyncio
    async def test_quarantines_using_the_issue_entity_type_not_unknown(self):
        from graphrag.graph.quarantine import QuarantineService

        svc = QuarantineService(neo4j_client=AsyncMock())
        svc.quarantine_entity = AsyncMock()

        report = {"issues": [
            {"type": "degree_anomaly", "entity": "Boeing 737", "entity_type": "Aircraft", "degree": 500},
        ]}
        count = await svc.auto_quarantine_anomalies(doc_id="doc-1", validation_report=report, tenant="aerospace")

        assert count == 1
        svc.quarantine_entity.assert_awaited_once_with(
            entity_name="Boeing 737",
            entity_type="Aircraft",
            reason="degree_anomaly:degree=500",
            flagged_by="ingestion_validator",
            tenant="aerospace",
        )

    @pytest.mark.asyncio
    async def test_missing_entity_type_is_not_counted_as_a_success(self):
        """The regression this fix targets: a hardcoded/missing type must not
        be silently reported as a successful quarantine."""
        from graphrag.graph.quarantine import QuarantineService

        svc = QuarantineService(neo4j_client=AsyncMock())
        svc.quarantine_entity = AsyncMock()

        report = {"issues": [
            {"type": "degree_anomaly", "entity": "Boeing 737", "degree": 500},  # no entity_type
        ]}
        count = await svc.auto_quarantine_anomalies(doc_id="doc-1", validation_report=report, tenant="aerospace")

        assert count == 0
        svc.quarantine_entity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_degree_anomaly_issues_are_ignored(self):
        from graphrag.graph.quarantine import QuarantineService

        svc = QuarantineService(neo4j_client=AsyncMock())
        svc.quarantine_entity = AsyncMock()

        report = {"issues": [{"type": "orphan_entity", "entity": "X", "entity_type": "CONCEPT"}]}
        count = await svc.auto_quarantine_anomalies(doc_id="doc-1", validation_report=report, tenant="aerospace")

        assert count == 0
        svc.quarantine_entity.assert_not_awaited()
