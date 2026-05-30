"""Confidence calibration — Brier score and calibration curve for KG edge confidence.

Problems solved
---------------
1. Overconfident extractors — LLMs systematically assign high confidence
   (0.85–0.99) to extracted relations regardless of actual accuracy.  A model
   that reports 0.9 confidence and is correct only 60% of the time is
   dangerously miscalibrated.

2. No feedback loop — the Bayesian confidence fusion (merge_relation) produces
   well-combined values given accurate inputs, but if the prior confidences
   themselves are biased the merged value is also biased.

3. Silent drift — extraction model updates may recalibrate confidence in ways
   that are invisible until RAGAS drops.

What is calibration?
--------------------
A model is perfectly calibrated if, among all predictions made with confidence p,
exactly p fraction are actually correct.  Calibration curve = (mean predicted, mean actual)
binned at 0.1 intervals.  Perfect calibration = identity line.

Brier score = mean((predicted - actual)²) over N verification samples.
Range [0, 1]; lower is better.  A uniform 0.5 baseline achieves 0.25.

Architecture
------------
- CalibrationService accumulates (predicted, actual) samples linked to doc IDs
  and prompt versions so you can track per-model calibration separately.
- Samples are persisted as CalibrationSample nodes in Neo4j.
- CalibrationService.apply_calibration() maps a raw confidence through the
  current calibration table (isotonic regression approximation via sorted bins).
- CalibrationSnapshot nodes record Brier score history for trend tracking.
- GraphEvaluator can include brier_score as a 7th health metric.

Golden set integration
----------------------
The simplest integration: when a human verifies a relation as correct or
incorrect (via the corrections API), that creates a CalibrationSample with
actual = 1.0 or 0.0.  Over time the calibration curve converges to reality.
"""

from __future__ import annotations

import math
from uuid import uuid4

import structlog

log = structlog.get_logger(__name__)

# Number of bins for the calibration curve
CALIBRATION_BINS = 10


class CalibrationService:
    """
    Accumulate calibration data and compute Brier-score metrics.

    Usage::

        svc = CalibrationService(neo4j_client)

        # Record a verification result
        await svc.add_sample(
            predicted_confidence=0.87,
            actual_outcome=1.0,          # 1.0 = correct, 0.0 = incorrect
            relation="CEO_OF",
            source_doc_id="doc_abc",
            prompt_version="v2",
            tenant="acme",
        )

        # Compute metrics
        score = await svc.brier_score(tenant="acme")
        curve = await svc.calibration_curve(tenant="acme")

        # Apply correction to a raw confidence value
        calibrated = await svc.apply_calibration(0.87, tenant="acme")

        # Persist a named snapshot for trend tracking
        snap_id = await svc.persist_snapshot(tenant="acme", label="post-v2-model")
    """

    def __init__(self, neo4j_client):
        self._neo4j = neo4j_client

    # ── Data collection ────────────────────────────────────────────────────────

    async def add_sample(
        self,
        predicted_confidence: float,
        actual_outcome: float,          # 1.0 = correct, 0.0 = incorrect
        relation: str = "",
        source_doc_id: str = "",
        prompt_version: str = "",
        tenant: str = "default",
        verified_by: str = "system",
    ) -> str:
        """
        Persist a single calibration sample.

        ``actual_outcome`` should be 1.0 if the relation was verified correct
        by a human or automated golden-set check, 0.0 otherwise.  Fractional
        values (e.g. 0.5 for "partially correct") are also accepted.

        Returns the sample node ID.
        """
        predicted = max(0.0, min(1.0, float(predicted_confidence)))
        actual    = max(0.0, min(1.0, float(actual_outcome)))

        sample_id = str(uuid4())
        await self._neo4j.run(
            """
            CREATE (s:CalibrationSample {
                id:                  $id,
                predicted_confidence: $predicted,
                actual_outcome:       $actual,
                error_sq:             $error_sq,
                relation:             $relation,
                source_doc_id:        $source_doc_id,
                prompt_version:       $prompt_version,
                tenant:               $tenant,
                verified_by:          $verified_by,
                recorded_at:          datetime()
            })
            """,
            id=sample_id,
            predicted=predicted,
            actual=actual,
            error_sq=(predicted - actual) ** 2,
            relation=relation,
            source_doc_id=source_doc_id,
            prompt_version=prompt_version,
            tenant=tenant,
            verified_by=verified_by,
        )
        return sample_id

    async def add_batch(
        self,
        samples: list[dict],
        tenant: str = "default",
    ) -> list[str]:
        """
        Convenience method for bulk loading calibration samples.

        Each dict must have keys: predicted_confidence, actual_outcome.
        Optional keys: relation, source_doc_id, prompt_version, verified_by.
        """
        ids = []
        for s in samples:
            sid = await self.add_sample(
                predicted_confidence=s["predicted_confidence"],
                actual_outcome=s["actual_outcome"],
                relation=s.get("relation", ""),
                source_doc_id=s.get("source_doc_id", ""),
                prompt_version=s.get("prompt_version", ""),
                tenant=tenant,
                verified_by=s.get("verified_by", "batch"),
            )
            ids.append(sid)
        return ids

    # ── Metrics ────────────────────────────────────────────────────────────────

    async def brier_score(
        self,
        tenant: str = "default",
        prompt_version: str | None = None,
        relation: str | None = None,
    ) -> float:
        """
        Compute Brier score = mean((predicted - actual)²) over stored samples.

        Lower is better (0.0 = perfect, 0.25 = uninformative baseline).
        Returns -1.0 if no samples are available.

        Optionally filter by prompt_version or relation to track per-model or
        per-relation calibration separately.
        """
        version_filter  = "AND s.prompt_version = $prompt_version" if prompt_version else ""
        relation_filter = "AND s.relation = $relation" if relation else ""
        params: dict = {"tenant": tenant}
        if prompt_version:
            params["prompt_version"] = prompt_version
        if relation:
            params["relation"] = relation

        rows = await self._neo4j.run(
            f"""
            MATCH (s:CalibrationSample)
            WHERE ($tenant = 'default' OR s.tenant = $tenant)
              {version_filter}
              {relation_filter}
            RETURN avg(s.error_sq) AS brier, count(s) AS n
            """,
            **params,
        )
        if not rows or not rows[0].get("n"):
            return -1.0
        brier = rows[0].get("brier")
        return round(float(brier), 6) if brier is not None else -1.0

    async def calibration_curve(
        self,
        tenant: str = "default",
        bins: int = CALIBRATION_BINS,
    ) -> list[dict]:
        """
        Compute the calibration curve as binned (mean_predicted, mean_actual, n) tuples.

        A perfectly calibrated model produces points on the identity line.
        Points above the line = under-confident; below = over-confident.

        Returns list of dicts sorted by bin midpoint:
            [{"bin_start": 0.0, "bin_end": 0.1, "mean_predicted": ..., "mean_actual": ..., "n": ...}, ...]
        """
        bin_width = 1.0 / bins
        rows = await self._neo4j.run(
            """
            MATCH (s:CalibrationSample)
            WHERE ($tenant = 'default' OR s.tenant = $tenant)
            RETURN s.predicted_confidence AS predicted,
                   s.actual_outcome       AS actual
            """,
            tenant=tenant,
        )

        # Bin the samples in Python — Neo4j doesn't have histogram aggregation
        buckets: dict[int, list[tuple[float, float]]] = {i: [] for i in range(bins)}
        for row in rows:
            p = float(row.get("predicted") or 0.0)
            a = float(row.get("actual")    or 0.0)
            bin_idx = min(int(p / bin_width), bins - 1)
            buckets[bin_idx].append((p, a))

        curve = []
        for i in range(bins):
            samples = buckets[i]
            bin_start = round(i * bin_width, 2)
            bin_end   = round((i + 1) * bin_width, 2)
            if samples:
                mean_predicted = sum(s[0] for s in samples) / len(samples)
                mean_actual    = sum(s[1] for s in samples) / len(samples)
                n              = len(samples)
            else:
                mean_predicted = bin_start + bin_width / 2
                mean_actual    = 0.0
                n              = 0
            curve.append({
                "bin_start":      bin_start,
                "bin_end":        bin_end,
                "mean_predicted": round(mean_predicted, 4),
                "mean_actual":    round(mean_actual, 4),
                "n":              n,
                "calibration_gap": round(mean_predicted - mean_actual, 4) if n else 0.0,
            })
        return curve

    async def calibration_summary(
        self,
        tenant: str = "default",
    ) -> dict:
        """
        High-level calibration health report:
        - brier_score: overall Brier score
        - sample_count: number of verified samples
        - max_calibration_gap: worst bin deviation from the ideal diagonal
        - over_confident_bins: bins where model confidence > actual rate
        - under_confident_bins: bins where model confidence < actual rate
        """
        brier = await self.brier_score(tenant)
        curve = await self.calibration_curve(tenant)

        sample_count      = sum(b["n"] for b in curve)
        gaps              = [b["calibration_gap"] for b in curve if b["n"] > 0]
        max_gap           = max(abs(g) for g in gaps) if gaps else 0.0
        over_confident    = sum(1 for g in gaps if g > 0.05)
        under_confident   = sum(1 for g in gaps if g < -0.05)

        return {
            "brier_score":          brier,
            "sample_count":         sample_count,
            "max_calibration_gap":  round(max_gap, 4),
            "over_confident_bins":  over_confident,
            "under_confident_bins": under_confident,
            "calibration_curve":    curve,
            "verdict": (
                "well-calibrated"      if brier < 0.10
                else "acceptable"      if brier < 0.20
                else "over-confident"  if over_confident > under_confident
                else "under-confident" if under_confident > over_confident
                else "needs-review"
            ),
        }

    # ── Confidence correction ──────────────────────────────────────────────────

    async def apply_calibration(
        self,
        raw_confidence: float,
        tenant: str = "default",
    ) -> float:
        """
        Map a raw extractor confidence through the empirical calibration curve.

        Uses isotonic regression approximation: find the bin containing the
        raw confidence and substitute mean_actual as the calibrated value.
        If no samples exist for that bin, returns the raw confidence unchanged.

        This is a lightweight post-hoc calibration that does not require
        retraining the extractor.
        """
        curve = await self.calibration_curve(tenant)
        for bin_data in curve:
            if bin_data["bin_start"] <= raw_confidence < bin_data["bin_end"]:
                if bin_data["n"] > 0:
                    return round(bin_data["mean_actual"], 4)
                break
        return round(float(raw_confidence), 4)

    # ── Snapshot ───────────────────────────────────────────────────────────────

    async def persist_snapshot(
        self,
        tenant: str = "default",
        label: str = "",
    ) -> str:
        """
        Persist a CalibrationSnapshot node capturing current metrics.

        Used for trend tracking — run after model updates or major ingestion
        batches to create a before/after record.
        """
        summary = await self.calibration_summary(tenant)
        snap_id = str(uuid4())

        await self._neo4j.run(
            """
            CREATE (cs:CalibrationSnapshot {
                id:                  $id,
                tenant:              $tenant,
                label:               $label,
                brier_score:         $brier,
                sample_count:        $n,
                max_calibration_gap: $gap,
                over_confident_bins: $over,
                under_confident_bins: $under,
                verdict:             $verdict,
                recorded_at:         datetime()
            })
            """,
            id=snap_id,
            tenant=tenant,
            label=label,
            brier=summary["brier_score"],
            n=summary["sample_count"],
            gap=summary["max_calibration_gap"],
            over=summary["over_confident_bins"],
            under=summary["under_confident_bins"],
            verdict=summary["verdict"],
        )
        log.info(
            "calibration.snapshot_persisted",
            snap_id=snap_id,
            brier=summary["brier_score"],
            n=summary["sample_count"],
            verdict=summary["verdict"],
            tenant=tenant,
        )
        return snap_id

    async def get_trend(
        self,
        tenant: str = "default",
        limit: int = 10,
    ) -> list[dict]:
        """Return recent CalibrationSnapshot records for trend analysis."""
        return await self._neo4j.run(
            """
            MATCH (cs:CalibrationSnapshot)
            WHERE ($tenant = 'default' OR cs.tenant = $tenant)
            RETURN cs.label               AS label,
                   cs.brier_score         AS brier_score,
                   cs.sample_count        AS sample_count,
                   cs.max_calibration_gap AS max_calibration_gap,
                   cs.verdict             AS verdict,
                   cs.recorded_at         AS recorded_at
            ORDER BY cs.recorded_at DESC
            LIMIT $limit
            """,
            tenant=tenant,
            limit=limit,
        )
