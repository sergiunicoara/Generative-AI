# ADR 0003 — Bayesian Confidence Accumulation over Last-write-wins

| Field | Value |
|---|---|
| **Status** | Accepted |
| **Date** | 2024-Q2 |
| **Deciders** | Platform architect, data quality lead |

---

## Context

When the same relation is asserted by multiple source documents (e.g. both a press
release and a regulatory filing state that "Company A acquired Company B"), the graph
must decide how to represent the resulting confidence in that fact.

Three approaches are common:

**Last-write-wins:** The most recently ingested document's confidence overwrites
previous values. Simple, deterministic, but loses information from prior sources.

**Average:** Average confidence across all asserting documents. Easy to compute
but statistically incorrect — two independent observations of the same event
provide more evidence than one, so confidence should *increase* with corroboration.

**Bayesian accumulation (independent evidence fusion):**
```
new_conf = 1 − (1 − c₁)(1 − c₂)
```
Correct for independent evidence sources: confidence grows with corroboration,
bounded at 1.0.

---

## Decision

**Use Bayesian accumulation.** Implemented in Neo4j Cypher inside `merge_relation()`:

```cypher
SET r.confidence = CASE
    WHEN r.confidence IS NULL THEN $confidence
    ELSE 1.0 − (1.0 − r.confidence) * (1.0 − $confidence)
END
```

---

## Reasons

### 1. Corroborating evidence should increase confidence

If three independent documents each assert "A `CEO_OF` B" with confidence 0.8, the
merged confidence should be higher than a single document's 0.8:

```
1 − (0.2)(0.2)(0.2) = 0.992
```

Last-write-wins would discard the first two observations entirely. Averaging would
yield 0.8 (same as one source) — ignoring the corroborative signal.

### 2. Mathematically correct for independent sources

The formula `1 − ∏(1 − cᵢ)` is the standard result for the probability that at
least one of N independent events occurs. Applied to knowledge extraction: the
probability that the relation is true, given N independent document-level assertions.

**Independence assumption:** Document sources within the same corpus may not be
truly independent (e.g. a press release and a news article both derived from the same
corporate announcement). The independence assumption may overestimate confidence in
such cases. This is accepted as a pragmatic approximation — the calibration service
(`confidence_calibration.py`) measures and corrects systematic overconfidence via
isotonic regression.

### 3. Provenance is preserved alongside confidence

The `source_doc_ids` list on every edge accumulates all asserting document IDs,
regardless of confidence merging:

```cypher
SET r.source_doc_ids = CASE
    WHEN $source_doc_id IN r.source_doc_ids THEN r.source_doc_ids
    ELSE r.source_doc_ids + [$source_doc_id]
END
```

This enables contradiction detection (identifying which documents disagree) and
authority chain analysis (which document is the current authority) independently
of the merged confidence value.

### 4. Confidence degrades correctly for inferred and superseded facts

The accumulation formula composes with:

- **Inference decay:** inferred edge confidence = product of premise confidences × decay^depth
  `c_inferred = c₁ × c₂ × decay^hops`
- **Supersession penalty:** edges from superseded documents multiplied by `superseded_penalty`
  (default 0.5) before GNN scoring
- **Temporal decay:** `c_effective = c_stored × exp(−ln2 / half_life × age_days)`

These penalties stack correctly because they operate on the stored confidence before
it reaches the GNN adjacency matrix.

---

## Consequences

**Positive:**
- Confidence increases with independent corroboration (correct Bayesian behaviour)
- Naturally bounded at [0, 1] — no clamping required after merge
- Composable with decay mechanisms

**Negative:**
- Independence assumption may not hold for corpus-level co-occurrence
- Historical confidence values are not queryable — only the current merged value is
  stored on the edge (provenance is preserved via source_doc_ids, not per-source conf)
- Overconfidence from correlated sources is possible without calibration

**Mitigation:** The `CalibrationService` measures Brier score and builds isotonic
correction curves. `apply_calibration(raw_conf)` maps stored values through the
empirical curve before use in contexts requiring calibrated probabilities (e.g.
threshold-based alerts, regulatory reports).
