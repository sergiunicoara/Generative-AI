# Provenance-backed intelligence ingestion

The ingestion path treats a source assertion as different from an answer-time
claim. `ClaimEvidenceGraph` remains the provenance graph for evaluated answers;
this flow writes `IntelligenceArtifact` nodes only for source-grounded content.

## Pipeline

`Document → Chunk → Entity/Relation → explicit aliases → IntelligenceArtifact /
TimePeriod / StructuredTable → IngestionRunManifest`

- `CLAIM`, `OBSERVATION`, `EVENT`, and `FINDING` are extracted only when the
  model returns an exact `evidence_quote` contained in the source chunk.
- `FINDING` is retained only when the document itself explicitly makes the
  conclusion. `Insight` and `Recommendation` are not generated during
  ingestion; they need an explicit reviewed synthesis policy.
- Every artifact links to its chunk and document (`DERIVED_FROM`,
  `ASSERTED_IN`) and to matching extracted entities (`ASSERTS_ABOUT`).
- Re-ingestion removes the document's old artifacts and structured tables
  before its new evidence is written, so stale source assertions cannot remain
  retrievable.

## Alias mining

The existing exact/fuzzy/embedding resolver is retained. The added miner only
accepts an alias where the document explicitly asserts it: `X (Y)` or an
`also known as`-style phrase, and only when both names were extracted in that
chunk. It stores the source document, chunk, exact quote, evidence kind, and
confidence on the `Alias` node. It does not collapse two freshly written nodes
in the same batch; the normal review/correction path remains responsible for
irreversible merges.

## Tables and time

Native Excel worksheets are represented as `StructuredTable` nodes with
columns, rows JSON, JSON-LD, source document, and extraction method. Other
layout-aware extractors can provide the same `Document.metadata["structured_tables"]`
contract. PDF text alone is not guessed into cells.

Explicit dates, month/year, quarter/year, and year references create
tenant-scoped `TimePeriod` nodes and `IN_PERIOD` links. Query expansion adds
only missing canonical parent periods (for example `January 2024 → 2024-Q1`),
without changing governed valid-time or transaction-time filtering.

## Ingestion receipt

Each normal ingestion writes an `IngestionRunManifest` with corpus identity,
correlation ID, configured model/prompt versions, stage durations/counts,
completion state, error state, and integrity hash. Provider usage continues to
be captured by GenAI telemetry. When a provider does not report usage, the
manifest records that cost as unknown rather than fabricating a dollar amount.

`intelligence_artifacts_enabled`, `temporal_hierarchy_enabled`, and
`temporal_query_expansion_enabled` in `config/settings.yml` control the new
stages. Dual raw/summary chunks remain benchmark-gated and are intentionally
not enabled by this feature.
