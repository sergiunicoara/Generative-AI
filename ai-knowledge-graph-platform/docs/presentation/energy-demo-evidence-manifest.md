# Energy Asset Intelligence — Demo Evidence Manifest

This manifest is the source-of-truth checklist for the client-facing demo and
technical walkthrough. It ties each visible claim to a repeatable command or
checked-in artifact. The data is synthetic and local; no customer system or
physical equipment is connected.

## Recommended presentation order

| Scene | Audience takeaway | Evidence | Reproduce with |
| --- | --- | --- | --- |
| 1. Decision | WT-01 needs advisory review | `artifacts/energy-demo-e2e-report.json`, `maintenance_review` | `python scripts/run_energy_demo_e2e.py --output artifacts/energy-demo-e2e-report.json` |
| 2. Sources | SAP-, Snowflake- and SharePoint-shaped records become RDF | `ontology/mappings/energy-assets.r2rml.ttl`, `ontology/mappings/energy-observations.rml.ttl`, `artifacts/energy-demo-e2e.ttl` | Same E2E command; source fixture is created deterministically |
| 3. Safety | Invalid records are rejected before publication | `ontology/shapes/energy-asset-intelligence.shapes.ttl`, E2E `invalid_batch` | Same E2E command |
| 4. Explanation | The answer is produced by governed SPARQL and carries evidence | `evals/energy_demo/sparql/maintenance_review.rq`, E2E `maintenance_review` | Same E2E command |
| 5. Time and abstention | Revision history is queryable and missing evidence produces no conclusion | E2E `historical_state`, `insufficient_evidence` | Same E2E command |
| 6. Access | Tenant scope is enforced | E2E `wrong_tenant`, `tests/unit/test_energy_dev_login.py` | Same E2E command plus unit tests |
| 7. GraphRAG | Neo4j is an optional rebuildable operational read model | E2E `neo4j_projection`, `graphrag/domains/energy/lpg_projection.py` | Same E2E command; `--live-neo4j` only with configured Neo4j |

## Captured run

The checked-in capture at `docs/presentation/energy_demo_real_run.json` records
the commands, return codes, stdout, and final capability list from the real
local workflow. The current captured run produced 154 published RDF triples,
30 candidate records, 0 quarantined records, and a dry-run Neo4j projection
whose 30-node, 28-relationship topology passed the projection acceptance
checks. These are fixture-level implementation observations, not
enterprise-scale performance claims.

The rendered assets are:

- `energy_asset_intelligence_real_run_demo.mp4` — implementation walkthrough
- `energy_asset_intelligence_client_teaser.mp4` — short stakeholder teaser
- `energy_architecture_preview.png` — architecture visual

The current local live-acceptance record is
`artifacts/energy-live-neo4j-acceptance.xml`: two tests passed with no errors,
failures, or skips. They ran the real projection CLI against Neo4j 5.20,
then read the graph back through a new driver to prove directed traversal,
tenant isolation, and retry-safe refresh.

## Claim boundaries to keep on screen

Use the caption **Synthetic data · Advisory POC · No equipment control**.

- “SAP/Snowflake/SharePoint” means deterministic source-shaped fixtures, not
  live customer connectors.
- GraphDB and Neo4j are optional local serving/read-model paths.
- RDF is the semantic evidence source of truth; Neo4j is one-way and rebuildable.
- The maintenance output is an advisory recommendation, not an automated
  work-order approval or equipment command.
- Local latency, throughput, and record counts must not be presented as
  production SLAs or capacity benchmarks.

## Final pre-send check

```powershell
python scripts/run_energy_demo_e2e.py --output artifacts/energy-demo-e2e-report.json
python -m pytest -q tests/unit/test_energy_demo_e2e.py tests/unit/test_energy_lpg_projection.py
```

If the report changes, regenerate or relabel the movie capture so the visible
numbers and provenance remain aligned with the run being presented.

## Render QA record

On 2026-09-13, the teaser, implementation walkthrough, and real-run
walkthrough were regenerated after the current command trace and UI screenshots.
Representative frames were inspected for readable text and clipped elements;
two layout defects found in that inspection were corrected before the final
render. Each movie's video/audio stream shape was verified with `ffprobe`:
1280×720 H.264 video and an AAC narration stream, on all three files.
Scene duration is calculated from its generated narration, so the captions and
voiceover use the same scene boundaries. The client-facing caption remains
**Synthetic data · Advisory POC · No equipment control**.

**QA scope, stated plainly:** the checks above are codec/resolution
verification (`ffprobe`) and spot-checked representative frames -- not a
full watch-through. Narration audio content, word-for-word caption accuracy,
and scene-to-scene timing across the entire runtime have **not** been
exhaustively verified. Treat full narration/caption correctness as open
until someone watches each video start to finish.
