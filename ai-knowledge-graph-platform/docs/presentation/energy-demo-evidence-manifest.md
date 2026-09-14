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
| 3. Graph query | GraphDB-style SPARQL interrogation returns source-linked evidence | Captured trace `run_demo`, `query_rows`, `answer_source` | Same E2E command; live GraphDB acceptance is separate |
| 4. Safety | Invalid records are rejected before publication | `ontology/shapes/energy-asset-intelligence.shapes.ttl`, E2E `invalid_batch` | Same E2E command |
| 5. Explanation | The answer is produced by governed SPARQL and carries evidence | `evals/energy_demo/sparql/maintenance_review.rq`, E2E `maintenance_review` | Same E2E command |
| 6. Time and abstention | Revision history is queryable and missing evidence produces no conclusion | E2E `historical_state`, `insufficient_evidence` | Same E2E command |
| 7. Access | Tenant scope is enforced | E2E `wrong_tenant`, `tests/unit/test_energy_dev_login.py` | Same E2E command plus unit tests |
| 8. GraphRAG | Neo4j is an optional rebuildable operational read model | E2E `neo4j_projection`, `graphrag/domains/energy/lpg_projection.py` | Same E2E command; `--live-neo4j` only with configured Neo4j |

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
- `energy_asset_intelligence_implementation_demo.mp4` — illustrated implementation path

The client teaser now includes a real GraphDB Workbench/SPARQL interrogation
scene captured from the local `energy-demo` repository. The query returns one
source-linked WT-01 result with its gearbox and WO-9001. Capture it again with
`python scripts/capture_energy_graphdb_sparql.py` when the local GraphDB data
changes. The repository's live GraphDB test remains the technical acceptance
evidence.
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

On 2026-09-13 all three movies were rebuilt after a review of all 24 static
scenes and their narration scripts. Corrections include mapping counts (13
entities / 3 relationships, now derived from the captured trace), explicit
read-plus-write scope for mutations, readable revision history, a labelled
close-up of the actual technical UI capture, and scorecard results rather
than empty metric headings. The teaser no longer implies demonstrated scale.
Every scene displays **Synthetic data · Advisory POC · No equipment control**.

`energy_movie_qa.json` records hashes, final MP4 stream metadata, every scene's
duration and narration margin, frame comparison results, expected narration,
and local speech-recognition transcripts of the final MP4 audio. The verifier
decodes each complete movie and compares a midpoint frame from every static
scene with a fresh in-memory render. All three files have 1280×720 H.264 video
and AAC audio. The reviewed ASR transcripts preserve the key counts, corrected
temperature, thresholds, and claim boundaries; technical acronym spellings
are imperfect and are not treated as word-perfect transcription.

**QA boundary:** this is full automated media coverage plus review of every
static scene and ASR transcript, not a human start-to-finish listening session.
Naturalness, acronym pronunciation, and speaker-device playback still need
human sign-off before external release. There are no word-synchronised
subtitles; on-screen slide text is a summary, not a verbatim caption track.
See [movie QA and rebuild instructions](energy-movie-qa.md).

## Evidence remediation scene (2026-09-14) — not yet captured

`render_energy_demo_client_teaser.py` now defines a sixth scene,
"Requesting the missing evidence," referencing a new
`dashboard_evidence_remediation.png` capture that
`scripts/capture_energy_demo_ui.py` was extended to produce. Neither the
screenshot nor a rebuilt teaser exists yet: two independent local-environment
blockers (a standalone headless-Chromium process unable to open a TCP
connection to the local API even though plain TCP connectivity to the same
port succeeded, and the interactive verification browser pane being hidden)
stopped the capture in this session. The underlying feature was verified live
through a real authenticated browser session regardless (see
`tasks/todo.md`'s "Evidence remediation workflow" entry). Do not treat the
current `energy_asset_intelligence_client_teaser.mp4` as showing this scene
until `scripts/capture_energy_demo_ui.py` and this renderer have actually been
rerun.
