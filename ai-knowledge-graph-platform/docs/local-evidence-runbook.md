# Local Evidence Runbook

This runbook creates reproducible **local** evidence for the platform. None
of its outputs establishes production availability, customer adoption, or
business impact without a separately documented deployment study.

## 1. Authenticated remote MCP

> **Token audience.** Since [ADR 0010](adr/0010-audience-bound-access-tokens.md)
> the gateway accepts only tokens whose `aud` is `GRAPHRAG_MCP_RESOURCE`.
> Request one with `resource=<GRAPHRAG_MCP_RESOURCE>` on `POST /auth/token`,
> or pass `--dev-token` to the evidence scripts below, which mint an
> MCP-audience token for you. A REST API token will 401.

Start the Docker MCP gateway:

```powershell
docker compose -f compose.dev.yaml up -d mcp
$env:GRAPHRAG_MCP_TOKEN = "<scoped JWT>"
python scripts/run_remote_mcp_smoke.py --output artifacts/remote-mcp-smoke.json
```

The smoke run performs an MCP `initialize` and `tools/list` exchange using the
Streamable HTTP endpoint. Keep the generated report with the commit SHA and
environment details; it demonstrates one authenticated local run only.

## 2. Seeded MCP graph-fact load evidence

Seed the isolated demo tenant, then mint a short-lived development token with
`read` and `tenant:local-evidence` scopes. The token must be generated from
your local development configuration and must never be committed.

```powershell
python scripts/seed_demo_data.py --commit --wipe --tenant local-evidence
$env:GRAPHRAG_MCP_TOKEN = "<local scoped JWT>"
python scripts/run_mcp_operation_load.py --token $env:GRAPHRAG_MCP_TOKEN --tenant local-evidence --matrix 100:5,1000:25 --output artifacts/mcp-graph-fact-load-matrix.json
```

This calls `query_graph_facts` over authenticated Streamable HTTP MCP for every
request and records success count, error rate, throughput, and p50/p95/p99 for
each request/concurrency scenario. It
includes a fresh MCP session initialization in each measured request, so it is a
reproducible local service measurement rather than a production capacity claim.

Measure the optimized warm-session path separately:

```powershell
python scripts/run_mcp_warm_session_benchmark.py --dev-token --tenant local-evidence --requests 1000 --concurrency 25 --output artifacts/mcp-warm-session-benchmark.json
```

This initializes one authenticated MCP session per worker and measures only
subsequent tool calls. Keep cold and warm reports side by side; the difference
is transport/session overhead, not an end-to-end application improvement.

## 3. Governed write evidence

```powershell
python scripts/run_governed_write_evidence.py --output artifacts/governed-write-evidence.json
```

The script drives real local MCP, API, and Neo4j paths against the isolated
tenant: read, approval-required write, human approval, execute, idempotent
replay, stale-version refusal, dry-run, and approval-gated compensation. Its
JSON receipt is evidence of that one synthetic local execution only.

## 4. Retrieval baseline comparison

```powershell
python scripts/run_graph_fact_golden_eval.py --token $env:GRAPHRAG_MCP_TOKEN --tenant local-evidence --repetitions 3 --output artifacts/graph-fact-golden-eval.json
```

The fixed three-case graph-fact set is compared with an explicit empty-corpus
baseline. It validates tenant-scoped graph retrieval, not open-ended RAG answer
quality or customer accuracy.

## 5. Controlled-query model cost

```powershell
python scripts/measure_controlled_query_cost.py --load-report artifacts/mcp-graph-fact-load.json --output artifacts/controlled-query-cost.json
```

`query_graph_facts` is deterministic and uses no model. This report therefore
records zero model tokens and model cost for that path; it does not imply zero
infrastructure cost or a customer saving.

## 6. Multi-tenant HTTP load evidence

Create a JSON array of requests with at least two tenant values, then run:

```powershell
python scripts/run_production_exercises.py load artifacts/load-cases.json --concurrency 20 > artifacts/load-report.json
```

The report includes request count, passed/failed count, error rate, elapsed
time, throughput, and p50/p95/p99 latency. Run it against Docker Compose, not
mock tests, before citing the numbers. `tests/load/` proves concurrency shape;
it is not a throughput benchmark.

## 7. Generic retrieval baseline comparison

Run the same versioned golden set and environment for both profiles. Extract
the selected numeric metrics into two JSON objects, then compare:

```powershell
python scripts/compare_retrieval_baseline.py artifacts/baseline.json artifacts/candidate.json --metrics faithfulness context_recall --output artifacts/retrieval-comparison.json
```

Only compare like-for-like corpus revision, tenant, prompt/model route, and
judge configuration. A difference is a local experiment result, not a customer
accuracy claim.

## 8. Manual versus agent-assisted investigation study

Use `data/evidence/investigation-tasks.json` and copy
`data/evidence/investigation-study-template.csv`. Have the same operator
solve matched, pre-defined investigation tasks manually and with the platform.
Randomise the condition order, do not reuse answers between conditions, and
start timing when each prompt is revealed. Stop timing only when the written
answer and cited evidence are complete. Record elapsed seconds, evidence score
(using the task rubric), and success in the copied CSV.

```powershell
python scripts/analyze_investigation_study.py data/evidence/investigation-study.csv --output artifacts/investigation-study.json
```

Use the report’s stated sample and rubric. The supplied local CSV is a format
example, not a measured study; do not generalize it to customer time savings.

## 9. Workflow, cost, recovery, and security evidence

```powershell
python scripts/run_engineering_workflow.py workflows/example.yaml --run-id evidence-demo
python scripts/summarize_workflow_evidence.py artifacts/workflow-runs.json artifacts/cost-events.json --output artifacts/workflow-evidence.json
python scripts/run_production_exercises.py security artifacts/security-cases.json
python scripts/run_production_exercises.py artifact-integrity artifacts/backup.dump artifacts/restored.dump > artifacts/artifact-integrity.json
```

Use `scripts/export_operational_evidence.py` to combine an authenticated
Prometheus scrape with explicitly measured deployment metadata. Leave every
unmeasured field as `null`.

## 10. Failure matrix and Kubernetes validation

```powershell
python scripts/run_local_failure_exercises.py --output artifacts/local-failure-exercises.json
kubectl kustomize deploy/kubernetes > artifacts/kubernetes-rendered.yaml
kubectl apply --dry-run=client -k deploy/kubernetes
```

The failure matrix records the local controls for duplicate writes, stale
versions, tenant boundaries, approval bypass, compensation replay, and backup
integrity. The Kubernetes commands validate rendered manifests and admission
shape. See `docs/gcp-production-deployment.md` for rollout and rollback steps;
neither exercise is a production availability or incident-prevention claim.

The command above compares two files only. It is an artifact-integrity check,
not evidence that a database was restored. Use the Docker-backed recovery
tests below for that proof.

With Docker available, exercise a real RDF and property-graph recovery:

```powershell
python -m pytest -q tests/e2e/test_live_graphdb.py tests/e2e/test_live_neo4j_backup_restore.py
```

The GraphDB test exports the Energy RDF dataset using a SPARQL `CONSTRUCT`,
loads it into a fresh repository, and reruns the committed
`maintenance_review.rq` query. The Neo4j test runs `scripts/kg_backup.py`,
wipes the test tenant, restores its NDJSON backup, and reruns Cypher against
the restored data. These tests prove dataset recovery, not a vendor-native
binary restore or repository-configuration recovery.

With Docker Compose running, exercise a real dependency restart:

```powershell
python scripts/run_docker_failure_exercise.py --service redis --output artifacts/docker-redis-failure-exercise.json
python scripts/run_docker_failure_exercise.py --service neo4j --output artifacts/docker-neo4j-failure-exercise.json
```

The command stops only the selected local dependency and always attempts to
start it again. It records container recovery, not application-level incident
prevention.

## 11. Release-evidence report

Stitches commit info, a dependency/vulnerability scan, and whichever of the
live-test, benchmark, and recovery artifacts above are present into one
report — the "keep the generated report with the commit SHA and environment
details" instruction from section 1 above, automated instead of manual:

```powershell
python scripts/build_release_evidence_report.py --output artifacts/release-evidence.json --markdown artifacts/release-evidence.md
```

Every section is independently optional: a source artifact this run didn't
produce shows as `available: false` with a stated reason, not a crash or a
silently missing key. Run it last, after whichever sections above you
generated for this evidence run.

## Public artifacts

- `docs/templates/public-evaluation-report-template.md`
- `docs/public-local-evaluation-report.md` — generated from the checked-in local run outputs
- `docs/articles/governed-mcp-and-agent-writes.md`
- `docs/articles/local-evidence-walkthrough.md`
- `docs/presentation/governed-mcp-walkthrough-video-script.md`
- `docs/presentation/local-evidence-walkthrough.mp4` — silent, locally rendered walkthrough
- the capability-contract section in `docs/mcp-operations.md`
- `artifacts/mcp-capabilities-v1.json` — exported versioned capability contract
- `artifacts/graphrag-ontologies-v1.zip` — ontology package with manifest and checksums

Regenerate the reproducible package and report after a new evidence run:

```powershell
python scripts/export_mcp_contract.py --output artifacts/mcp-capabilities-v1.json
python scripts/export_ontology_package.py --output artifacts/graphrag-ontologies-v1.zip
python scripts/build_public_local_evaluation_report.py --output docs/public-local-evaluation-report.md
python scripts/render_local_evidence_walkthrough.py --output docs/presentation/local-evidence-walkthrough.mp4
```

The repository can generate and publish these artifacts; conference acceptance,
open-source adoption, domain-expert review, and customer outcomes require
external participation.
