# Documentation Guide

The specification-to-implementation runner is documented in
`engineering-workflows.md` and demonstrated by `../workflows/example.yaml`.

## Canonical technical docs

- `roadmap.md` — implementation status and remaining work
- `archive/audits/` — dated audit snapshots and follow-up evidence
  conformance, cache and rate-limit resource safety, and what was checked and
  found clean
- `knowledge-graph-architecture.md` — system architecture and data flow
- `graphrag-terminology.md` — shared vocabulary and algorithms
- `graphrag-tutorial.md` — setup and end-to-end usage
- `runbook.md` — operations and troubleshooting
- `mcp-operations.md` — authenticated remote MCP deployment and incident response
- `local-evidence-runbook.md` — reproducible local MCP, retrieval, write, cost, and load evidence
- `public-local-evaluation-report.md` — bounded results from the checked-in synthetic local run
- `entity-resolution.md`, `ontology-model.md`, `ontology-governance.md`,
  `cypher-patterns.md` — focused KG references
- `enterprise-content-governance.md` — provider-neutral ACL, SharePoint sync,
  explicit document-link topology, and late-target reconciliation
- `intelligence-ingestion.md` — provenance and source-authority model for
  ingested intelligence
- `performance-metrics-inventory.md` — metric definitions and verification queries
- `slo.md` — service-level indicators and objectives
- `gcp-production-deployment.md` — Terraform/GKE production deployment path
- `CONTRIBUTING.md` — branch, review, and eval-gate process
- `../monitoring/prometheus/alerts.yml` — alerting rules, with the action an
  operator should take in each rule's annotations
- `adr/` — architecture decisions, including the Context Graph decision trace,
  capability-gated Neo4j vector search, adaptive retrieval routing, agent
  platform trust boundaries, audience-bound access tokens, and JWT key
  rotation/revocation

Interview, outreach, and role-specific material is deliberately **not** kept in
this repository — it was removed on 2026-08-13 (it had been committed under
`archive/job-search/` and linked from the README, which meant anyone sent the
repo also received the outreach tracker and JD mappings).
