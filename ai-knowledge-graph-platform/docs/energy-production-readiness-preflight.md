# Energy Asset Intelligence: production-readiness preflight

This checklist is the boundary between the controls demonstrated in the POC
and evidence that must be collected in the client's environment. It is a
deployment gate, not a claim that an enterprise deployment is already live.

## Repository-controlled controls

- Canonical Energy YAML compiles deterministically to OWL/RDFS, SHACL, Neo4j
  constraints, and capability-loss diagnostics; `make semantic-model-check`
  rejects drift.
- R2RML materialises SAP-shaped relational records to RDF. The mapping emits
  both `energy:Asset` and the explicit `energy:WindTurbine` type for turbine
  rows, so a turbine query does not depend on an inference setting.
- Candidate RDF is SHACL validated, invalid source records are quarantined,
  and only a versioned published graph is served to the advisory API.
- The maintenance recommendation uses a committed SPARQL query and returns its
  asset, telemetry, work-order, bulletin, and threshold evidence. Missing
  evidence produces no conclusion.
- The optional Neo4j path is a governed, one-way read-model projection. It
  keeps the RDF IRI, datatype metadata, tenant, and source provenance, and
  rejects RDF that cannot be represented without guessing. RDF is never
  mutated from Neo4j.
- Dev-only Energy login is regression-tested. Production login remains bound
  to the configured identity-provider claims and tenant scopes.

## Required evidence before a client deployment

| Gate | Evidence to collect | Owner |
| --- | --- | --- |
| Source contracts | Approved SAP, Snowflake, and SharePoint schemas; identity, change-data-capture, deletion, and revision semantics | Data owners |
| Semantic sign-off | Business glossary, competency questions, URI policy, ontology version, SHACL severity and exception process | Domain steward |
| Identity and access | Production OIDC issuer/JWKS, group-to-tenant mapping, least-privilege scopes, break-glass and access-review process | IAM/security |
| RDF serving | Chosen GraphDB/Stardog/RDFox/Neptune edition, repository configuration, inference policy, backups, restore drill, encryption and credentials | Platform operations |
| Neo4j read model | Sizing, constraints/indexes, ACL-aware retrieval queries, rebuild schedule, reconciliation count and rollback procedure | GraphRAG/platform team |
| Reliability | Load, concurrency, failover, backup/restore, source replay, and disaster-recovery tests against representative volume | SRE/platform operations |
| Observability | Dashboards and alerts for mapping failures, SHACL quarantine, projection lag, query latency, authorization denials, and audit retention | SRE/security |
| Advisory governance | Named human approver, escalation SLAs, work-order integration contract, and explicit prohibition on direct equipment control | Operations owner |

## Repeatable local checks

```powershell
make semantic-model-check
python -m pytest -q tests/unit/test_energy_demo.py tests/unit/test_r2rml_rdf_materialization.py tests/unit/test_energy_lpg_projection.py tests/unit/test_energy_dev_login.py
python scripts/run_energy_demo.py --export-turtle artifacts/energy-demo.ttl
python scripts/project_energy_rdf_to_neo4j.py  # requires local Neo4j
```

The Docker-backed GraphDB and Neo4j recovery tests exercise portable dataset
or graph recovery locally. They do not replace a client-environment recovery
drill, capacity test, or security review.
