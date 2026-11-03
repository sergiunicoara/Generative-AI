# Interoperability targets and curator exchange

Implemented 2026-11-03 from the open roadmap items. Everything here is
**verified offline** (mocked HTTP transports, in-process rdflib, temporary
SQLite). Nothing was run against a real Stardog, GraphDB, Fuseki, Jira or GQL
engine, and none of it should be described as live-validated.

## Typed target outcomes and the capability matrix

`Diagnostic.outcome` states what a target does with a rule it cannot enforce
natively, derived from `fidelity` and the compensating control (`fidelity`
stays for existing consumers such as the Energy demo):

| Outcome | Meaning |
| --- | --- |
| `enforced` | The target rejects violations natively (matrix only) |
| `translated` | Converted to the closest construct (flattened, expanded, reified) |
| `runtime-mitigated` | Not enforced by the target; a named runtime control compensates |
| `warning` | Carried but not enforced here (for example open-world OWL cardinality) |
| `unsupported` | Neither enforced nor compensated |

`diagnostics.json` remains a loss report. Each model with a
`capability_matrix` artifact also gets `capability-matrix.json`: every declared
rule, once per target, with one of the five outcomes. A cell starts from the
target's static capability for that rule kind and is overridden by the
compiler's own diagnostic, so the matrix cannot claim more than the
diagnostics admit (a test checks every diagnostic against its cell, and that
every diagnostic code maps to a rule kind).

## Qualified cardinalities, relationship attributes, composite keys

Declared in the canonical YAML (`graphrag/semantic_model/models.py`):

```yaml
types:
  Product:
    properties:
      tenant: {datatype: string, required: true}
      sku: {datatype: string, required: true}
    composite_keys: [[tenant, sku]]        # members must exist and be required
relations:
  contains:
    source: Product
    target: Component
    qualified:                             # counted over targets of one class
      - on_class: SafetyComponent          # must be a subtype of the target
        cardinality: {min: 1, max: 2}
    properties:                            # relationship attributes
      quantity: {datatype: integer, required: true}
```

| Construct | OWL | SHACL | Neo4j | GQL |
| --- | --- | --- | --- | --- |
| Composite key | `owl:hasKey` | not enforceable: runtime-mitigated | composite `IS UNIQUE` constraint | unsupported |
| Qualified cardinality | `owl:onClass` + qualified restriction | `sh:qualifiedValueShape` + qualified counts | runtime-mitigated | unsupported |
| Relationship attribute | reified `rdf:Statement` class | shape on the reified statement | native property, runtime-mitigated | edge type property |

The runtime validator (`SemanticMutationValidator`) gained
`validate_relation_properties` (presence, datatype, undeclared names) and a
qualified-maximum check on `validate_relation` when the caller passes the
existing relation counts by target type. Like the unqualified maximum, that
check is only atomic if the caller runs it in the same transaction as the write.
Examples: the `contextual-knowledge` model (composite key, edge attribute) and
the synthetic model in `tests/unit/test_semantic_model_extensions.py`.

## GQL graph-type rendering

`render_gql` writes `graph-type.gql` for models that configure a `gql`
artifact: an ISO/IEC 39075-style `CREATE GRAPH TYPE` with node types, typed
properties and directed edge types. It is an interoperability surface; the
YAML stays canonical. **No GQL engine is available here, so the output has not
been accepted by one.** Inheritance and mixins are flattened, abstract types
are omitted (edges on an abstract endpoint expand to its concrete subtypes),
and keys, cardinalities and deprecation are not part of a graph type; each
loss has a `GQL_*` diagnostic.

## SPARQL endpoint conformance (Stardog, GraphDB, Fuseki)

```bash
python scripts/sparql_conformance.py                       # local baseline, 10 cases
python scripts/sparql_conformance.py --vendor stardog --base-url https://host:5820 \
    --db mydb --domain automotive --level instance --auth-env STARDOG_AUTH --json out.json
```

Runs the SPARQL competency suite (`evals/competency/`) against any SPARQL 1.1
endpoint through the existing read-only `RemoteSPARQLEndpoint`, and prints the
dataset fingerprint (sha256 over `cases.json` and the exact TTL inputs) the
operator must have loaded. Exit code 1 on any mismatch or zero cases. Only
`SELECT`/`ASK` text is sent. A Fuseki URL builder was added to `triplestore.py`
(documented default paths, unverified against a running Fuseki).

## GraphQL consumer API

`POST /graphql` (read scope) is a read-only GraphQL schema whose every
resolver calls the capability registry (`build_registry(audit_transport="graphql")`),
so scopes, tenant binding (the tenant is never an argument), argument
validation, size limits and audit are the same as for MCP and the other routes.
Fields: `graphStats`, `facts`, `entity`, `precedents`, `evaluateKnowledge`,
`capabilities`. Query only; depth, length and top-level field limits apply.
Answer generation is deliberately not exposed: `kg.answer.query` writes traces
and may call an LLM. Payloads are a `JSON` scalar (no static typing of results).
`requirements.lock` needs `make lock` to pick up `graphql-core`.

## Curator exchange (CSV/XLSX and Jira)

Under `/kg/ontology/proposals`:

| Route | Scope | Purpose |
| --- | --- | --- |
| `GET /export.csv`, `GET /export.xlsx` | read | Export proposals; cells starting with `= + - @`, tab or CR are escaped |
| `POST /decisions/import` | write | Decisions sheet as the raw body; `dry_run` defaults to true; any invalid row applies nothing |
| `POST /{id}/jira` | write | Create one Jira issue from a proposal; 503 `jira_not_configured` unless `JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`, `JIRA_PROJECT_KEY` are set |

The reviewer recorded on an import is the token `sub`; request and body
`reviewed_by` are never read. Known limits: the Jira route is not idempotent
(two calls create two tickets; fixing that needs the issue key stored on the
proposal, a graph-schema decision), apply is sequential and not transactional
(per-row results are reported), and the Jira client was only exercised against
a mock transport.

## Named graphs, release stamps and the shape registry (2026-11-03)

* **Layout** (ADR-0013): `https://graphrag.example.com/graph/{tenant}/{source}/{publication}`.
  `graphrag/graph/named_graphs.py` builds validated graph IRIs, lists a
  tenant's graphs with SPARQL and drops them by name, and serialises to and
  from JSON-LD. Tested on an in-memory `rdflib.Dataset`; existing exports still
  write a single graph.
* **Release stamp.** `export_graph(..., release=ref)` adds a
  `KnowledgeRelease` node with release id and content hash, referenced by every
  projected item and exception (part of the canonical model and its shapes).
  `release_stamp(graph)` reads it back and
  `KnowledgeEvaluationService.stale_projections` flags projections whose
  content hash differs from the active release; identity is the content, so a
  rollback to identical content is current again.
* **Shape registry as RDF.** `registry_graph(models, bindings)` describes each
  model (id, version, model and shapes SHA-256, shapes path, active flag) and
  binds data graphs to models with `dcterms:conformsTo`; `shapes_for_graph`
  answers "which shapes apply to graph X" with one SPARQL query, and
  `shapes_drift` reports committed shapes that no longer match the recorded
  hash.

