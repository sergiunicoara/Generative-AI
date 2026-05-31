# Contributing to the AI Knowledge Graph Platform

This guide covers the ADR process, PR workflow, coding standards, and how to
extend the platform with new capabilities.

---

## Architecture Decision Records (ADRs)

Every non-trivial architectural decision is captured as an ADR in `docs/adr/`.
The existing records set the baseline:

- [ADR-0001](docs/adr/0001-property-graph-over-triple-store.md) — Why Neo4j over RDF triple stores
- [ADR-0002](docs/adr/0002-forward-chaining-over-backward-chaining.md) — Why materialised inference over query-time
- [ADR-0003](docs/adr/0003-bayesian-confidence-accumulation.md) — Why `1−(1−c₁)(1−c₂)` over last-write-wins

### When to write an ADR

Write one when the decision:
- Changes the core data model (entities, edges, schema)
- Selects or replaces an external dependency (LLM provider, vector DB, queue)
- Introduces a new architectural pattern (e.g. new retrieval stage, inference rule type)
- Makes a trade-off between correctness, performance, and operational complexity

### ADR format

```markdown
# ADR-NNNN: Title

## Status
Accepted | Superseded by ADR-XXXX | Deprecated

## Context
What problem are we solving and why does it matter here?

## Decision
What we chose and the specific implementation approach.

## Consequences
What becomes easier. What becomes harder. What risks we accepted.
```

File: `docs/adr/NNNN-short-title.md` (zero-padded, kebab-case).

---

## Pull Request Process

### Branch naming

```
feat/short-description        # new capability
fix/short-description         # bug fix
refactor/short-description    # internal restructuring, no behaviour change
docs/short-description        # documentation only
```

### PR checklist

Before marking a PR ready for review:

- [ ] Tests added or updated for every changed behaviour
- [ ] `python -m pytest tests/unit tests/integration -q` — 0 failures
- [ ] `python scripts/demo_regulatory.py` runs clean (no live services needed)
- [ ] If config keys added: wired to a property in `graphrag/core/config.py`, not via `getattr(settings, "key", {})`
- [ ] If Cypher changed: tested against Neo4j 5.x (not just Neo4j 4.x syntax)
- [ ] If a new dependency added: added to `requirements.txt` with a minimum version pin
- [ ] Docstrings updated — no module-level docstrings referencing the old provider (e.g. "uses Gemini" after migrating to Groq)
- [ ] For ADR-level changes: ADR document created in `docs/adr/`
- [ ] Lessons updated in `tasks/lessons.md` if a new failure pattern was discovered

### Review expectations

- **Cypher queries**: reviewer checks tenant filter, composite entity key `(name, type, tenant)`, Neo4j 5.x syntax
- **LLM calls**: all text generation through `get_llm()`, embeddings through `get_embedder()` — no direct SDK calls outside `llm_client.py`
- **Async correctness**: `asyncio.get_running_loop()` not `get_event_loop()`; blocking I/O in `run_in_executor`
- **Tests**: mocks must intercept the actual call path — after LLM provider changes, re-verify mocks target the new path

---

## Adding a New KG Feature

### File placement

| Type | Location |
|---|---|
| New graph operation (persist, query, reason) | `graphrag/graph/` |
| New retrieval stage | `graphrag/retrieval/` |
| New ingestion step | `graphrag/ingestion/` |
| New API endpoint(s) | `graphrag/api/routes/` |
| New CLI script | `scripts/` |
| New unit tests | `tests/unit/test_<feature>.py` |

### Wiring checklist

1. **Config flag**: if the feature is toggleable, add it to `config/settings.yml` and declare it as a property on `Settings` in `graphrag/core/config.py`
2. **Tenant isolation**: every MATCH/MERGE on `:Entity` or `:RELATES_TO` must include `{..., tenant: $tenant}`
3. **Entity key**: MERGE on `:Entity` uses `(name, type, tenant)` — three dimensions, never two
4. **Strict mode**: if the feature has an external dependency (Redis, a new API), add a `verify_connection()` path called from the FastAPI lifespan hook
5. **Project structure**: update the `Project Structure` table in `README.md`
6. **ADR**: write one if the feature involves a non-obvious design trade-off

### Adding a new inference rule

Rules live in `graphrag/graph/inference_engine.py` as `InferenceRule` dataclasses.
Add to `DEFAULT_RULES` for domain-agnostic rules, or inject via the `rules=` parameter
for domain-specific ones loaded from a YAML ontology.

```python
InferenceRule(
    name="my_rule",
    rule_type="transitivity",   # transitivity | symmetry | inverse | composition
    relation="MY_RELATION",
    max_depth=3,
    confidence_decay=0.9,
)
```

### Adding a new domain ontology

1. Create `config/ontologies/<domain>.yml` following the schema in `aerospace_regulatory.yml`
2. Set `ontology.domain_ontology_path: "config/ontologies/<domain>.yml"` in `settings.yml`
3. The `OntologyRegistry` loads it automatically at startup via `load()`

---

## Coding Standards

### Python style

- **async everywhere** — all Neo4j, Redis, and LLM calls are async; no `asyncio.run()` inside library code
- **structlog for logging** — `log = structlog.get_logger(__name__)` at module level; use keyword args: `log.info("event.name", key=value)`
- **No bare `except Exception`** — narrow to specific types where possible; if broad catch is necessary, add a comment explaining what can actually raise
- **Type hints on all public functions** — return types included; `from __future__ import annotations` at top of every module

### Cypher standards

```cypher
-- Every entity MATCH must use the full composite key
MATCH (e:Entity {name: $name, type: $type, tenant: $tenant})

-- Relationship counting (Neo4j 5.x)
COUNT { (e)-[:RELATES_TO]-() } AS degree   -- NOT size((e)-[:RELATES_TO]-())

-- DDL requires explicit consume in the async driver
result = await session.run("CREATE INDEX ...")
await result.consume()
```

### LLM call standards

```python
# Text generation — always via central router
from graphrag.core.llm_client import get_llm
raw: str = await get_llm().generate(prompt, json_mode=True)

# Embeddings — always via central router
from graphrag.core.llm_client import get_embedder
vectors: list[list[float]] = await get_embedder().embed(texts)

# Never call Groq SDK or google-genai directly outside llm_client.py
```

---

## Testing

```bash
# Full unit suite (~100 tests, no live services)
python -m pytest tests/unit -q

# Integration tests (require Neo4j + RabbitMQ + Redis via docker-compose)
python -m pytest tests/integration -q

# Load tests
python -m pytest tests/load -q

# Regulatory demo (in-process mocks, zero services)
python scripts/demo_regulatory.py
```

Integration tests auto-skip if Docker is unavailable.

---

## Dependency management

Direct dependencies go in `requirements.txt` with `>=` lower bounds.
After any change, regenerate the full pin:

```bash
make lock   # pip-compile requirements.txt → requirements.lock
```

The lock file is committed. CI installs from the lock file.

---

## Lessons log

`tasks/lessons.md` captures mistake patterns found during development.
After any correction: add an entry following the existing format:

```markdown
## ANN — Short title

**What happened:** <one paragraph>
**Root cause:** <one paragraph>
**Rule:** <imperative rule in a blockquote>
```

The log is reviewed at session start for patterns relevant to current work.
