# ADR-0005 — Redis as cross-process result store

**Status:** Accepted  
**Date:** 2026-05-30  
**Author:** Sergiu Nicoara

---

## Context

The platform runs three process types:

1. **API process** (`uvicorn api.main:app`) — accepts `POST /query`, returns a `query_id`, and later serves `GET /query/{id}` with the result
2. **Query worker** (`graphrag/workers/query_worker.py`) — consumes from RabbitMQ, runs the 6-stage retrieval pipeline, produces a `QueryResult`
3. **Dashboard** (`:8050`) — reads KPI summaries and latency metrics

In a single-process development setup, the API and query worker run in the same process and can share an in-memory dict. In any container deployment (Fly.io, Kubernetes, Docker Compose), they run in separate containers with no shared memory.

The initial implementation used an in-process dict (`_result_store: dict[str, dict]`). This worked in dev but silently failed in multi-container deployments: the API process's dict was never updated by the worker process, so `GET /query/{id}` always returned `{"status": "pending"}`.

---

## Decision

Use Redis as the shared result store between the API process and query workers.

Implementation: `graphrag/retrieval/result_store.py` — a thin wrapper around `redis.asyncio` with:
- `set(query_id, result_dict, ttl=3600)` — worker writes completed result
- `get(query_id)` — API reads result
- In-memory dict fallback when Redis is unreachable (transparent; warning logged)

The `get_result_store()` singleton is initialised at worker startup and at API startup independently. Both processes point at the same Redis instance via `REDIS_URL` env var.

---

## Considered alternatives

### Option A — PostgreSQL result table

- Durable across restarts (Redis data is lost on restart without AOF/RDB)
- Higher latency per write/read (~5ms vs ~0.5ms for Redis)
- Requires schema migration when result format changes
- Rejected: query results are ephemeral (TTL 1 hour default); durability not required; latency matters for the polling path

### Option B — RabbitMQ reply-to queue

- Worker publishes result to a per-query reply queue; API consumes it
- Eliminates polling (push instead of pull)
- Much more complex: API must maintain a long-lived consumer, handle queue expiry, manage per-request correlation IDs
- Rejected: the polling model (`GET /query/{id}` every 1s) is simpler and sufficient at current scale

### Option C — Keep in-process dict, require single-process deployment

- Zero infra dependency
- Fails immediately in any horizontal scaling scenario
- Rejected: multi-container is the deployment target (Fly.io)

---

## Consequences

**Positive:**
- API and worker processes are fully decoupled — can scale independently
- In-memory fallback means development works without Docker
- Query cache (`query_cache.py`) reuses the same Redis connection pool for cache pre-checks, so the Redis connection is amortised across two features

**Negative / watch:**
- Redis data is not durable by default. If Redis restarts mid-query, the result is lost and the user sees a perpetual `pending`. Mitigated by: (a) short TTL means the impact window is small; (b) Redis AOF can be enabled for production deployments
- TTL default of 3600s (1 hour) may be too short for batch pipelines. Expose `QUERY_RESULT_TTL_SECONDS` env var to allow ops override without redeploy

---

## Update 2026-07-24 — silent in-memory fallback was a split-brain risk

The original "in-memory dict fallback when Redis is unreachable (transparent;
warning logged)" behaviour described in the Decision section above was found to
be a real split-brain hazard, not just a dev convenience: on a mid-operation
Redis failure, `result_store.py`, `session_store.py`, and `alias_registry.py`
each silently wrote to the calling process's own in-memory dict, invisible to
every other process. A worker could write a query result that the API process
could never read back, and the client would hang until timeout with no error
in the logs.

Fixed: `ResultStore` no longer falls back silently — it logs an ERROR and drops
the write/read instead of pretending it succeeded. `SessionStore` in strict mode
re-raises instead of silently losing session context. `AliasRegistry` logs an
ERROR (not WARNING) on cross-worker push failure. The decision to use Redis as
the shared store is unchanged; only the failure-mode behavior was corrected.
