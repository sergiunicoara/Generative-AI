"""FastAPI application — AI Knowledge Graph & Ontology Platform API with OAuth 2.0."""

from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth.dependencies import require_scope
from api.auth.default_auth import RequireAuthMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.limiter import limiter
from api.request_limits import RequestBodyLimitMiddleware
from api.routes import agent, auth, ingest, query, evaluation, kpis, corrections, kg_features, demo, context_graph, business, skills, wellknown, enterprise
from graphrag.core.config import get_settings, is_dev_env

log = structlog.get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before accepting traffic, teardown on shutdown."""
    from graphrag.observability.tracing import configure_tracing
    configure_tracing("graphrag-api")
    # ── Startup ───────────────────────────────────────────────────────────────
    # Resolve the OAuth resource identifiers once, here, so a malformed
    # GRAPHRAG_API_RESOURCE / GRAPHRAG_MCP_RESOURCE aborts startup instead of
    # turning every authenticated request into a 500 (they are read on the
    # token-mint and token-verify paths -- see ADR 0010). Same fail-closed
    # posture as Settings' production secret validation.
    from graphrag.core.resource_identifiers import known_resources

    log.info("startup.oauth_resources", resources=list(known_resources()))
    # Verify Redis session store connectivity.
    # When session_store_strict=true this raises immediately on failure so
    # the process exits with a visible error instead of silently falling back
    # to in-memory sessions.  Non-strict mode logs a warning and continues.
    from graphrag.retrieval.session_store import get_session_store
    store = get_session_store()
    try:
        await store.verify_connection()
    except (ConnectionError, ImportError) as exc:
        log.error(
            "startup.session_store_unavailable",
            error=str(exc),
            hint="set session_store_strict=false to allow in-memory fallback",
        )
        raise   # abort startup — let the process supervisor restart with correct config

    # ── RabbitMQ connectivity check ────────────────────────────────────────────
    # Non-fatal: the API can still serve read endpoints if the broker is down.
    # Logs an error so ops is alerted without aborting the whole startup.
    try:
        from graphrag.messaging.rabbitmq_client import get_rabbitmq
        await get_rabbitmq()
        log.info("startup.rabbitmq_ok")
    except Exception as exc:
        log.error(
            "startup.rabbitmq_unavailable",
            error=str(exc),
            impact="POST /ingest and POST /query will return 503 until broker is reachable",
        )

    log.info("startup.complete")
    try:
        yield
    finally:
        # ── Shutdown ──────────────────────────────────────────────────────────
        from graphrag.core.lifecycle import close_shared_resources

        await close_shared_resources()
        log.info("shutdown.complete")


app = FastAPI(  # noqa: E302 — rate limiter attached below
    title="AI Knowledge Graph & Ontology Platform API",
    description=(
        "Production knowledge graph platform with Neo4j, OWL-RL reasoning, SPARQL, TransE link prediction, RabbitMQ, RAGAS, and dual-LLM agentic retrieval (IRCoT).\n\n"
        "**Browser auth:** visit [`/auth/dev-login`](/auth/dev-login) (dev) "
        "or [`/auth/login`](/auth/login) (Google)\n\n"
        "**M2M auth:** `POST /auth/token` with `client_credentials` grant"
    ),
    version="0.2.0",
    swagger_ui_parameters={"withCredentials": True},
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_middleware(request, call_next):
    from graphrag.observability.correlation import correlation_context, new_correlation_id
    from graphrag.observability.tracing import trace_span

    incoming = request.headers.get("X-Correlation-ID", "").strip()
    correlation_id = incoming if 0 < len(incoming) <= 128 and incoming.isprintable() else new_correlation_id()
    request.state.correlation_id = correlation_id
    with correlation_context(correlation_id), trace_span(
        "http.request", http_method=request.method, http_route=request.url.path,
        correlation_id=correlation_id,
    ):
        response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

# ── Rate limiting ─────────────────────────────────────────────────────────────
# The limiter is a FastAPI dependency (see api/limiter.py), so there is no
# app-level exception handler to register: RateLimitExceeded is an HTTPException
# subclass and FastAPI's own handler already renders it with its Retry-After and
# X-RateLimit-Limit headers intact.
app.state.limiter = limiter

# ── Prometheus metrics ─────────────────────────────────────────────────────────
# Exposes /metrics in Prometheus text format.  Requires:
#   prometheus-fastapi-instrumentator>=2.3.0  (already in requirements.txt)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", tags=["Observability"])
    log.info("startup.prometheus_metrics_enabled", endpoint="/metrics")
except ImportError:
    log.warning("startup.prometheus_unavailable",
                hint="pip install prometheus-fastapi-instrumentator")

# ── Middleware ─────────────────────────────────────────────────────────────────
# Order matters. Starlette's most-recently-added middleware is OUTERMOST
# (runs first on the request, last on the response) -- see
# api/auth/default_auth.py's module docstring for why RequireAuthMiddleware
# must be added here, between SessionMiddleware and CORSMiddleware, and not
# last: CORSMiddleware must stay outermost so it still handles preflight
# OPTIONS requests and attaches CORS headers to 401 responses this
# middleware produces, instead of a preflight hitting a 401 before
# CORSMiddleware ever sees it.
app.add_middleware(
    SessionMiddleware,
    # Use a dedicated session secret distinct from the JWT signing key so
    # rotating one doesn't invalidate the other.  Falls back to a derived
    # value from jwt_secret_key for backward compatibility when not set.
    secret_key=settings.session_secret_key or (settings.jwt_secret_key + ":session"),
    session_cookie="graphrag_session",
    max_age=3600,
    same_site="lax",
    # Was `settings.env == "production"` -- exact-match only, so any unset
    # or misspelled prod env ("prod", "Production ", "") sent auth cookies
    # over plain HTTP. Inverted to the same allow-list every other env check
    # in this codebase now uses.
    https_only=not is_dev_env(settings.env),
)

app.add_middleware(RequireAuthMiddleware)

# This middleware buffers at most the configured bound and therefore must run
# outside application parsing. It applies to both fixed-length and chunked
# requests; endpoint models enforce smaller domain-specific limits.
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_request_bytes=settings.api_max_request_bytes,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-CSRF-Token", "X-Requested-With", "X-Correlation-ID"],
    expose_headers=["X-Correlation-ID"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(wellknown.router,  tags=["Auth"])   # unauthenticated OAuth discovery
app.include_router(auth.router,       prefix="/auth",       tags=["Auth"])
app.include_router(ingest.router,     prefix="/ingest",     tags=["Ingestion"])
app.include_router(query.router,      prefix="/query",      tags=["Query"])
app.include_router(enterprise.router, tags=["Enterprise Content"])
app.include_router(evaluation.router,  prefix="/evaluation",  tags=["Evaluation"],
                   dependencies=[Depends(require_scope("read"))])
app.include_router(kpis.router,        prefix="/kpis",        tags=["KPIs"])
app.include_router(corrections.router, prefix="/corrections", tags=["Corrections"])
app.include_router(kg_features.router, prefix="/kg",          tags=["KG Features"])
app.include_router(agent.router,       prefix="/agent",       tags=["Agent Tools"])
app.include_router(skills.router)
app.include_router(demo.router)
app.include_router(context_graph.router)
app.include_router(business.router)


@app.get("/health", tags=["Health"])
async def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["Health"])
async def health_ready():
    """Readiness probe — verifies Neo4j, Redis, and LLM provider health.

    Returns HTTP 200 when all dependencies are healthy, HTTP 503 otherwise.
    Orchestrators (Kubernetes, ECS, docker-compose healthcheck) should use
    this endpoint to gate traffic.
    """
    from fastapi import HTTPException

    checks: dict[str, str] = {}
    failed = False

    # ── Neo4j ──────────────────────────────────────────────────────────────────
    try:
        from graphrag.graph.neo4j_client import get_neo4j
        await get_neo4j().run("RETURN 1 AS ok")
        checks["neo4j"] = "ok"
    except Exception as exc:  # noqa: BLE001
        log.warning("health.ready_check_failed", component="neo4j", error=str(exc))
        checks["neo4j"] = "unavailable"
        failed = True

    # ── Redis ──────────────────────────────────────────────────────────────────
    # Redis is critical in multi-process deployments: the result_store uses it
    # to hand query results from the worker process back to the API process.
    # If Redis is down, queries execute but results never reach the client.
    try:
        from graphrag.retrieval.session_store import get_session_store
        alive = await get_session_store().ping()
        if alive:
            checks["redis"] = "ok"
        else:
            checks["redis"] = "unavailable (in-memory fallback active — result delivery broken in multi-process deployments)"
            failed = True
    except Exception as exc:  # noqa: BLE001
        log.warning("health.ready_check_failed", component="redis", error=str(exc))
        checks["redis"] = "unavailable"
        failed = True

    # ── LLM provider ─────────────────────────────────────────────────────────
    # Unlike Redis (no fallback exists — a Redis failure is always gating),
    # get_llm() is now a redundant, multi-provider FallbackLLM chain (see
    # llm_client.py — the 2026-07-24 incident is what motivated the primary
    # never having zero fallback again). Default chain is
    # Cerebras -> DeepSeek -> Groq (changed 2026-08-17, was DeepSeek -> Groq);
    # LLM_INGEST_PROVIDER overrides which provider leads. Only gate readiness
    # when EVERY provider in the chain is unhealthy, i.e. there is truly no
    # viable synthesis path left — a single link being down is degraded, not
    # down.
    try:
        from graphrag.core.provider_health import is_healthy
        from graphrag.core.config import get_settings
        cfg = get_settings()
        if cfg.llm_ingest_provider == "groq":
            chain = ["groq", "deepseek"]
        elif cfg.llm_ingest_provider == "deepseek":
            chain = ["deepseek", "groq"]
        else:
            chain = ["cerebras", "deepseek", "groq"]
        healthy = [p for p in chain if is_healthy(p)]
        if healthy and healthy[0] == chain[0]:
            checks["llm_provider"] = f"ok (primary={chain[0]})"
        elif healthy:
            checks["llm_provider"] = (
                f"degraded — {'/'.join(p for p in chain if p not in healthy)} unhealthy, "
                f"serving via {healthy[0]} fallback"
            )
        else:
            checks["llm_provider"] = f"error — all of {'/'.join(chain)} unhealthy, no viable LLM path"
            failed = True
    except Exception as exc:  # noqa: BLE001
        log.warning("health.ready_check_failed", component="llm_provider", error=str(exc))
        checks["llm_provider"] = "unavailable"
        failed = True

    if failed:
        raise HTTPException(status_code=503, detail={"status": "unhealthy", "checks": checks})
    return {"status": "healthy", "checks": checks}


# ── Admin dashboard ────────────────────────────────────────────────────────────
# Mount the Dash admin panel at /admin using a2wsgi (the modern WSGI→ASGI bridge;
# starlette.middleware.wsgi.WSGIMiddleware is deprecated and removed in newer
# Starlette versions).
try:
    from a2wsgi import WSGIMiddleware
    from graphrag.dashboard.app import app as dash_app
    app.mount("/admin", WSGIMiddleware(dash_app.server))
    log.info("startup.admin_dashboard_mounted", path="/admin")
except ImportError:
    log.warning("startup.admin_dashboard_unavailable",
                hint="pip install a2wsgi to enable the admin dashboard")
except Exception as _dash_exc:  # noqa: BLE001
    log.warning("startup.admin_dashboard_unavailable", error=str(_dash_exc))
