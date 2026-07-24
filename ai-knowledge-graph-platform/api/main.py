"""FastAPI application — AI Knowledge Graph & Ontology Platform API with OAuth 2.0."""

import secrets
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.sessions import SessionMiddleware

from api.limiter import limiter
from api.routes import auth, ingest, query, evaluation, kpis, corrections, kg_features, demo
from graphrag.core.config import get_settings

log = structlog.get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before accepting traffic, teardown on shutdown."""
    # ── Startup ───────────────────────────────────────────────────────────────
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
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
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

# ── Rate limiting ─────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
app.add_middleware(
    SessionMiddleware,
    # Use a dedicated session secret distinct from the JWT signing key so
    # rotating one doesn't invalidate the other.  Falls back to a derived
    # value from jwt_secret_key for backward compatibility when not set.
    secret_key=settings.session_secret_key or (settings.jwt_secret_key + ":session"),
    session_cookie="graphrag_session",
    max_age=3600,
    same_site="lax",
    https_only=(settings.env == "production"),   # enforce HTTPS in prod
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Admin-Token", "X-Requested-With"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,       prefix="/auth",       tags=["Auth"])
app.include_router(ingest.router,     prefix="/ingest",     tags=["Ingestion"])
app.include_router(query.router,      prefix="/query",      tags=["Query"])
app.include_router(evaluation.router,  prefix="/evaluation",  tags=["Evaluation"])
app.include_router(kpis.router,        prefix="/kpis",        tags=["KPIs"])
app.include_router(corrections.router, prefix="/corrections", tags=["Corrections"])
app.include_router(kg_features.router, prefix="/kg",          tags=["KG Features"])
app.include_router(demo.router)


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
        checks["neo4j"] = f"error: {exc}"
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
        checks["redis"] = f"error: {exc}"
        failed = True

    # ── LLM provider ─────────────────────────────────────────────────────────
    # Unlike Redis (no fallback exists — a Redis failure is always gating),
    # get_llm() is now a redundant, multi-provider FallbackLLM (see
    # llm_client.py, 2026-07-24 — the primary having zero fallback is what
    # let a deprecated DeepSeek model id take down synthesis for ~40min
    # undetected). If the primary is unhealthy but the secondary is serving,
    # the service is degraded, not down — only gate readiness when BOTH are
    # unhealthy, i.e. there is truly no viable synthesis path left.
    try:
        from graphrag.core.provider_health import is_healthy
        from graphrag.core.config import get_settings
        cfg = get_settings()
        primary = "groq" if cfg.llm_ingest_provider == "groq" else "deepseek"
        secondary = "deepseek" if primary == "groq" else "groq"
        if is_healthy(primary):
            checks["llm_provider"] = f"ok (primary={primary})"
        elif is_healthy(secondary):
            checks["llm_provider"] = f"degraded — {primary} unhealthy, serving via {secondary} fallback"
        else:
            checks["llm_provider"] = f"error — both {primary} and {secondary} unhealthy, no viable LLM path"
            failed = True
    except Exception as exc:  # noqa: BLE001
        checks["llm_provider"] = f"error: {exc}"
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
