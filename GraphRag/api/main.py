"""FastAPI application — GraphRAG API with OAuth 2.0."""

import secrets
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.routes import auth, ingest, query, evaluation, kpis, corrections, kg_features
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

    log.info("startup.complete")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("shutdown.complete")


app = FastAPI(
    title="GraphRAG API",
    description=(
        "Enterprise GraphRAG pipeline with Neo4j, RabbitMQ, RAGAS, and Google ADK.\n\n"
        "**Browser auth:** visit [`/auth/dev-login`](/auth/dev-login) (dev) "
        "or [`/auth/login`](/auth/login) (Google)\n\n"
        "**M2M auth:** `POST /auth/token` with `client_credentials` grant"
    ),
    version="0.2.0",
    swagger_ui_parameters={"withCredentials": True},
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.jwt_secret_key,
    session_cookie="graphrag_session",
    max_age=3600,
    same_site="lax",
    https_only=False,  # set True behind HTTPS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth.router,       prefix="/auth",       tags=["Auth"])
app.include_router(ingest.router,     prefix="/ingest",     tags=["Ingestion"])
app.include_router(query.router,      prefix="/query",      tags=["Query"])
app.include_router(evaluation.router,  prefix="/evaluation",  tags=["Evaluation"])
app.include_router(kpis.router,        prefix="/kpis",        tags=["KPIs"])
app.include_router(corrections.router, prefix="/corrections", tags=["Corrections"])
app.include_router(kg_features.router, prefix="/kg",          tags=["KG Features"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


# ── Admin dashboard ────────────────────────────────────────────────────────────
# Mount the Dash admin panel at /admin using the WSGI bridge.
# Dash renders server-side via Flask/Werkzeug; WSGIMiddleware adapts it to ASGI.
try:
    from starlette.middleware.wsgi import WSGIMiddleware
    from graphrag.dashboard.app import app as dash_app
    app.mount("/admin", WSGIMiddleware(dash_app.server))
    log.info("startup.admin_dashboard_mounted", path="/admin")
except Exception as _dash_exc:
    log.warning("startup.admin_dashboard_unavailable", error=str(_dash_exc))
