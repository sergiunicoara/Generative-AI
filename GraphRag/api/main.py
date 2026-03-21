"""FastAPI application — GraphRAG API with OAuth 2.0."""

import secrets

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.routes import auth, ingest, query, evaluation, kpis
from graphrag.core.config import get_settings

settings = get_settings()

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
app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
app.include_router(kpis.router,       prefix="/kpis",       tags=["KPIs"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
