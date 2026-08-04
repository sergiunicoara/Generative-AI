from __future__ import annotations

from fastapi import FastAPI

from api.routes import context, health, ingestions, unresolved_mentions

app = FastAPI(title="Sales Context Graph API", version="0.1.0")
app.include_router(health.router)
app.include_router(ingestions.router)
app.include_router(unresolved_mentions.router)
app.include_router(context.router)
