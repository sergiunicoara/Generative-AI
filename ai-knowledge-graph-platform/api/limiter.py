"""Shared slowapi rate-limiter instance.

Kept in its own module to avoid circular imports:
  api/main.py imports limiter to register it on the app
  api/routes/*.py import limiter to decorate endpoints

Limits (configurable via environment variable GRAPHRAG_RATE_LIMIT_*):
  POST /ingest : 20/minute — LLM + Neo4j write, quota-sensitive
  POST /query  : 60/minute — Redis read + LLM call, latency-sensitive

Key function uses the real client IP, honouring X-Forwarded-For when the
app runs behind a reverse proxy (nginx, ALB, Cloudflare, etc.).
"""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

INGEST_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_INGEST", "20/minute")
QUERY_LIMIT  = os.getenv("GRAPHRAG_RATE_LIMIT_QUERY",  "60/minute")

limiter = Limiter(key_func=get_remote_address)
