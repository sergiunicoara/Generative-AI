# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install all Python dependencies in an isolated layer so the final image
# only copies the installed packages, not build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder (no gcc/build tools in runtime image)
COPY --from=builder /install /usr/local

# Create a non-root user to run the app
RUN addgroup --system app && adduser --system --ingroup app app

# Copy application source
COPY --chown=app:app . .

# Drop to non-root for all subsequent operations
USER app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check via the /health/ready readiness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/health')" || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
