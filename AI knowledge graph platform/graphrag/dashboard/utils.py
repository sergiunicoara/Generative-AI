"""Shared helpers for the GraphRAG admin dashboard.

Provides the API client functions (_get, _post), error sentinels, and
shared UI primitives (_kpi_card, _err, style constants) used across all
tab modules.
"""

from __future__ import annotations

import os

import httpx
import structlog
from dash import html

log = structlog.get_logger(__name__)

# ── API client config ──────────────────────────────────────────────────────────

API_BASE  = os.getenv("GRAPHRAG_API_URL",   "http://localhost:8000")
API_TOKEN = os.getenv("GRAPHRAG_API_TOKEN", "")
_HEADERS  = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

# ── Shared UI style constants ──────────────────────────────────────────────────

CARD_STYLE = {
    "border": "1px solid #e0e0e0",
    "borderRadius": "8px",
    "padding": "16px",
    "margin": "8px",
    "minWidth": "150px",
    "textAlign": "center",
    "backgroundColor": "#fafafa",
}
H2 = {"marginTop": "20px", "marginBottom": "10px", "color": "#333"}
TAB_STYLE = {"fontFamily": "system-ui, sans-serif", "padding": "12px 20px"}


# ── API client ─────────────────────────────────────────────────────────────────

def _get(path: str, params: dict | None = None) -> dict | list | None:
    """Synchronous HTTP GET against the REST API.

    Returns the parsed JSON on success, or a dict with key ``_http_error``
    containing a human-readable error string on any failure.
    """
    try:
        r = httpx.get(f"{API_BASE}{path}", params=params, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:  # broad: any network/parse failure falls back to error sentinel
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def _post(path: str, json: dict | None = None) -> dict | None:
    """Synchronous HTTP POST.  Returns parsed JSON or ``{"_http_error": …}``."""
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json, headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = f"HTTP {exc.response.status_code} {exc.response.reason_phrase} — {path}"
        log.warning("dashboard.http_error", path=path,
                    status=exc.response.status_code, detail=detail)
        return {"_http_error": detail}
    except Exception as exc:  # broad: any network/parse failure falls back to error sentinel
        detail = f"{type(exc).__name__}: {exc} — {path}"
        log.warning("dashboard.request_error", path=path, error=detail)
        return {"_http_error": detail}


def http_error(data: dict | list | None) -> str | None:
    """Return the HTTP error string if *data* is an error sentinel, else None."""
    if isinstance(data, dict):
        return data.get("_http_error")
    return None


# ── UI primitives ──────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, color: str = "#1a1a2e") -> html.Div:
    return html.Div([
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "color": color}),
        html.Div(label, style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),
    ], style=CARD_STYLE)


def err(msg: str) -> html.P:
    return html.P(f"⚠ {msg}", style={"color": "#c00", "padding": "8px"})
