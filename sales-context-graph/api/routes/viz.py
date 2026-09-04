"""Minimal browser visualization for the P4.5 Context Graph plus the
product-completeness pass's Q&A/insights/ask/digest endpoints.

Not part of docs/plan.md's required API surface (§11) — a debugging/demo aid
layered on top of existing GET/POST endpoints. Four tabs on `/viz`:

- "Context Graph": renders Claims as a subject --predicate--> object
  node-link graph via a small hand-rolled force layout (no CDN dependency).
- "Browse Intents": a generic runner over every Q&A intent (api/routes/qa.py)
  and insights endpoint (api/routes/insights.py). Increment 20 replaced the
  previously hardcoded per-endpoint JS array with a fetch of GET
  /api/v1/qa/intents (src/nlq/catalog.py) — the same catalog the natural-
  language layer and its own structural tests already treat as the single
  source of truth, so this list can no longer drift from the real API surface.
- "Ask": free-text question -> POST /api/v1/ask, showing the resolved intent,
  confidence, any ambiguities the system refused to guess through, the
  grounded narrative (if requested), and the underlying structured result.
- "Alerts": GET /api/v1/digest, the proactive signals from Increment 17.

Separately, `GET /viz/panel` (Increment 20) is a compact, single-opportunity,
iframe-embeddable view (open objections + buying committee) meant for
embedding in Salesforce/Showpad — an embeddable panel, not a packaged
Salesforce/Showpad app (no OAuth, no AppExchange packaging; see README.md).
"""

from __future__ import annotations

import html
import json

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from api.dependencies import get_access_context, verify_api_key, verify_panel_token
from api.routes.viz_templates import (  # noqa: F401 -- BRAND_PALETTE, TYPOGRAPHY, and
    _BUYER_PORTAL_PAGE,
    _PAGE,
    _PANEL_PAGE,
    # the helper functions below are re-exported for tests/unit/api/test_viz_route.py,
    # which asserts against them via viz.BRAND_PALETTE / viz._root_css_vars() etc.
    BRAND_PALETTE,
    TYPOGRAPHY,
    _js_color_constants,
    _legend_swatches_html,
    _root_css_vars,
)
from src.auth.policy import AccessContext, AccessDenied, require_opportunity
from src.core.config import get_settings
from src.viz.panel_tokens import PanelTokenClaims, PanelTokenError, mint_panel_token

router = APIRouter(tags=["viz"])

# Deterministic id produced by the canonical data/sample Gong call when it is
# seeded into the default ws-demo workspace. Used only to make the opt-in
# public preview useful on first click; normal deployments keep the field empty.
_DEMO_CONVERSATION_ID = "eb91dade3fd7c13bd32a60989af6d0ea1b2a1d61cd601c8b6a0b640619282dbe"
_DEMO_OPPORTUNITY_ID = "14acbc36edf9af9616f29e2662a0fe9cd2ca16c843485c022780e4c75627ac32"
_DEMO_SELLER_ID = "f462ce5ef6096057f0603576b8946be8e4362e9d4ea28144f2e23568212c08a8"
_DEMO_REVIEWER_ID = "demo-reviewer"
_DEMO_BUYER_CONTACT_ID = "e7122acf4d06d2aa02c9f053637580536c39eb3ffc40e7ca51512c2b44145b72"
_DEMO_SUBJECT_ID = "spk_1"


class PanelTokenRequest(BaseModel):
    opportunity_id: str


@router.post("/viz/panel-token")
async def create_panel_token(
    body: PanelTokenRequest,
    workspace_id: str = Depends(verify_api_key),
    access: AccessContext = Depends(get_access_context),
) -> dict:
    """Mints the token GET /viz/panel now requires, replacing the raw
    X-Api-Key that used to sit directly in the panel's URL (docs/
    evaluation.md's Showpad-compatibility analysis, item 3). Requires the
    real API key -- whoever configures the Salesforce/Showpad embed calls
    this once, out of band, and only the returned token goes into the
    iframe src; the real key never reaches the browser. See
    src/viz/panel_tokens.py for the token's shape, expiry, and revocation.
    """
    if get_settings().authz_enforcement_enabled:
        try:
            require_opportunity(access, body.opportunity_id)
        except AccessDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
    try:
        token = await mint_panel_token(workspace_id, body.opportunity_id)
    except PanelTokenError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"token": token, "panel_url": f"/viz/panel?token={token}"}


@router.get("/viz", response_class=HTMLResponse)
async def context_graph_viz(locale: str = "en") -> str:
    locale = locale if locale in {"en", "ro"} else "en"
    settings = get_settings()
    if not (settings.demo_public_access_enabled and settings.demo_public_api_key):
        return (_PAGE.replace("__DEMO_OPPORTUNITY_ID_JSON__", "null")
                .replace("__DEMO_BROWSER_TTS_ENABLED__", "false")
                .replace("__LOCALE__", locale))
    # The key is exposed only in the deliberately opt-in demo mode. It is
    # scoped to the synthetic workspace and accepted only on read-only paths.
    key = html.escape(settings.demo_public_api_key, quote=True)
    workspace = html.escape(settings.demo_public_workspace_id, quote=True)
    page = _PAGE
    for element_id in ("apiKey", "qaApiKey", "askApiKey", "alertsApiKey", "reviewApiKey", "workflowApiKey"):
        page = page.replace(f'id="{element_id}" type="password" placeholder="X-Api-Key"',
                            f'id="{element_id}" type="password" value="{key}" placeholder="X-Api-Key"')
    for element_id in ("workspaceId", "qaWorkspaceId", "askWorkspaceId", "alertsWorkspaceId", "reviewWorkspaceId", "workflowWorkspaceId"):
        page = page.replace(f'id="{element_id}" value="ws-demo"',
                            f'id="{element_id}" value="{workspace}"')
    page = page.replace('id="conversationId" placeholder="optional"',
                        f'id="conversationId" value="{_DEMO_CONVERSATION_ID}" placeholder="optional"')
    page = page.replace('id="askOpportunityId">',
                        f'id="askOpportunityId" value="{_DEMO_OPPORTUNITY_ID}">')
    page = page.replace('id="askBuyerContactId">',
                        f'id="askBuyerContactId" value="{_DEMO_BUYER_CONTACT_ID}">')
    page = page.replace('id="askSubjectId">',
                        f'id="askSubjectId" value="{_DEMO_SUBJECT_ID}">')
    page = page.replace('id="reviewOpportunityId" placeholder="opportunity id">',
                        f'id="reviewOpportunityId" value="{_DEMO_OPPORTUNITY_ID}" placeholder="opportunity id">')
    page = page.replace('id="reviewSellerId" placeholder="seller id">',
                        f'id="reviewSellerId" value="{_DEMO_SELLER_ID}" placeholder="seller id">')
    page = page.replace('id="reviewerId" placeholder="required for mention decisions">',
                        f'id="reviewerId" value="{_DEMO_REVIEWER_ID}" placeholder="required for mention decisions">')
    page = page.replace('id="workflowOpportunityId" placeholder="required for Buyer Space and meeting brief">',
                        f'id="workflowOpportunityId" value="{_DEMO_OPPORTUNITY_ID}" placeholder="required for Buyer Space and meeting brief">')
    page = page.replace('id="workflowSellerId" placeholder="required for readiness">',
                        f'id="workflowSellerId" value="{_DEMO_SELLER_ID}" placeholder="required for readiness">')
    # The public preview does not hold a cloud TTS credential. Still make
    # voice demonstrable there with the browser's native speech engine; it
    # never sends answer text to a third-party TTS endpoint. An operator who
    # enables DEMO_PUBLIC_TTS_ENABLED instead gets the configured cloud path.
    page = page.replace(
        "__DEMO_BROWSER_TTS_ENABLED__",
        "false" if settings.demo_public_tts_enabled else "true",
    )
    page = page.replace("__DEMO_OPPORTUNITY_ID_JSON__", json.dumps(_DEMO_OPPORTUNITY_ID))
    return page.replace("__LOCALE__", locale)


@router.get("/viz/manifest.webmanifest")
async def viz_manifest() -> Response:
    return Response(
        json.dumps({
            "name": "Sales Context Graph", "short_name": "Sales Graph",
            "start_url": "/viz", "display": "standalone",
            "background_color": "#f0ece8", "theme_color": "#8c3fcc",
        }),
        media_type="application/manifest+json",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/viz/service-worker.js")
async def viz_service_worker() -> Response:
    """Offline shell and local draft support for the responsive `/viz` PWA.

    Authenticated API responses are deliberately never cached: a shared device
    must not expose a previous user's deal data while offline.
    """
    script = """const CACHE = 'scg-viz-v1';
self.addEventListener('install', event => event.waitUntil(caches.open(CACHE).then(cache => cache.add('/viz'))));
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET' || !new URL(request.url).pathname.startsWith('/viz')) return;
  event.respondWith(fetch(request).then(response => { const copy = response.clone(); caches.open(CACHE).then(cache => cache.put(request, copy)); return response; }).catch(() => caches.match(request).then(hit => hit || caches.match('/viz'))));
});"""
    return Response(script, media_type="application/javascript", headers={"Cache-Control": "no-cache"})


@router.get("/viz/buyer", response_class=HTMLResponse)
async def buyer_space_portal() -> str:
    """A standalone, buyer-safe surface.

    The invitation bearer token is supplied in the URL fragment, which a
    browser never sends to the server or a referrer.  The page subsequently
    sends it only in ``X-Buyer-Token`` request headers and keeps it in session
    storage for the tab lifetime.
    """
    return _BUYER_PORTAL_PAGE


@router.get("/viz/panel", response_class=HTMLResponse)
async def opportunity_panel(response: Response, token: str, claims: PanelTokenClaims = Depends(verify_panel_token)) -> str:
    settings = get_settings()
    allowed = settings.embed_allowed_origins.strip()
    # No CORSMiddleware is registered for this single-purpose header (see
    # docs/operations.md's "no new infrastructure until measured need" stance
    # applied here too) — frame-ancestors is set directly on this one route's
    # response. Empty setting means "no origin may embed this," not "any can."
    response.headers["Content-Security-Policy"] = (
        f"frame-ancestors {allowed}" if allowed else "frame-ancestors 'none'"
    )
    # workspace_id/opportunity_id come from the *validated token*, not
    # re-parsed from client-supplied query params -- a caller can no longer
    # claim a different opportunity_id than the one their token was minted
    # for. `token` itself is re-embedded verbatim so the page's own JS can
    # present it as X-Panel-Token on its two downstream fetches below.
    page = _PANEL_PAGE
    page = page.replace("__PANEL_TOKEN_JSON__", json.dumps(token))
    page = page.replace("__WORKSPACE_ID_JSON__", json.dumps(claims.workspace_id))
    page = page.replace("__OPPORTUNITY_ID_JSON__", json.dumps(claims.opportunity_id))
    return page


