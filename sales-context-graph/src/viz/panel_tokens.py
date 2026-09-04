"""Panel tokens for GET /viz/panel (api/routes/viz.py) -- replaces putting
the real workspace API key in the panel's URL (docs/evaluation.md's
Showpad-compatibility analysis, item 3).

/viz/panel is deliberately a static, admin-configured iframe `src`
(README.md: "no OAuth flow, credentials passed as query params by whatever
embeds it"), so a short-TTL session token doesn't fit its usage model --
whoever configures the Salesforce/Showpad embed mints one panel token, once,
out of band (via the authenticated POST /viz/panel-token endpoint, which
requires the real X-Api-Key), and only that token -- scoped to one workspace
and opportunity, long-lived but bounded, independently revocable -- goes
into the iframe src. The real API key never reaches the browser.

Token shape: base64url(json payload) + "." + hex(HMAC-SHA256(payload, secret)).
Not a JWT -- no library dependency for a single-purpose, single-algorithm,
internally-verified token; the format is deliberately minimal.

Revocation: each workspace has a "current version" counter in Redis
(scg:panel_token_version:{workspace_id}, default 0 if unset). Minting embeds
the version current at mint time; verifying rejects a token whose embedded
version no longer matches the stored one. Bumping the counter instantly
invalidates every previously issued token for that workspace without
touching WORKSPACE_API_KEYS or needing a per-token blocklist.

Fails closed, not open: unlike most Redis use in this repo (job status,
alias_registry's warm cache), a missing/unreachable Redis here means panel
tokens cannot be minted or verified at all -- there is no safe "degrade to
no revocation capability" for an auth primitive, so this raises rather than
silently accepting tokens nothing could revoke.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from src.core.config import get_settings
from src.core.redis_client import get_redis

_VERSION_KEY_PREFIX = "scg:panel_token_version:"


class PanelTokenError(RuntimeError):
    """Raised for any minting/verification failure -- not configured,
    malformed, expired, or revoked. Callers translate this to a single
    401/503 at the API boundary; the specific reason is logged, never
    echoed back to the caller."""


@dataclass(frozen=True)
class PanelTokenClaims:
    workspace_id: str
    opportunity_id: str


def _secret() -> bytes:
    secret = get_settings().panel_token_secret
    if not secret:
        raise PanelTokenError("panel_token_secret is not configured")
    return secret.encode("utf-8")


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


async def _current_version(workspace_id: str) -> int:
    client = get_redis()
    if client is None:
        raise PanelTokenError("panel tokens require REDIS_URL (revocation store)")
    raw = await client.get(_VERSION_KEY_PREFIX + workspace_id)
    return int(raw) if raw is not None else 0


async def bump_panel_token_version(workspace_id: str) -> int:
    """Revoke every panel token previously issued for this workspace."""
    client = get_redis()
    if client is None:
        raise PanelTokenError("panel tokens require REDIS_URL (revocation store)")
    return await client.incr(_VERSION_KEY_PREFIX + workspace_id)


async def mint_panel_token(workspace_id: str, opportunity_id: str) -> str:
    version = await _current_version(workspace_id)
    payload = {
        "workspace_id": workspace_id,
        "opportunity_id": opportunity_id,
        "issued_at": time.time(),
        "version": version,
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


async def verify_panel_token(token: str) -> PanelTokenClaims:
    if not token or "." not in token:
        raise PanelTokenError("malformed panel token")
    # The token arrives straight off a query param / header, so it is fully
    # attacker-controlled. Both halves must be checked for ASCII *before* they
    # reach hmac: `encoded.encode("ascii")` raises UnicodeEncodeError and
    # compare_digest() raises TypeError on non-ASCII str, and neither is a
    # PanelTokenError -- so callers (api/dependencies.py catches PanelTokenError
    # only) turned a one-character malformed token into a 500 instead of a 401.
    if not token.isascii():
        raise PanelTokenError("malformed panel token: non-ASCII characters")
    encoded, _, signature = token.rpartition(".")
    expected_signature = hmac.new(_secret(), encoded.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise PanelTokenError("panel token signature mismatch")

    try:
        payload = json.loads(_b64decode(encoded))
        workspace_id = payload["workspace_id"]
        opportunity_id = payload["opportunity_id"]
        issued_at = float(payload["issued_at"])
        token_version = int(payload["version"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise PanelTokenError(f"malformed panel token payload: {exc}") from exc

    age = time.time() - issued_at
    if age < 0 or age > get_settings().panel_token_ttl_seconds:
        raise PanelTokenError("panel token expired")

    if token_version != await _current_version(workspace_id):
        raise PanelTokenError("panel token revoked")

    return PanelTokenClaims(workspace_id=workspace_id, opportunity_id=opportunity_id)
