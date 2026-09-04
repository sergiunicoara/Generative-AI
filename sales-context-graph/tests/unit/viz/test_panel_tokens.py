from __future__ import annotations

import pytest

from src.core.config import get_settings

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def redis_and_secret(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    import src.viz.panel_tokens as panel_tokens

    monkeypatch.setenv("PANEL_TOKEN_SECRET", "unit-test-panel-secret")
    get_settings.cache_clear()
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(panel_tokens, "get_redis", lambda: client)
    yield panel_tokens
    await client.aclose()
    get_settings.cache_clear()


async def test_mint_and_verify_round_trip(redis_and_secret):
    pt = redis_and_secret
    token = await pt.mint_panel_token("ws-1", "opp-1")
    claims = await pt.verify_panel_token(token)
    assert claims.workspace_id == "ws-1"
    assert claims.opportunity_id == "opp-1"


async def test_verify_rejects_a_tampered_token(redis_and_secret):
    pt = redis_and_secret
    token = await pt.mint_panel_token("ws-1", "opp-1")
    encoded, _, signature = token.rpartition(".")
    tampered = encoded + "." + ("0" if signature[0] != "0" else "1") + signature[1:]
    with pytest.raises(pt.PanelTokenError, match="signature mismatch"):
        await pt.verify_panel_token(tampered)


async def test_verify_rejects_malformed_token(redis_and_secret):
    pt = redis_and_secret
    with pytest.raises(pt.PanelTokenError):
        await pt.verify_panel_token("not-a-token-at-all")
    with pytest.raises(pt.PanelTokenError):
        await pt.verify_panel_token("")


async def test_verify_rejects_an_expired_token(redis_and_secret, monkeypatch):
    pt = redis_and_secret
    monkeypatch.setenv("PANEL_TOKEN_TTL_SECONDS", "1")
    get_settings.cache_clear()
    token = await pt.mint_panel_token("ws-1", "opp-1")
    monkeypatch.setattr(pt.time, "time", lambda: 4102444800.0)  # year 2100 -- well past any TTL
    with pytest.raises(pt.PanelTokenError, match="expired"):
        await pt.verify_panel_token(token)


async def test_bumping_the_version_revokes_prior_tokens(redis_and_secret):
    pt = redis_and_secret
    token = await pt.mint_panel_token("ws-1", "opp-1")
    await pt.bump_panel_token_version("ws-1")
    with pytest.raises(pt.PanelTokenError, match="revoked"):
        await pt.verify_panel_token(token)
    # a freshly minted token after the bump is valid again
    new_token = await pt.mint_panel_token("ws-1", "opp-1")
    claims = await pt.verify_panel_token(new_token)
    assert claims.workspace_id == "ws-1"


async def test_revoking_one_workspace_does_not_affect_another(redis_and_secret):
    pt = redis_and_secret
    token_a = await pt.mint_panel_token("ws-a", "opp-1")
    token_b = await pt.mint_panel_token("ws-b", "opp-1")
    await pt.bump_panel_token_version("ws-a")
    with pytest.raises(pt.PanelTokenError, match="revoked"):
        await pt.verify_panel_token(token_a)
    claims_b = await pt.verify_panel_token(token_b)
    assert claims_b.workspace_id == "ws-b"


async def test_minting_without_a_secret_fails_closed(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    import src.viz.panel_tokens as panel_tokens

    monkeypatch.setenv("PANEL_TOKEN_SECRET", "")
    get_settings.cache_clear()
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(panel_tokens, "get_redis", lambda: client)
    with pytest.raises(panel_tokens.PanelTokenError, match="not configured"):
        await panel_tokens.mint_panel_token("ws-1", "opp-1")
    await client.aclose()
    get_settings.cache_clear()


async def test_minting_without_redis_fails_closed(monkeypatch):
    import src.viz.panel_tokens as panel_tokens

    monkeypatch.setenv("PANEL_TOKEN_SECRET", "unit-test-panel-secret")
    get_settings.cache_clear()
    monkeypatch.setattr(panel_tokens, "get_redis", lambda: None)
    with pytest.raises(panel_tokens.PanelTokenError, match="REDIS_URL"):
        await panel_tokens.mint_panel_token("ws-1", "opp-1")
    get_settings.cache_clear()


@pytest.mark.parametrize("token", ["\u00e9.abc", "abc.\u00e9", "\u00e9\u00e9.\u00e9\u00e9"])
async def test_non_ascii_token_raises_panel_token_error_not_a_500(monkeypatch, token):
    """A non-ASCII token used to escape as UnicodeEncodeError/TypeError.

    Neither is a PanelTokenError, and api/dependencies.py catches only
    PanelTokenError -- so one non-ASCII character in an unauthenticated,
    fully attacker-controlled query param turned a 401 into a 500.
    """
    import src.viz.panel_tokens as panel_tokens

    monkeypatch.setenv("PANEL_TOKEN_SECRET", "unit-test-panel-secret")
    get_settings.cache_clear()
    with pytest.raises(panel_tokens.PanelTokenError, match="non-ASCII"):
        await panel_tokens.verify_panel_token(token)
    get_settings.cache_clear()
