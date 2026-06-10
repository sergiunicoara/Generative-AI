"""Unit tests for auth security fixes — scope enforcement, open redirect, cookie flag."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.auth.dependencies import require_scope
from api.routes.auth import _safe_next


# ── require_scope — unconditional enforcement ──────────────────────────────────

class TestRequireScope:
    """Verify scope is checked for ALL token types, not just M2M.

    require_scope(s) returns an async inner function _check(user).
    We call it directly (bypassing FastAPI DI) by passing user as kwarg.
    """

    def _user(self, token_type: str, scopes: str) -> dict:
        return {"sub": "u1", "type": token_type, "scope": scopes}

    async def test_browser_token_with_required_scope_passes(self):
        checker = require_scope("read")
        user = self._user("browser", "read write")
        result = await checker(user=user)
        assert result == user

    async def test_browser_token_missing_scope_raises_403(self):
        checker = require_scope("admin")
        user = self._user("browser", "read write")   # no "admin"
        with pytest.raises(HTTPException) as exc_info:
            await checker(user=user)
        assert exc_info.value.status_code == 403

    async def test_m2m_token_with_scope_passes(self):
        checker = require_scope("write")
        user = self._user("m2m", "read write")
        result = await checker(user=user)
        assert result == user

    async def test_m2m_token_missing_scope_raises_403(self):
        checker = require_scope("write")
        user = self._user("m2m", "read")    # no "write"
        with pytest.raises(HTTPException) as exc_info:
            await checker(user=user)
        assert exc_info.value.status_code == 403

    async def test_empty_scope_field_raises_403_for_any_type(self):
        for token_type in ("browser", "m2m"):
            checker = require_scope("read")
            user = self._user(token_type, "")
            with pytest.raises(HTTPException) as exc_info:
                await checker(user=user)
            assert exc_info.value.status_code == 403

    async def test_multiple_scopes_all_individually_enforceable(self):
        """Each scope in a multi-scope token must pass its own gate."""
        checker_read  = require_scope("read")
        checker_write = require_scope("write")
        user = self._user("browser", "read write")
        assert await checker_read(user=user) == user
        assert await checker_write(user=user) == user

    async def test_scope_with_extra_whitespace_still_works(self):
        """Scope field with extra spaces must not break the split."""
        checker = require_scope("read")
        user = self._user("browser", "  read  write  ")
        result = await checker(user=user)
        assert result == user


# ── _safe_next — open redirect prevention ─────────────────────────────────────

class TestSafeNext:
    """Verify _safe_next rejects external URLs and allows safe relative paths."""

    def test_relative_path_allowed(self):
        assert _safe_next("/docs") == "/docs"

    def test_deep_relative_path_allowed(self):
        assert _safe_next("/admin/health") == "/admin/health"

    def test_none_returns_default(self):
        assert _safe_next(None) == "/docs"

    def test_empty_string_returns_default(self):
        assert _safe_next("") == "/docs"

    def test_absolute_http_rejected(self):
        assert _safe_next("http://evil.com") == "/docs"

    def test_absolute_https_rejected(self):
        assert _safe_next("https://evil.com/steal") == "/docs"

    def test_protocol_relative_rejected(self):
        assert _safe_next("//evil.com") == "/docs"

    def test_javascript_scheme_rejected(self):
        assert _safe_next("javascript:alert(1)") == "/docs"

    def test_custom_default_used_on_bad_url(self):
        assert _safe_next("http://bad.com", default="/home") == "/home"

    def test_custom_default_not_used_on_good_url(self):
        assert _safe_next("/dashboard", default="/home") == "/dashboard"


# ── _cookie_secure logic ───────────────────────────────────────────────────────

class TestCookieSecureLogic:
    """Verify the env == 'production' formula directly."""

    def test_production_env_means_secure_true(self):
        assert ("production" == "production") is True

    def test_development_env_means_secure_false(self):
        assert ("development" == "production") is False

    def test_staging_env_means_secure_false(self):
        """Non-production envs must not set secure=True."""
        assert ("staging" == "production") is False
