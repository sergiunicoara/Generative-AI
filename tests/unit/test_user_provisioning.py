"""Unit tests for api/auth/user_provisioning.py — the email -> tenant mapping
that closes F14 (GET /auth/callback used to accept any Google account and
unconditionally issue a token for settings.default_tenant).

_get_redis_sync() is patched to return None throughout: this dev environment
has a real Redis configured even under ENV=test, and without the patch these
tests would silently exercise the live-Redis path instead of the in-memory
fallback they're meant to isolate, leaving cross-run pollution in a real
`graphrag:user_tenant_map` hash. Same reasoning as SessionStore's
`redis_url=None` fixture pattern elsewhere in this suite.

See docs/context_graph_gap_plan.md F14.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from api.auth import user_provisioning as up


@pytest.fixture(autouse=True)
def _memory_only():
    """Force the in-memory fallback and start each test with a clean table."""
    up._users_mem.clear()
    with patch("api.auth.user_provisioning._get_redis_sync", return_value=None):
        yield
    up._users_mem.clear()


class TestNormalizeEmail:
    def test_lowercases_and_strips(self):
        assert up.normalize_email("  Alice@Example.COM  ") == "alice@example.com"


class TestGetSetDelete:
    def test_unprovisioned_email_returns_none(self):
        assert up.get_user_record("nobody@example.com") is None

    def test_set_then_get_round_trips(self):
        up.set_user_record("alice@example.com", tenant="acme", scopes=["read", "write"], added_by="admin@acme.com")
        record = up.get_user_record("alice@example.com")
        assert record["tenant"] == "acme"
        assert record["scopes"] == ["read", "write"]
        assert record["added_by"] == "admin@acme.com"
        assert "added_at" in record

    def test_lookup_is_case_insensitive(self):
        up.set_user_record("Alice@Example.com", tenant="acme", scopes=["read"], added_by="a")
        assert up.get_user_record("alice@example.com") is not None
        assert up.get_user_record("ALICE@EXAMPLE.COM") is not None

    def test_re_provisioning_overwrites(self):
        up.set_user_record("alice@example.com", tenant="acme", scopes=["read"], added_by="a")
        up.set_user_record("alice@example.com", tenant="acme", scopes=["read", "write"], added_by="a")
        assert up.get_user_record("alice@example.com")["scopes"] == ["read", "write"]

    def test_delete_removes_record(self):
        up.set_user_record("alice@example.com", tenant="acme", scopes=["read"], added_by="a")
        assert up.delete_user_record("alice@example.com") is True
        assert up.get_user_record("alice@example.com") is None

    def test_delete_of_unprovisioned_returns_false(self):
        assert up.delete_user_record("nobody@example.com") is False


class TestListUserRecordsTenantScoped:
    """Adversarial: one tenant's admin must never see another tenant's
    provisioned users."""

    def test_only_returns_matching_tenant(self):
        up.set_user_record("a@acme.com", tenant="acme", scopes=["read"], added_by="x")
        up.set_user_record("b@other.com", tenant="other", scopes=["read"], added_by="x")

        acme_users = up.list_user_records(tenant="acme")
        assert [r["email"] for r in acme_users] == ["a@acme.com"]

        other_users = up.list_user_records(tenant="other")
        assert [r["email"] for r in other_users] == ["b@other.com"]

    def test_empty_tenant_returns_empty_list(self):
        assert up.list_user_records(tenant="nobody-here") == []

    def test_tenant_is_required_keyword(self):
        with pytest.raises(TypeError):
            up.list_user_records()
