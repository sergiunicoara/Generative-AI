"""Explicit email -> tenant provisioning table (F14).

Problem solved
--------------
GET /auth/callback accepted ANY Google account and unconditionally issued a
token for settings.default_tenant -- there was no persistent mapping from a
Google identity to a tenant anywhere in the codebase. Every real browser
login landed in the same tenant regardless of who signed in, so production
multi-tenancy was never actually exercised by real users; only the dev-token
and M2M paths ever produced a non-default tenant.

This module is the missing mapping: an admin-scoped caller provisions
(email -> tenant, scopes) before that email can obtain a token via OAuth. An
unprovisioned email is rejected at /auth/callback, not defaulted anywhere.

Same storage shape as the M2M client registry already shipped in
api/routes/auth.py (_client_get/_client_set): Redis hash when available, with
an in-memory fallback for non-Redis environments. Not factored into a shared
helper -- graphrag/monitoring/alerts.py has its own equivalent
_get_redis_sync() too, so this follows the existing (if imperfect) per-module
convention rather than refactoring unrelated code as a side effect of this
fix.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

_USERS_KEY = "graphrag:user_tenant_map"
_users_mem: dict[str, dict] = {}   # fallback for non-Redis environments


def _get_redis_sync():
    """Return a sync Redis client for the provisioning table, or None."""
    try:
        import redis as redis_lib
        from graphrag.core.config import get_settings
        redis_url = get_settings().retrieval.get("redis_url", "")
        if not redis_url:
            return None
        return redis_lib.from_url(redis_url, socket_connect_timeout=1,
                                  socket_timeout=1, decode_responses=True)
    except (ImportError, OSError, ConnectionError, ValueError):
        return None


def _redis_error_types() -> tuple[type[BaseException], ...]:
    try:
        import redis as redis_lib
        return (redis_lib.exceptions.RedisError, OSError, ConnectionError, ValueError)
    except ImportError:
        return (OSError, ConnectionError, ValueError)


def normalize_email(email: str) -> str:
    """The provisioning key. Google emails are case-insensitive; storing and
    looking up with mixed case would make provisioning silently miss a match."""
    return email.strip().lower()


def get_user_record(email: str) -> dict | None:
    """Return the provisioning record for `email`, or None if unprovisioned."""
    key = normalize_email(email)
    r = _get_redis_sync()
    if r is not None:
        try:
            raw = r.hget(_USERS_KEY, key)
            return json.loads(raw) if raw else None
        except _redis_error_types():
            pass
    return _users_mem.get(key)


def set_user_record(
    email: str, *, tenant: str, scopes: list[str], added_by: str,
) -> dict:
    """Provision (or re-provision) `email` for `tenant` with `scopes`.

    Returns the stored record. Caller is responsible for scope validation and
    the escalation guard (granted scopes must not exceed the provisioning
    admin's own) -- this module only persists what it's given.
    """
    key = normalize_email(email)
    record = {
        "email": key,
        "tenant": tenant,
        "scopes": sorted(scopes),
        "added_by": added_by,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = json.dumps(record)
    r = _get_redis_sync()
    if r is not None:
        try:
            r.hset(_USERS_KEY, key, payload)
            return record
        except _redis_error_types():
            pass
    _users_mem[key] = record
    return record


def delete_user_record(email: str) -> bool:
    """Remove a provisioning record. Returns True if a record was removed."""
    key = normalize_email(email)
    r = _get_redis_sync()
    if r is not None:
        try:
            removed = r.hdel(_USERS_KEY, key)
            return bool(removed)
        except _redis_error_types():
            pass
    return _users_mem.pop(key, None) is not None


def list_user_records(*, tenant: str) -> list[dict]:
    """Return every record provisioned for `tenant`.

    Filtered server-side after fetching rather than via a secondary Redis
    index -- this table is an admin allowlist, not a hot path, so an O(n)
    scan over all provisioned users is the right tradeoff for now. `tenant`
    is required and non-optional: this is exactly the kind of read that must
    never default to "everyone's", the same principle applied throughout
    F11-F13.
    """
    r = _get_redis_sync()
    if r is not None:
        try:
            raw_map = r.hgetall(_USERS_KEY)
            records = [json.loads(v) for v in raw_map.values()]
            return [rec for rec in records if rec.get("tenant") == tenant]
        except _redis_error_types():
            pass
    return [rec for rec in _users_mem.values() if rec.get("tenant") == tenant]
