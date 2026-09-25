"""Access-token revocation by JWT ID.

Why this exists
---------------
A JWT is a bearer credential that is valid until it expires, and nothing else.
Before this module the platform had no way to answer "this token leaked, stop
honouring it" other than rotating the signing secret — which logs out every
other caller too. For a one-hour token that means an hour of known-compromised
access, or an outage. Neither is an acceptable answer during an incident.

Every issued token now carries a ``jti`` (RFC 7519 §4.1.7). Revoking a token
records that id in a deny-list, and every verification path consults it.

Design notes
------------
**Deny-list, not allow-list.** An allow-list of live tokens would make Redis a
hard dependency of every authenticated request — a Redis outage would deny all
traffic. A deny-list fails the other way: an outage means recently-revoked
tokens are honoured until the store returns, which is a smaller and
time-bounded failure. ``strict`` inverts that for deployments that would rather
fail closed, and is off by default so the safe-by-default choice is the
available one.

**Entries expire with the token.** A revoked ``jti`` only needs to outlive the
token that carries it. The TTL is the access-token lifetime plus a margin, so
the deny-list stays proportional to the revocation rate rather than growing
forever.

**Subject-wide revocation** ("log this client out everywhere") is recorded as a
timestamp per subject: any token whose ``iat`` predates it is refused. That
covers the case where the operator does not have the specific token in hand,
which during an incident is the usual case.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from graphrag.observability.operational_metrics import set_store_degraded

log = structlog.get_logger(__name__)

_JTI_PREFIX = "graphrag:revoked-jti:v1:"
_SUBJECT_PREFIX = "graphrag:revoked-subject:v1:"


class RevocationBackendUnavailable(RuntimeError):
    """Raised in strict mode when the revocation store cannot be reached."""


class TokenRevocationStore:
    """Redis-backed deny-list with a bounded in-process fallback."""

    def __init__(
        self,
        redis_url: str | None = None,
        *,
        ttl_seconds: int = 3900,
        strict: bool = False,
    ) -> None:
        self._redis_url = redis_url
        self._redis = None
        self._ttl = max(1, ttl_seconds)
        self._strict = strict
        # (value -> expiry epoch). Swept on access; a revocation list is small
        # by nature, so a plain dict with lazy expiry is the right shape here.
        self._memory_jti: dict[str, float] = {}
        self._memory_subject: dict[str, float] = {}

    async def connect(self) -> None:
        if not self._redis_url:
            if self._strict:
                raise RevocationBackendUnavailable(
                    "jwt_revocation_strict is set but no redis_url is configured"
                )
            log.warning(
                "token_revocation.no_redis",
                impact="revocations apply to this process only, not other replicas",
            )
            set_store_degraded("token_revocation", True)
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            log.info("token_revocation.redis_connected")
            set_store_degraded("token_revocation", False)
        except Exception as exc:  # Redis error types vary across redis-py versions.
            self._redis = None
            if self._strict:
                raise RevocationBackendUnavailable(
                    f"revocation store requires Redis but it is unreachable: {exc}"
                ) from exc
            log.warning("token_revocation.redis_unavailable", error=str(exc))
            # A revocation issued on another replica will not be seen here.
            set_store_degraded("token_revocation", True)

    # ── Writes ───────────────────────────────────────────────────────────────

    async def revoke_token(self, jti: str, *, subject: str = "", reason: str = "") -> bool:
        """Deny one specific token. Returns whether it was durably recorded."""
        if not jti:
            return False
        log.info("token_revocation.revoked", jti=jti[:12], subject=subject, reason=reason)
        if self._redis is not None:
            try:
                await self._redis.setex(f"{_JTI_PREFIX}{jti}", self._ttl, reason or "revoked")
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("token_revocation.write_error", error=str(exc))
                if self._strict:
                    raise RevocationBackendUnavailable(str(exc)) from exc
        # Reached only when there is no Redis at all, or the durable write
        # above failed. Either way this revocation exists in one process only.
        # Returning `self._redis is not None` here reported success whenever a
        # client object existed, so a failed write told the operator the
        # credential was dead fleet-wide when it was dead on one replica.
        self._memory_jti[jti] = time.time() + self._ttl
        return False

    async def revoke_subject(self, subject: str, *, reason: str = "") -> bool:
        """Deny every token issued to `subject` before now.

        Uses ``iat`` rather than enumerating tokens, because the operator
        performing an incident response has the client id, not the tokens.
        """
        if not subject:
            return False
        cutoff = time.time()
        log.info("token_revocation.subject_revoked", subject=subject, reason=reason)
        if self._redis is not None:
            try:
                await self._redis.setex(
                    f"{_SUBJECT_PREFIX}{subject}", self._ttl, str(cutoff),
                )
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("token_revocation.write_error", error=str(exc))
                if self._strict:
                    raise RevocationBackendUnavailable(str(exc)) from exc
        # Same durability contract as revoke_token above.
        self._memory_subject[subject] = cutoff
        return False

    # ── Reads ────────────────────────────────────────────────────────────────

    async def is_revoked(self, claims: dict[str, Any]) -> bool:
        """True if `claims` names a token that must no longer be honoured."""
        jti = str(claims.get("jti") or "")
        subject = str(claims.get("sub") or "")
        issued_at = claims.get("iat")

        if self._redis is not None:
            try:
                if jti and await self._redis.get(f"{_JTI_PREFIX}{jti}") is not None:
                    return True
                if subject:
                    cutoff = await self._redis.get(f"{_SUBJECT_PREFIX}{subject}")
                    if cutoff is not None and _issued_before(issued_at, cutoff):
                        return True
                return False
            except Exception as exc:  # noqa: BLE001
                log.warning("token_revocation.read_error", error=str(exc))
                if self._strict:
                    # Fail closed: an unreachable deny-list in strict mode
                    # means we cannot prove the token is still valid.
                    raise RevocationBackendUnavailable(str(exc)) from exc

        self._sweep()
        if jti and jti in self._memory_jti:
            return True
        cutoff = self._memory_subject.get(subject)
        return bool(cutoff is not None and _issued_before(issued_at, cutoff))

    def _sweep(self) -> None:
        now = time.time()
        for key in [k for k, expiry in self._memory_jti.items() if expiry <= now]:
            self._memory_jti.pop(key, None)
        for key in [k for k, at in self._memory_subject.items() if at + self._ttl <= now]:
            self._memory_subject.pop(key, None)

    async def stats(self) -> dict[str, Any]:
        self._sweep()
        return {
            "backend": "redis" if self._redis is not None else "memory",
            "revoked_tokens": len(self._memory_jti),
            "revoked_subjects": len(self._memory_subject),
            "ttl_seconds": self._ttl,
        }

    async def close(self) -> None:
        redis, self._redis = self._redis, None
        if redis is not None:
            try:
                await redis.aclose()
            except Exception as exc:  # noqa: BLE001 - shutdown must never raise
                log.warning("token_revocation.close_error", error=str(exc))


def _issued_before(issued_at: Any, cutoff: Any) -> bool:
    """Whether `issued_at` predates `cutoff`, tolerating missing/odd types.

    A token with no readable ``iat`` is treated as revoked when its subject has
    been revoked: we cannot show it was issued after the cutoff, and the safe
    reading of "cannot prove it is current" is to refuse it.
    """
    try:
        cutoff_value = float(cutoff)
    except (TypeError, ValueError):
        return False
    try:
        return float(issued_at) < cutoff_value
    except (TypeError, ValueError):
        return True


_store: TokenRevocationStore | None = None
_store_lock: asyncio.Lock | None = None


async def get_revocation_store() -> TokenRevocationStore:
    """Process singleton, safe against concurrent cold-start."""
    global _store, _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    async with _store_lock:
        if _store is None:
            import os

            from graphrag.core.config import get_settings, is_dev_env

            cfg = get_settings()
            redis_url = os.getenv("REDIS_URL") or cfg.retrieval.get("redis_url", "") or None
            # Outside dev, a revocation that only one replica knows about is a
            # revoked credential still honoured by the others -- fail closed
            # unless the operator explicitly opts out.
            strict = cfg.retrieval.get("jwt_revocation_strict")
            if strict is None:
                strict = not is_dev_env(cfg.env)
            candidate = TokenRevocationStore(
                redis_url=redis_url,
                ttl_seconds=cfg.jwt_revocation_ttl_seconds,
                strict=bool(strict),
            )
            await candidate.connect()
            _store = candidate
    return _store


async def close_revocation_store() -> None:
    """Close and reset the process singleton when it was initialized."""
    global _store, _store_lock
    store, _store = _store, None
    _store_lock = None
    if store is not None:
        await store.close()
