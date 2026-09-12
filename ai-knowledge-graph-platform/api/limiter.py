"""Async rate limiting for the API, built directly on `limits.aio`.

Why not slowapi
---------------
slowapi 0.1.10 builds its storage with `limits.storage.storage_from_string`
and its strategies from `limits.strategies` — both synchronous, with no async
Limiter in that version. A Redis-backed limiter therefore performed a
**blocking** round trip inside the event loop on every limited request,
stalling every other in-flight coroutine for its duration.

That was tolerable while only six low-rate endpoints were decorated and each
was dominated by an LLM or graph call. It stops being tolerable the moment a
cheap endpoint is limited, or the request rate rises — which is exactly when
rate limiting matters most. `limits` ships a first-class async implementation
(`limits.aio.storage`, `limits.aio.strategies`); this module uses it, so the
Redis hop is awaited rather than blocking.

Shape
-----
A FastAPI **dependency**, not a decorator. slowapi's decorator required every
limited endpoint to accept `request: Request` in its signature purely so the
decorator could find it; a dependency gets `Request` injected by the framework
and composes with the `Depends(require_scope(...))` gates already on these
routes.

Storage
-------
Counters live in Redis when `REDIS_URL` is set, so every replica shares one
bucket. In-process memory is the fallback, which silently multiplies each
limit by the replica count — a "20/minute" limit becomes
"20/minute/replica/restart", which is not a limit anyone reasoned about. That
degradation is reported through `graphrag_store_degraded` rather than only
logged.

A Redis outage falls back to per-process counting rather than failing every
request: losing precision on a rate limit is a much smaller incident than
returning 500 for all traffic.

Client identity
---------------
The bucket key is the authenticated subject (`sub` claim) when the request
carries a verified token, and the peer address otherwise.

Keying purely on IP gets both directions wrong: every caller behind one NAT or
egress gateway shares a bucket and throttles each other, while one credential
driven from many source addresses is never throttled at all. Unauthenticated
endpoints — `POST /auth/token` above all — necessarily still key on address,
which is exactly where address-based limiting is the right control.

`X-Forwarded-For` is honoured **only** when `GRAPHRAG_TRUSTED_PROXIES` is set
to the number of proxy hops in front of the app. The header is client-writable,
so trusting it unconditionally lets any caller forge a fresh identity per
request and bypass the limit entirely. With N trusted hops, the client address
is the Nth entry from the right — the last value a proxy you control appended.

Conversation limit
------------------
`enforce_conversation_limit` adds a second, narrower bucket per conversation.
Be precise about what it is for: `session_id` is supplied by the client
(`QueryRequest.session_id`), so a caller who wants more throughput can simply
send a new one. It is therefore **not** an abuse control — the per-subject
`QUERY_LIMIT` above remains the only thing an adversary cannot sidestep. What
this bounds is the cost of a *single* conversation: a client stuck in a retry
loop, a runaway agent, or a UI bug re-submitting the same thread, all of which
sit comfortably under the per-subject limit while burning LLM spend on one
session.

Its bucket key is the caller identity **and** the session id, never the session
id alone. Keying on the session alone would mean anyone who learned or guessed
another user's session id could exhaust that conversation's quota and deny
service to its real owner — turning a cost control into a griefing primitive.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import structlog
from fastapi import HTTPException, Request, status

from graphrag.observability.access_control_metrics import record_rate_limit_rejected
from graphrag.observability.operational_metrics import set_store_degraded

log = structlog.get_logger(__name__)

INGEST_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_INGEST", "20/minute")
QUERY_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_QUERY", "60/minute")
AUTH_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_AUTH", "10/minute")
# Deliberately a third of QUERY_LIMIT: a human conversation runs far below
# this, so it only bites on a loop, while still leaving a caller room to hold
# several conversations at once without the narrow bucket becoming the
# effective per-subject limit.
CONVERSATION_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_CONVERSATION", "20/minute")
# Own knob rather than reusing QUERY_LIMIT: search carries no LLM cost but
# still runs a cross-encoder reranking pass per request, so it is CPU-bound
# rather than spend-bound and may need tuning independently of the LLM-backed
# query path.
SEARCH_LIMIT = os.getenv("GRAPHRAG_RATE_LIMIT_SEARCH", "60/minute")


class RateLimitExceeded(HTTPException):
    """429 carrying the headers a well-behaved client needs to back off."""

    def __init__(self, retry_after: int, limit: str):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit}",
            headers={
                "Retry-After": str(max(1, retry_after)),
                "X-RateLimit-Limit": limit,
            },
        )


def _trusted_proxy_hops() -> int:
    try:
        return max(0, int(os.getenv("GRAPHRAG_TRUSTED_PROXIES", "0")))
    except ValueError:
        return 0


def _authenticated_subject(request: Request) -> str:
    """Return the verified `sub` claim, or "" for an unauthenticated request.

    Read from `request.state.user`, which RequireAuthMiddleware populates only
    after verifying the token's signature, expiry, audience, and revocation
    status — never from the raw Authorization header, which any caller can vary
    freely to mint themselves an unlimited number of buckets.
    """
    claims = getattr(request.state, "user", None)
    if not isinstance(claims, dict):
        return ""
    subject = claims.get("sub")
    return str(subject) if subject else ""


def client_address(request: Request) -> str:
    """The real client IP, given a declared proxy depth."""
    hops = _trusted_proxy_hops()
    if hops:
        forwarded = request.headers.get("x-forwarded-for", "")
        chain = [part.strip() for part in forwarded.split(",") if part.strip()]
        if len(chain) >= hops:
            return chain[-hops]
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def client_key(request: Request) -> str:
    """Rate-limit bucket for `request`.

    Namespaced by identity kind so a subject named like an IP address can never
    share a bucket with that address.
    """
    subject = _authenticated_subject(request)
    if subject:
        return f"sub:{subject}"
    return f"ip:{client_address(request)}"


def conversation_key(request: Request, session_id: str) -> str:
    """Rate-limit bucket for one caller's one conversation.

    Composite by design — see the module docstring: the session half alone is
    client-controlled and guessable, so it can neither be trusted as an
    identity nor safely own a bucket of its own.
    """
    return f"conv:{client_key(request)}:{session_id}"


async def enforce_conversation_limit(
    request: Request, session_id: str, limit: str = CONVERSATION_LIMIT
) -> None:
    """Consume one unit of `session_id`'s conversation budget.

    A plain call rather than a FastAPI dependency, unlike `rate_limit` above:
    `session_id` arrives in the request *body*, and a dependency would have to
    read and re-parse that body before the endpoint's own model does, purely to
    reach one field. The route calls this once it has a parsed body instead.

    Sessionless requests are not limited here — they have no conversation to
    attribute cost to, and `rate_limit(QUERY_LIMIT)` already covers them.
    """
    if not session_id:
        return
    allowed, retry_after = await limiter.check(limit, conversation_key(request, session_id))
    if not allowed:
        log.info(
            "rate_limit.conversation_rejected",
            path=request.url.path,
            key=conversation_key(request, session_id)[:32],
            limit=limit,
        )
        record_rate_limit_rejected(request.url.path)
        raise RateLimitExceeded(retry_after, limit)


def _storage_uri() -> str | None:
    """Shared counter storage, or None to keep per-process counting.

    Read from the `REDIS_URL` process environment only — deliberately not from
    `retrieval.redis_url` in settings.yml, whose committed value points at
    localhost as a development default. Every deployment already exports
    `REDIS_URL`, so this opts real deployments into shared counters while
    leaving unit tests and offline tooling on in-process storage instead of
    dialling a Redis that is not there.
    """
    return os.getenv("REDIS_URL", "").strip() or None


@dataclass
class _Backend:
    storage: object
    strategy: object
    shared: bool


class AsyncRateLimiter:
    """Moving-window limiter over async `limits` storage.

    Moving window rather than fixed: a fixed window lets a caller send a full
    quota at the end of one window and another full quota at the start of the
    next, which is a 2x burst at exactly the boundary an attacker can predict.
    """

    def __init__(self, storage_uri: str | None = None):
        self._storage_uri = storage_uri if storage_uri is not None else _storage_uri()
        self._backend: _Backend | None = None
        self._lock: asyncio.Lock | None = None

    async def _ensure_backend(self) -> _Backend:
        # Built lazily and under a lock: async storage must be constructed
        # inside a running loop, and two concurrent first requests would
        # otherwise each open a connection pool with one of them leaking.
        if self._backend is not None:
            return self._backend
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._backend is None:
                self._backend = self._build_backend()
        return self._backend

    def _build_backend(self) -> _Backend:
        from limits.aio.storage import MemoryStorage
        from limits.aio.strategies import MovingWindowRateLimiter

        if self._storage_uri:
            try:
                # Top-level factory, not limits.aio.storage: it dispatches on
                # the `async+` scheme and returns the async implementation.
                from limits.storage import storage_from_string

                storage = storage_from_string(self._async_uri(self._storage_uri))
                set_store_degraded("rate_limiter", False)
                return _Backend(storage, MovingWindowRateLimiter(storage), True)
            except Exception as exc:  # noqa: BLE001 - fall back, never fail startup
                log.warning(
                    "rate_limit.shared_storage_unavailable",
                    error=str(exc),
                    impact="limits are enforced per replica until Redis returns",
                )
        else:
            log.warning(
                "rate_limit.per_process_storage",
                impact="limits are enforced per replica; set REDIS_URL to share them",
            )
        storage = MemoryStorage()
        set_store_degraded("rate_limiter", True)
        return _Backend(storage, MovingWindowRateLimiter(storage), False)

    @staticmethod
    def _async_uri(uri: str) -> str:
        """`limits` selects its async backend from an `async+` URI scheme."""
        return uri if uri.startswith("async+") else f"async+{uri}"

    async def check(self, limit: str, key: str) -> tuple[bool, int]:
        """Consume one unit. Returns (allowed, seconds until reset).

        A storage failure allows the request. Rate limiting is a protection
        mechanism, not an authorization one — failing closed here would let a
        Redis blip deny all traffic, which is a strictly worse outcome than
        briefly under-enforcing a throttle.
        """
        from limits import parse

        try:
            backend = await self._ensure_backend()
            item = parse(limit)
            allowed = await backend.strategy.hit(item, key)
            if allowed:
                return True, 0
            stats = await backend.strategy.get_window_stats(item, key)
            import time as _time

            return False, max(1, int(stats.reset_time - _time.time()))
        except Exception as exc:  # noqa: BLE001
            log.warning("rate_limit.check_failed", error=str(exc), key=key[:32])
            set_store_degraded("rate_limiter", True)
            return True, 0

    async def reset(self) -> None:
        """Drop all counters. Test-only; there is no operational use."""
        backend = self._backend
        self._backend = None
        if backend is not None:
            reset = getattr(backend.storage, "reset", None)
            if reset is not None:
                try:
                    result = reset()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as exc:  # noqa: BLE001
                    log.debug("rate_limit.reset_failed", error=str(exc))


limiter = AsyncRateLimiter()


def rate_limit(limit: str):
    """FastAPI dependency enforcing `limit` for the calling identity.

    Usage::

        @router.post("", dependencies=[Depends(rate_limit(QUERY_LIMIT))])

    Placed in `dependencies=` rather than the signature so the endpoint's own
    parameters stay about its domain.
    """

    async def _check(request: Request) -> None:
        allowed, retry_after = await limiter.check(limit, client_key(request))
        if not allowed:
            log.info(
                "rate_limit.rejected",
                path=request.url.path,
                key=client_key(request)[:32],
                limit=limit,
            )
            record_rate_limit_rejected(request.url.path)
            raise RateLimitExceeded(retry_after, limit)

    return _check
