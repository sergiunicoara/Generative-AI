"""Rate-limit bucketing and enforcement.

Three properties are pinned:

1. an authenticated caller is bucketed by its verified `sub` claim, so one
   credential driven from many source addresses is still throttled, and callers
   sharing one NAT egress do not throttle each other;
2. the subject is read from the *verified* claims the auth middleware stored,
   never from the raw Authorization header — otherwise a caller mints an
   unlimited number of buckets by varying an unverified token;
3. enforcement is async end-to-end. slowapi's limiter was synchronous, so a
   Redis-backed check blocked the event loop on every limited request.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api.limiter import (
    AsyncRateLimiter,
    RateLimitExceeded,
    _storage_uri,
    client_address,
    client_key,
    conversation_key,
    enforce_conversation_limit,
    rate_limit,
)


def _request(
    *,
    peer: str = "203.0.113.10",
    headers: dict | None = None,
    user: dict | None = None,
    path: str = "/query",
) -> SimpleNamespace:
    """A minimal stand-in with the attributes the key functions actually read."""
    return SimpleNamespace(
        client=SimpleNamespace(host=peer, port=1234),
        headers={k.lower(): v for k, v in (headers or {}).items()},
        state=SimpleNamespace(**({"user": user} if user is not None else {})),
        url=SimpleNamespace(path=path),
    )


class TestIdentityAwareKeys:
    def test_unauthenticated_request_keys_on_address(self):
        assert client_key(_request()) == "ip:203.0.113.10"

    def test_authenticated_request_keys_on_the_verified_subject(self):
        request = _request(user={"sub": "graphrag_abc", "tenant": "aerospace"})
        assert client_key(request) == "sub:graphrag_abc"

    def test_same_credential_from_many_addresses_shares_one_bucket(self):
        first = _request(peer="198.51.100.1", user={"sub": "graphrag_abc"})
        second = _request(peer="198.51.100.2", user={"sub": "graphrag_abc"})
        assert client_key(first) == client_key(second)

    def test_different_callers_behind_one_address_get_separate_buckets(self):
        first = _request(peer="203.0.113.10", user={"sub": "client-a"})
        second = _request(peer="203.0.113.10", user={"sub": "client-b"})
        assert client_key(first) != client_key(second)

    def test_subject_and_address_namespaces_cannot_collide(self):
        spoofed = _request(user={"sub": "203.0.113.10"})
        assert client_key(spoofed) != client_key(_request(peer="203.0.113.10"))

    def test_unverified_state_is_ignored(self):
        assert client_key(_request(user=None)) == "ip:203.0.113.10"
        assert client_key(_request(user={})) == "ip:203.0.113.10"

    def test_bearer_header_alone_does_not_create_a_bucket(self):
        request = _request(headers={"Authorization": "Bearer forged.token.value"})
        assert client_key(request) == "ip:203.0.113.10"

    def test_missing_client_does_not_crash(self):
        # Starlette leaves request.client None for some transports; an
        # AttributeError here would 500 every request on that transport.
        request = _request()
        request.client = None
        assert client_address(request) == "unknown"


class TestForwardedForHandling:
    def test_forwarded_for_is_ignored_without_a_declared_proxy_depth(self):
        request = _request(headers={"X-Forwarded-For": "1.2.3.4"})
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRAPHRAG_TRUSTED_PROXIES", None)
            assert client_address(request) == "203.0.113.10"

    def test_nth_hop_from_the_right_is_used_when_declared(self):
        request = _request(headers={"X-Forwarded-For": "9.9.9.9, 1.2.3.4, 10.0.0.1"})
        with patch.dict(os.environ, {"GRAPHRAG_TRUSTED_PROXIES": "2"}):
            assert client_address(request) == "1.2.3.4"


class TestEnforcement:
    async def test_requests_are_allowed_up_to_the_limit_then_rejected(self):
        limiter = AsyncRateLimiter(storage_uri=None)
        outcomes = [(await limiter.check("3/minute", "ip:1.2.3.4"))[0] for _ in range(5)]
        assert outcomes == [True, True, True, False, False]

    async def test_rejection_reports_a_usable_retry_after(self):
        limiter = AsyncRateLimiter(storage_uri=None)
        for _ in range(2):
            await limiter.check("2/minute", "ip:5.6.7.8")
        allowed, retry_after = await limiter.check("2/minute", "ip:5.6.7.8")
        assert allowed is False
        assert 0 < retry_after <= 60

    async def test_separate_keys_have_separate_budgets(self):
        limiter = AsyncRateLimiter(storage_uri=None)
        for _ in range(3):
            await limiter.check("3/minute", "sub:a")
        assert (await limiter.check("3/minute", "sub:b"))[0] is True

    async def test_storage_failure_allows_the_request(self, monkeypatch):
        # Rate limiting is a protection mechanism, not an authorization one:
        # failing closed would let a Redis blip deny all traffic, which is
        # strictly worse than briefly under-enforcing a throttle.
        limiter = AsyncRateLimiter(storage_uri=None)

        async def _explode():
            raise ConnectionError("storage gone")

        monkeypatch.setattr(limiter, "_ensure_backend", _explode)
        assert await limiter.check("1/minute", "sub:a") == (True, 0)

    async def test_concurrent_cold_start_builds_one_backend(self):
        limiter = AsyncRateLimiter(storage_uri=None)
        built = 0
        real_build = limiter._build_backend

        def counting_build():
            nonlocal built
            built += 1
            return real_build()

        limiter._build_backend = counting_build
        await asyncio.gather(*(limiter.check("100/minute", "sub:x") for _ in range(8)))
        assert built == 1


class TestDependencyShape:
    def test_rate_limit_returns_an_async_dependency(self):
        # slowapi's decorator forced every limited endpoint to accept
        # `request: Request` purely so the decorator could find it. A
        # dependency gets Request injected instead.
        dependency = rate_limit("5/minute")
        assert inspect.iscoroutinefunction(dependency)
        assert list(inspect.signature(dependency).parameters) == ["request"]

    async def test_dependency_raises_429_with_backoff_headers(self):
        from api import limiter as limiter_module

        limiter_module.limiter = AsyncRateLimiter(storage_uri=None)
        dependency = rate_limit("1/minute")
        request = _request(user={"sub": "client-a"})

        await dependency(request)  # first call consumes the budget
        with pytest.raises(RateLimitExceeded) as excinfo:
            await dependency(request)

        assert excinfo.value.status_code == 429
        assert int(excinfo.value.headers["Retry-After"]) >= 1
        assert excinfo.value.headers["X-RateLimit-Limit"] == "1/minute"


class TestConversationBucketing:
    """A conversation bucket must belong to *a caller's* conversation.

    Keying on the client-supplied session id alone would let anyone who
    learned or guessed another user's session id drain that conversation's
    budget, turning a cost control into a denial-of-service primitive against
    its rightful owner.
    """

    def test_bucket_is_scoped_to_the_caller_not_the_session_alone(self):
        victim = _request(user={"sub": "victim"})
        attacker = _request(user={"sub": "attacker"})
        assert conversation_key(victim, "s-1") != conversation_key(attacker, "s-1")

    def test_one_caller_two_conversations_get_separate_buckets(self):
        request = _request(user={"sub": "client-a"})
        assert conversation_key(request, "s-1") != conversation_key(request, "s-2")

    def test_same_caller_and_session_is_stable_across_calls(self):
        first = _request(peer="198.51.100.1", user={"sub": "client-a"})
        second = _request(peer="198.51.100.2", user={"sub": "client-a"})
        assert conversation_key(first, "s-1") == conversation_key(second, "s-1")

    async def test_sessionless_requests_are_not_conversation_limited(self):
        # Nothing to attribute conversation cost to, and rate_limit(QUERY_LIMIT)
        # already covers them. Must not consume a bucket or raise.
        from api import limiter as limiter_module

        limiter_module.limiter = AsyncRateLimiter(storage_uri=None)
        request = _request(user={"sub": "client-a"})
        for _ in range(5):
            await enforce_conversation_limit(request, "", limit="1/minute")

    async def test_conversation_over_its_limit_is_rejected_with_backoff(self):
        from api import limiter as limiter_module

        limiter_module.limiter = AsyncRateLimiter(storage_uri=None)
        request = _request(user={"sub": "client-a"})

        await enforce_conversation_limit(request, "s-1", limit="1/minute")
        with pytest.raises(RateLimitExceeded) as excinfo:
            await enforce_conversation_limit(request, "s-1", limit="1/minute")

        assert excinfo.value.status_code == 429
        assert int(excinfo.value.headers["Retry-After"]) >= 1

    async def test_exhausting_one_conversation_leaves_another_usable(self):
        # The point of the narrower bucket: a looping thread must not consume
        # the caller's ability to hold a different conversation.
        from api import limiter as limiter_module

        limiter_module.limiter = AsyncRateLimiter(storage_uri=None)
        request = _request(user={"sub": "client-a"})

        await enforce_conversation_limit(request, "looping", limit="1/minute")
        with pytest.raises(RateLimitExceeded):
            await enforce_conversation_limit(request, "looping", limit="1/minute")

        await enforce_conversation_limit(request, "healthy", limit="1/minute")


class TestSharedStorage:
    def test_redis_url_env_var_selects_shared_counters(self):
        with patch.dict(os.environ, {"REDIS_URL": "redis://cache:6379/1"}):
            assert _storage_uri() == "redis://cache:6379/1"

    def test_absent_redis_url_keeps_per_process_storage(self):
        with patch.dict(os.environ, {"REDIS_URL": ""}):
            assert _storage_uri() is None

    def test_async_scheme_is_applied_to_the_uri(self):
        # `limits` selects its async backend from the `async+` scheme; without
        # it the sync storage is returned and every check blocks the loop.
        assert AsyncRateLimiter._async_uri("redis://h:6379/0") == "async+redis://h:6379/0"
        assert AsyncRateLimiter._async_uri("async+redis://h:6379/0") == "async+redis://h:6379/0"

    async def test_unreachable_shared_storage_falls_back_without_raising(self):
        # Construction must not raise or block: the API has to start before
        # Redis is necessarily up.
        limiter = AsyncRateLimiter(storage_uri="redis://127.0.0.1:6399/0")
        backend = await limiter._ensure_backend()
        assert backend is not None


class TestNoSyncLimiterRemains:
    def test_slowapi_is_no_longer_imported(self):
        """No module may import slowapi again.

        Matches import statements only, not the word: api/limiter.py's
        docstring names slowapi to explain why the platform moved off it, and
        that prose is the reason someone will not silently reintroduce it.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        imports_slowapi = re.compile(r"^\s*(?:from|import)\s+slowapi", re.M)
        offenders = [
            str(path.relative_to(root))
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
            and not {"graphify-out", ".venv", "node_modules"} & set(path.parts)
            and imports_slowapi.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert not offenders, (
            f"slowapi's Limiter is synchronous and blocks the event loop on "
            f"Redis-backed checks; still imported by {offenders}"
        )
