"""Direct HTTP-layer coverage for graphrag/enterprise/sharepoint.py's
MicrosoftGraphClient -- this did not exist before: its only prior tests
(tests/unit/test_semantic_interchange.py) inject a hand-rolled fake client
that bypasses MicrosoftGraphClient's own auth/retry/pagination entirely.

Mirrors tests/unit/test_sap_source_connector.py's and
test_snowflake_source_connector.py's concern-groups (token caching/
renewal, pagination, throttling/retry), plus what's specific to the
hardening done here: the token fetch is now routed through the injectable
client (previously always opened its own throwaway one, making it
untestable), a per-item content() failure no longer aborts the whole
sync_once() batch, and delta() has a bounded pagination-loop guard.
"""

from __future__ import annotations

import httpx
import pytest

from graphrag.enterprise.sharepoint import (
    GraphConnectorError,
    MicrosoftGraphClient,
    SharePointSourceConfig,
    SharePointSyncConnector,
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _config(**overrides) -> SharePointSourceConfig:
    defaults: dict = dict(
        source_id="sp-1", tenant_id="tenant-1", client_id="client-1",
        client_secret_env="TEST_SHAREPOINT_CLIENT_SECRET",
        site_id="site-1", drive_id="drive-1", tenant="acme",
    )
    defaults.update(overrides)
    return SharePointSourceConfig(**defaults)


def _is_token_request(request: httpx.Request) -> bool:
    return request.url.path.endswith("/oauth2/v2.0/token")


def _token_response(access_token: str = "tok", expires_in: int = 3600) -> httpx.Response:
    return httpx.Response(200, json={"access_token": access_token, "expires_in": expires_in})


@pytest.fixture(autouse=True)
def _client_secret(monkeypatch):
    monkeypatch.setenv("TEST_SHAREPOINT_CLIENT_SECRET", "s3cr3t")


class TestOAuthTokenCachingAndRenewal:
    async def test_token_is_reused_across_calls(self):
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if _is_token_request(request):
                token_calls += 1
                return _token_response()
            assert request.headers["Authorization"] == "Bearer tok"
            return httpx.Response(200, json={"value": []})

        client = MicrosoftGraphClient(_config(), client=_client(handler))
        await client.delta()
        await client.delta()
        assert token_calls == 1

    async def test_token_is_renewed_once_it_expires(self, monkeypatch):
        clock = {"now": 0.0}
        monkeypatch.setattr("graphrag.enterprise.sharepoint.time.monotonic", lambda: clock["now"])
        token_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal token_calls
            if _is_token_request(request):
                token_calls += 1
                return _token_response(access_token=f"tok-{token_calls}", expires_in=100)
            return httpx.Response(200, json={"value": []})

        client = MicrosoftGraphClient(_config(), client=_client(handler))
        await client.delta()
        assert token_calls == 1
        clock["now"] += 200
        await client.delta()
        assert token_calls == 2

    async def test_token_fetch_now_goes_through_the_injected_client(self):
        """The gap this hardening closes: previously the token fetch always
        opened its own throwaway httpx.AsyncClient, so an injected
        MockTransport client could never observe or mock it -- proven here
        simply by the fact that this test's MockTransport handler is what
        answers the token request at all (it would hang/error against a
        real network call otherwise)."""
        seen_token_request = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_token_request
            if _is_token_request(request):
                seen_token_request = True
                return _token_response()
            return httpx.Response(200, json={"value": []})

        client = MicrosoftGraphClient(_config(), client=_client(handler))
        await client.delta()
        assert seen_token_request is True

    async def test_missing_secret_env_raises(self, monkeypatch):
        monkeypatch.delenv("TEST_SHAREPOINT_CLIENT_SECRET", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("must not make any HTTP call")

        client = MicrosoftGraphClient(_config(), client=_client(handler))
        with pytest.raises(ValueError, match="TEST_SHAREPOINT_CLIENT_SECRET"):
            await client.delta()


class TestDeltaPagination:
    async def test_follows_nextlink_and_returns_the_final_deltalink(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            if "page=2" in str(request.url):
                return httpx.Response(200, json={
                    "value": [{"id": "2"}], "@odata.deltaLink": "https://graph.microsoft.com/v1.0/delta?token=final",
                })
            return httpx.Response(200, json={
                "value": [{"id": "1"}],
                "@odata.nextLink": "https://graph.microsoft.com/v1.0/delta?page=2",
            })

        client = MicrosoftGraphClient(_config(), client=_client(handler))
        items, delta_link = await client.delta()
        assert [i["id"] for i in items] == ["1", "2"]
        assert delta_link == "https://graph.microsoft.com/v1.0/delta?token=final"

    async def test_a_malformed_infinite_nextlink_chain_is_bounded(self):
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if _is_token_request(request):
                return _token_response()
            call_count += 1
            # Always returns another nextLink -- simulates a malformed/
            # looping chain that never terminates.
            return httpx.Response(200, json={
                "value": [], "@odata.nextLink": f"https://graph.microsoft.com/v1.0/delta?page={call_count}",
            })

        client = MicrosoftGraphClient(_config(), client=_client(handler), max_delta_pages=5)
        with pytest.raises(GraphConnectorError, match="exceeded 5 pages"):
            await client.delta()
        assert call_count <= 6  # bounded, not unbounded


class TestThrottlingAndRetry:
    async def test_429_eventually_succeeds(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if _is_token_request(request):
                return _token_response()
            attempts += 1
            if attempts == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json={"value": []})

        client = MicrosoftGraphClient(_config(), client=_client(handler), backoff_seconds=0)
        await client.delta()
        assert attempts == 2

    async def test_retries_are_bounded_then_a_clear_error(self):
        """Previously a single 429/5xx was immediately fatal -- no retry at
        all."""
        def handler(request: httpx.Request) -> httpx.Response:
            if _is_token_request(request):
                return _token_response()
            return httpx.Response(503)

        client = MicrosoftGraphClient(_config(), client=_client(handler), max_retries=2, backoff_seconds=0)
        with pytest.raises(GraphConnectorError):
            await client.delta()

    async def test_non_retryable_4xx_is_not_retried(self):
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            if _is_token_request(request):
                return _token_response()
            attempts += 1
            return httpx.Response(403)

        client = MicrosoftGraphClient(_config(), client=_client(handler), backoff_seconds=0)
        with pytest.raises(httpx.HTTPStatusError):
            await client.delta()
        assert attempts == 1


class _FakeGraphForSyncTests:
    """Minimal fake matching MicrosoftGraphClient's surface, used only to
    drive SharePointSyncConnector.sync_once()'s per-item failure isolation
    -- a different concern from MicrosoftGraphClient's own HTTP-layer
    behavior above, same shape test_semantic_interchange.py's existing
    fakes already use for the change-mapping logic."""

    def __init__(self, items, delta_link="cursor-2", content_by_id=None, fail_content_for=()):
        self._items = items
        self._delta_link = delta_link
        self._content_by_id = content_by_id or {}
        self._fail_content_for = set(fail_content_for)

    async def delta(self, cursor: str = ""):
        return self._items, self._delta_link

    async def content(self, item_id: str) -> bytes:
        if item_id in self._fail_content_for:
            raise httpx.HTTPStatusError("boom", request=httpx.Request("GET", "https://x"),
                                         response=httpx.Response(503))
        return self._content_by_id.get(item_id, b"content")

    async def permissions(self, item_id: str):
        return []


class _FakeSync:
    def __init__(self):
        self.applied: list[dict] = []

    async def current_cursor(self, source_id: str, tenant: str) -> str:
        return "cursor-1"

    async def apply_changes(self, source_id, changes, tenant, *, cursor, trigger):
        self.applied.append({"cursor": cursor, "changes": len(changes)})
        return {"applied": len(changes)}


class TestPerItemFailureIsolation:
    async def test_one_bad_item_does_not_abort_the_rest_of_the_batch(self):
        items = [
            {"id": "1", "name": "a.txt", "webUrl": "https://x/a.txt"},
            {"id": "2", "name": "b.txt", "webUrl": "https://x/b.txt"},
        ]
        graph = _FakeGraphForSyncTests(items, fail_content_for={"1"})
        sync = _FakeSync()
        connector = SharePointSyncConnector(_config(), graph_client=graph, sync_service=sync)

        result = await connector.sync_once()

        assert result["failed_items"] == ["1"]
        # Item 2 still got applied despite item 1's failure.
        assert sync.applied[0]["changes"] == 1

    async def test_cursor_does_not_advance_when_any_item_failed(self):
        items = [{"id": "1", "name": "a.txt", "webUrl": "https://x/a.txt"}]
        graph = _FakeGraphForSyncTests(items, delta_link="cursor-2", fail_content_for={"1"})
        sync = _FakeSync()
        connector = SharePointSyncConnector(_config(), graph_client=graph, sync_service=sync)

        await connector.sync_once()

        # Held back at the ORIGINAL cursor ("cursor-1"), not advanced to
        # "cursor-2" -- so the failed item is retried next run instead of
        # being silently skipped forever.
        assert sync.applied[0]["cursor"] == "cursor-1"

    async def test_cursor_advances_normally_when_nothing_failed(self):
        items = [{"id": "1", "name": "a.txt", "webUrl": "https://x/a.txt"}]
        graph = _FakeGraphForSyncTests(items, delta_link="cursor-2")
        sync = _FakeSync()
        connector = SharePointSyncConnector(_config(), graph_client=graph, sync_service=sync)

        await connector.sync_once()

        assert sync.applied[0]["cursor"] == "cursor-2"

    async def test_permissions_failure_still_fails_closed_per_item_not_batch(self):
        """Regression pin: the pre-existing permissions() fail-closed
        behavior (unaffected by this hardening) must still work."""
        class _GraphPermissionsFail(_FakeGraphForSyncTests):
            async def permissions(self, item_id: str):
                raise ValueError("cannot resolve permissions")

        items = [{"id": "1", "name": "a.txt", "webUrl": "https://x/a.txt"}]
        graph = _GraphPermissionsFail(items)
        sync = _FakeSync()
        connector = SharePointSyncConnector(_config(), graph_client=graph, sync_service=sync)

        result = await connector.sync_once()

        assert result["failed_items"] == []
        assert sync.applied[0]["changes"] == 1
