"""Confluence connector: page-tree permission mapping, change detection and
removal of deleted pages (offline; httpx.MockTransport and fakes only)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from graphrag.context_graph.knowledge_evaluation import can_read
from graphrag.enterprise.confluence import (
    ConfluenceClient,
    ConfluenceSourceConfig,
    ConfluenceSyncConnector,
    access_policy,
    storage_to_text,
)
from graphrag.enterprise.models import AccessContext, ACLState, SyncChangeType


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("TEST_CONFLUENCE_TOKEN", "t0ken")


def _config(**overrides) -> ConfluenceSourceConfig:
    base = dict(source_id="wiki", base_url="https://acme.atlassian.net", email="bot@acme.example",
                api_token_env="TEST_CONFLUENCE_TOKEN", space_keys=("ENG",), tenant="acme",
                space_principals={"ENG": ("group:eng",)}, backoff_seconds=0)
    base.update(overrides)
    return ConfluenceSourceConfig(**base)


def _reader(*principals: str) -> AccessContext:
    return AccessContext(subject_id="u", principals=list(principals), groups_resolved=True, resolution_source="test")


# -- permission mapping ---------------------------------------------------------------------


def test_an_unrestricted_page_inherits_the_space_readers():
    policy, reason = access_policy([set(), set()], ("group:eng",))
    assert reason == "" and policy.state == ACLState.KNOWN and policy.allow_principals == ["group:eng"]
    assert policy.requires_group_resolution


def test_a_space_without_configured_readers_is_denied():
    policy, reason = access_policy([set()], None)
    assert policy.state == ACLState.UNKNOWN and reason == "space_readers_unknown"
    assert not can_read(policy, _reader("group:eng"))


def test_an_ancestor_restriction_is_inherited_by_an_unrestricted_child():
    policy, _ = access_policy([set(), {"user:ana", "group:legal"}], ("group:eng",))
    assert policy.allow_principals == ["group:legal", "user:ana"]
    assert not can_read(policy, _reader("group:eng")), "space access alone does not pass the parent restriction"


def test_restrictions_on_several_levels_intersect():
    policy, _ = access_policy([{"user:ana", "user:bob"}, {"user:bob", "group:legal"}], ("group:eng",))
    assert policy.allow_principals == ["user:bob"]
    assert can_read(policy, _reader("user:bob"))
    assert not can_read(policy, _reader("user:ana"))


def test_an_empty_intersection_is_denied_not_opened():
    policy, reason = access_policy([{"user:ana"}, {"group:legal"}], ("group:eng",))
    assert policy.state == ACLState.UNKNOWN and reason == "restriction_intersection_empty"


# -- sync ---------------------------------------------------------------------------------


def _page(page_id, *, ancestors=(), when="2026-11-04T10:00:00.000Z", status="current", body="<p>Hello <b>world</b></p>"):
    return {"id": page_id, "title": f"Page {page_id}", "status": status, "space": {"key": "ENG"},
            "version": {"number": 3, "when": when}, "ancestors": [{"id": a} for a in ancestors],
            "body": {"storage": {"value": body}}, "_links": {"webui": f"/spaces/ENG/pages/{page_id}"}}


class _FakeClient:
    def __init__(self, pages, restrictions, failing=()):
        self.pages, self.restrictions, self.failing = pages, restrictions, set(failing)
        self.restriction_calls: list[str] = []

    async def changed_pages(self, since):
        self.since = since
        return self.pages

    async def page_ids(self):
        return [p["id"] for p in self.pages]

    async def read_restrictions(self, page_id):
        self.restriction_calls.append(page_id)
        if page_id in self.failing:
            raise httpx.ConnectError("down")
        return set(self.restrictions.get(page_id, set()))


def _sync(cursor=""):
    sync = MagicMock()
    sync.current_cursor = AsyncMock(return_value=cursor)
    sync.apply_changes = AsyncMock(return_value={"run_id": "r", "queued": 0, "tombstoned": 0})
    sync.reconcile = AsyncMock(return_value={"tombstoned": 1})
    return sync


async def test_sync_maps_inherited_restrictions_and_advances_the_cursor():
    client = _FakeClient(
        [_page("1"), _page("2", ancestors=("1",), when="2026-11-04T11:00:00.000Z")],
        {"1": {"group:legal"}},
    )
    sync = _sync("2026-11-01T00:00:00+00:00")
    result = await ConfluenceSyncConnector(_config(), client, sync).sync_once()
    changes = sync.apply_changes.await_args.args[1]
    assert [c.access_policy.allow_principals for c in changes] == [["group:legal"], ["group:legal"]]
    assert changes[0].text == "Hello world" and changes[0].metadata.source_version == "3"
    assert sync.apply_changes.await_args.kwargs["cursor"] == "2026-11-04T11:00:00.000Z"
    assert client.restriction_calls.count("1") == 1, "ancestor restrictions are read once per run"
    assert result["received"] == 2 and result["failed_items"] == []


async def test_an_unreadable_restriction_is_denied_and_audited():
    client = _FakeClient([_page("2", ancestors=("1",))], {}, failing=("1",))
    connector = ConfluenceSyncConnector(_config(), client, _sync())
    await connector.sync_once()
    change = connector._sync.apply_changes.await_args.args[1][0]
    assert change.access_policy.state == ACLState.UNKNOWN
    assert connector.acl_audit[0]["reason"] == "restrictions_unreadable"


async def test_a_trashed_page_becomes_a_delete_and_reconcile_removes_vanished_pages():
    sync = _sync()
    client = _FakeClient([_page("9", status="trashed")], {})
    connector = ConfluenceSyncConnector(_config(), client, sync)
    await connector.sync_once()
    assert sync.apply_changes.await_args.args[1][0].change_type == SyncChangeType.DELETE
    client.pages = [_page("1")]
    await connector.reconcile()
    assert sync.reconcile.await_args.args == ("wiki", ["1"], "acme")


async def test_a_failed_page_holds_the_cursor_back():
    sync = _sync("2026-11-01T00:00:00+00:00")
    connector = ConfluenceSyncConnector(_config(), _FakeClient([_page("1")], {}), sync)

    async def boom(page, cache):
        raise httpx.ReadTimeout("slow")

    connector._to_change = boom
    result = await connector.sync_once()
    assert result["failed_items"] == ["1"]
    assert sync.apply_changes.await_args.kwargs["cursor"] == "2026-11-01T00:00:00+00:00"


# -- HTTP client ---------------------------------------------------------------------------


async def test_client_searches_with_cql_paginates_and_reads_restrictions():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/content/search"):
            start = int(request.url.params["start"])
            more = {"next": "/wiki/rest/api/content/search?start=1"} if start == 0 else {}
            return httpx.Response(200, json={"results": [{"id": str(start + 1)}], "_links": more})
        return httpx.Response(200, json={"restrictions": {
            "user": {"results": [{"accountId": "ana"}]}, "group": {"results": [{"name": "legal", "id": "g-1"}]}}})

    client = ConfluenceClient(_config(space_keys=("ENG", "OPS")),
                              httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    pages = await client.changed_pages("2026-11-04T10:15:30+00:00")
    assert [p["id"] for p in pages] == ["1", "2"]
    cql = seen[0].url.params["cql"]
    assert 'space in ("ENG","OPS")' in cql and 'lastmodified >= "2026/11/04 10:15"' in cql
    assert seen[0].headers["authorization"].startswith("Basic ")
    assert await client.read_restrictions("7") == {"user:ana", "group:g-1"}


def test_config_rejects_unsafe_urls_and_bad_space_keys():
    with pytest.raises(ValueError):
        _config(base_url="ftp://acme")
    with pytest.raises(ValueError):
        _config(space_keys=('ENG") OR space="X',))


def test_storage_format_becomes_plain_text():
    assert storage_to_text("<h1>Title</h1><p>a &amp; b</p><ul><li>one</li></ul>") == "Title\na & b\none"


# -- routes --------------------------------------------------------------------------------


def _app(tenant="acme"):
    from api.auth.dependencies import get_current_user
    from api.routes import enterprise

    app = FastAPI()
    app.include_router(enterprise.router)
    app.dependency_overrides[get_current_user] = lambda: {"scope": "read write", "sub": "u", "tenant": tenant}
    return TestClient(app)


def test_routes_are_tenant_bound_and_unknown_sources_are_404():
    connector = MagicMock()
    connector.config = _config()
    connector.sync_once = AsyncMock(return_value={"queued": 1})
    with patch("api.routes.enterprise.ConfluenceSyncConnector.from_settings", return_value=connector):
        assert _app().post("/sync/confluence/wiki/run").json() == {"queued": 1}
        assert _app(tenant="other").post("/sync/confluence/wiki/run").status_code == 403
    with patch("api.routes.enterprise.ConfluenceSyncConnector.from_settings", side_effect=ValueError("nope")):
        assert _app().post("/sync/confluence/wiki/reconcile").status_code == 404
