"""Remote MCP transport is authenticated per request and deny-by-default."""

from __future__ import annotations

import json

import httpx
import pytest

from api.main import app
from src.core.config import get_settings
from src.sales.adapter import LocalCRMEmulator

pytestmark = pytest.mark.asyncio


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer mcp-test-key", "X-Workspace-Id": "ws-mcp-test"}


@pytest.fixture(autouse=True)
def _offline_mcp_environment(monkeypatch):
    """MCP route tests exercise auth/transport, not a live Redis limiter."""
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_mcp_requires_bearer_authentication():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", json={"method": "tools/list"})
    assert response.status_code == 401


async def test_mcp_is_disabled_unless_explicitly_enabled(monkeypatch):
    monkeypatch.setenv("WORKSPACE_API_KEYS", json.dumps({"ws-mcp-test": "mcp-test-key"}))
    monkeypatch.setenv("MCP_ENABLED", "false")
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", headers=_headers(), json={"method": "tools/list"})
    assert response.status_code == 503


async def test_mcp_lists_only_entitled_non_write_tools(monkeypatch):
    monkeypatch.setenv("WORKSPACE_API_KEYS", json.dumps({"ws-mcp-test": "mcp-test-key"}))
    monkeypatch.setenv("MCP_ENABLED", "true")
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", headers=_headers(), json={"method": "tools/list"})
    assert response.status_code == 200
    assert response.json()["tools"]
    assert not any(item["scope"] == "sales:write" for item in response.json()["tools"])


async def test_mcp_write_tool_call_returns_verified_receipt(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKSPACE_API_KEYS", json.dumps({"ws-mcp-test": "mcp-test-key"}))
    monkeypatch.setenv("MCP_ENABLED", "true")
    crm_path = tmp_path / "local-crm.json"
    monkeypatch.setenv("LOCAL_CRM_EMULATOR_PATH", str(crm_path))
    get_settings.cache_clear()
    LocalCRMEmulator(storage_path=crm_path).seed(
        workspace_id="ws-mcp-test", object_id="opp-1", values={"summary": "Initial call notes"},
    )
    body = {
        "method": "tools/call",
        "params": {
            "name": "sales.opportunity.update",
            "arguments": {
                "command_id": "cmd-1", "capability": "sales.opportunity.update", "object_id": "opp-1",
                "patch": {"summary": "Updated after demo call"},
                "expected_version": 1, "correlation_id": "corr-1",
            },
        },
    }
    headers = {**_headers(), "X-User-Roles": "manager"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", headers=headers, json=body)
    assert response.status_code == 200
    receipt = response.json()["receipt"]
    assert receipt["outcome"] == "EXECUTED"
    assert receipt["verified"] is True


async def test_mcp_rejects_oversized_request(monkeypatch):
    monkeypatch.setenv("WORKSPACE_API_KEYS", json.dumps({"ws-mcp-test": "mcp-test-key"}))
    monkeypatch.setenv("MCP_ENABLED", "true")
    monkeypatch.setenv("MCP_REQUEST_MAX_BYTES", "16")
    get_settings.cache_clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/mcp", headers=_headers(), content=b'{"method":"tools/list"}')
    assert response.status_code == 413
