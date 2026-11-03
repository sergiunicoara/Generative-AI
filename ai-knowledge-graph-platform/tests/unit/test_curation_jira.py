"""Jira payload mapping, env config and REST client (offline via httpx.MockTransport)."""
from __future__ import annotations

import base64
import json
import logging

import httpx
import pytest

from graphrag.curation.jira import (
    JiraClient,
    JiraConfig,
    JiraError,
    proposal_to_jira_issue,
)

TOKEN = "s3cr3t-api-token"
PROPOSAL = {
    "id": "p-1", "kind": "entity_type", "proposed_value": "VENDOR", "entity_name": "Beta",
    "source_type": "A", "target_type": "B", "reason": "unknown_entity_type", "status": "pending",
    "confidence": 0.8, "conflicts": ["relation"], "affected_use_cases": ["uc1", "uc2"],
    "business_impact": "high", "urgency": "high", "effort": "low", "risk": "low",
    "dependencies": ["p-0"], "requesting_team": "ops", "seen_count": 3,
    "created_at": "2026-01-01", "last_seen_at": "2026-01-02",
}
ENV = {
    "JIRA_BASE_URL": "https://example.atlassian.net", "JIRA_EMAIL": "bot@example.com",
    "JIRA_API_TOKEN": TOKEN, "JIRA_PROJECT_KEY": "ONT",
}


def _config(**overrides) -> JiraConfig:
    base = dict(base_url="https://example.atlassian.net", email="bot@example.com",
                api_token=TOKEN, project_key="ONT")
    return JiraConfig(**{**base, **overrides})


def _client(handler, **config) -> JiraClient:
    return JiraClient(_config(**config), client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


class TestPayload:
    def test_targets_project_and_issue_type(self):
        fields = proposal_to_jira_issue(PROPOSAL, project_key="ONT")["fields"]
        assert (fields["project"], fields["issuetype"]) == ({"key": "ONT"}, {"name": "Task"})

    def test_summary_names_kind_and_value(self):
        summary = proposal_to_jira_issue(PROPOSAL, project_key="ONT")["fields"]["summary"]
        assert "entity_type" in summary and "VENDOR" in summary

    def test_summary_is_bounded_and_single_line(self):
        long = {**PROPOSAL, "proposed_value": "line1\nline2 " + "x" * 1000}
        summary = proposal_to_jira_issue(long, project_key="ONT")["fields"]["summary"]
        assert len(summary) <= 255 and "\n" not in summary

    def test_description_carries_review_and_provenance_fields(self):
        description = proposal_to_jira_issue(PROPOSAL, project_key="ONT")["fields"]["description"]
        for needle in ("p-1", "0.8", "relation", "uc1", "ops", "unknown_entity_type", "2026-01-02"):
            assert needle in description

    def test_description_neutralises_wiki_macros(self):
        evil = {**PROPOSAL, "proposed_value": "{code}x{code} [http://evil|x]"}
        description = proposal_to_jira_issue(evil, project_key="ONT")["fields"]["description"]
        assert "{code}" not in description.replace("\\{code\\}", "")

    def test_description_is_bounded(self):
        huge = {**PROPOSAL, "reason": "r" * 100000}
        assert len(proposal_to_jira_issue(huge, project_key="ONT")["fields"]["description"]) <= 32000

    def test_labels_have_no_spaces(self):
        odd = {**PROPOSAL, "kind": "weird kind", "urgency": "very high"}
        labels = proposal_to_jira_issue(odd, project_key="ONT")["fields"]["labels"]
        assert all(" " not in label for label in labels) and "ontology-proposal" in labels

    def test_custom_issue_type(self):
        fields = proposal_to_jira_issue(PROPOSAL, project_key="ONT", issue_type="Story")["fields"]
        assert fields["issuetype"] == {"name": "Story"}

    def test_invalid_project_key_is_rejected(self):
        with pytest.raises(ValueError):
            proposal_to_jira_issue(PROPOSAL, project_key="ont; DROP")


class TestConfig:
    def test_from_env_returns_config_when_all_set(self):
        config = JiraConfig.from_env(ENV)
        assert (config.base_url, config.project_key) == ("https://example.atlassian.net", "ONT")

    @pytest.mark.parametrize("missing", sorted(ENV))
    def test_from_env_is_none_when_any_value_missing(self, missing):
        assert JiraConfig.from_env({k: v for k, v in ENV.items() if k != missing}) is None

    def test_from_env_is_none_for_blank_values(self):
        assert JiraConfig.from_env({**ENV, "JIRA_API_TOKEN": "  "}) is None

    def test_from_env_is_none_for_unsafe_url(self):
        assert JiraConfig.from_env({**ENV, "JIRA_BASE_URL": "file:///etc/passwd"}) is None

    def test_from_env_is_none_for_invalid_project_key(self):
        assert JiraConfig.from_env({**ENV, "JIRA_PROJECT_KEY": "bad key"}) is None

    def test_trailing_slash_is_stripped(self):
        assert JiraConfig.from_env({**ENV, "JIRA_BASE_URL": "https://x.example/jira/"}).base_url == "https://x.example/jira"

    def test_repr_hides_token(self):
        assert TOKEN not in repr(_config())

    def test_unsafe_url_in_constructor_is_rejected(self):
        with pytest.raises(ValueError):
            _config(base_url="gopher://x")


class TestClient:
    async def test_posts_to_v2_issue_endpoint_and_returns_key(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"], seen["method"] = str(request.url), request.method
            return httpx.Response(201, json={"id": "10001", "key": "ONT-7"})

        key = await _client(handler).create_issue({"fields": {}})
        assert (key, seen["method"], seen["url"]) == ("ONT-7", "POST", "https://example.atlassian.net/rest/api/2/issue")

    async def test_sends_basic_auth_header(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(201, json={"key": "ONT-1"})

        await _client(handler).create_issue({"fields": {}})
        assert seen["auth"] == "Basic " + base64.b64encode(f"bot@example.com:{TOKEN}".encode()).decode()

    async def test_sends_payload_as_json(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(201, json={"key": "ONT-1"})

        await _client(handler).create_issue({"fields": {"summary": "s"}})
        assert seen["body"] == {"fields": {"summary": "s"}}

    async def test_http_error_raises_structured_error_without_token(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"errorMessages": ["unauthorized"]})

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert (err.value.code, err.value.status, TOKEN in str(err.value)) == ("jira_request_failed", 401, False)

    async def test_error_response_echoing_credentials_is_not_leaked(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text=f"bad request for {TOKEN}")

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert TOKEN not in err.value.message

    async def test_timeout_maps_to_jira_timeout(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert err.value.code == "jira_timeout"

    async def test_transport_error_maps_to_unreachable_without_token(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot connect with {TOKEN}", request=request)

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert (err.value.code, TOKEN in err.value.message) == ("jira_unreachable", False)

    async def test_timeout_is_not_retried(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            raise httpx.ReadTimeout("slow", request=request)

        with pytest.raises(JiraError):
            await _client(handler).create_issue({"fields": {}})
        assert len(calls) == 1

    async def test_server_error_is_not_retried(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503)

        with pytest.raises(JiraError):
            await _client(handler).create_issue({"fields": {}})
        assert len(calls) == 1

    async def test_response_without_key_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, json={"id": "1"})

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert err.value.code == "jira_bad_response"

    async def test_non_json_response_is_an_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(201, text="<html>")

        with pytest.raises(JiraError) as err:
            await _client(handler).create_issue({"fields": {}})
        assert err.value.code == "jira_bad_response"

    async def test_redirects_are_not_followed_with_credentials(self):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(302, headers={"location": "https://evil.example/steal"})

        with pytest.raises(JiraError):
            await _client(handler).create_issue({"fields": {}})
        assert calls == ["https://example.atlassian.net/rest/api/2/issue"]

    async def test_token_never_appears_in_logs(self, caplog):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        with caplog.at_level(logging.DEBUG), pytest.raises(JiraError):
            await _client(handler).create_issue({"fields": {}})
        assert TOKEN not in caplog.text
