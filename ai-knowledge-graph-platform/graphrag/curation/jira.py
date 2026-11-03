"""Optional Jira ticketing for ontology proposals (REST API v2, create-issue only).

Disabled unless ``JIRA_BASE_URL``, ``JIRA_EMAIL``, ``JIRA_API_TOKEN`` and
``JIRA_PROJECT_KEY`` are all set (``JiraConfig.from_env`` returns ``None``).
The API token is excluded from ``repr``, is never logged, and is scrubbed from
any error text.  Creating an issue is not idempotent, so the client never
retries a request (a retry after a timeout could create a duplicate ticket).
Redirects are not followed, so credentials cannot be forwarded to another host.

Not validated against a live Jira instance; behaviour is covered by
``httpx.MockTransport`` tests against the documented v2 contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import os
import re
from typing import Any

import httpx
import structlog

from graphrag.core.connector_url_safety import UnsafeConnectorURLError, assert_safe_connector_url

log = structlog.get_logger(__name__)

ENV_BASE_URL, ENV_EMAIL = "JIRA_BASE_URL", "JIRA_EMAIL"
ENV_API_TOKEN, ENV_PROJECT_KEY = "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"
_ISSUE_PATH = "/rest/api/2/issue"
_PROJECT_KEY = re.compile(r"^[A-Z][A-Z0-9_]{1,19}$")
_SUMMARY_MAX = 255
_DESCRIPTION_MAX = 30000  # Jira's limit is 32767
_VALUE_MAX = 1000
_ERROR_SNIPPET_MAX = 300
_DEFAULT_TIMEOUT_SECONDS = 15.0
_WIKI_SPECIALS = re.compile(r"([\\{}\[\]|])")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")
_LABEL_BASE = "ontology-proposal"

# (label, proposal key) shown in the description, in this order.
_DESCRIPTION_FIELDS = (
    ("Proposal ID", "id"), ("Kind", "kind"), ("Proposed value", "proposed_value"),
    ("Entity name", "entity_name"), ("Source type", "source_type"), ("Target type", "target_type"),
    ("Reason", "reason"), ("Status", "status"), ("Confidence", "confidence"),
    ("Conflicts", "conflicts"), ("Affected use cases", "affected_use_cases"),
    ("Business impact", "business_impact"), ("Urgency", "urgency"), ("Effort", "effort"),
    ("Risk", "risk"), ("Dependencies", "dependencies"), ("Requesting team", "requesting_team"),
    ("Times seen", "seen_count"), ("First seen", "created_at"), ("Last seen", "last_seen_at"),
)


class JiraError(Exception):
    """Structured Jira failure.  ``message`` never contains credentials."""

    def __init__(self, code: str, message: str, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def _validate_project_key(project_key: str) -> str:
    if not _PROJECT_KEY.match(project_key or ""):
        raise ValueError("Jira project key must be uppercase letters, digits or underscores")
    return project_key


@dataclass(frozen=True)
class JiraConfig:
    base_url: str
    email: str
    api_token: str = field(repr=False)
    project_key: str
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", self.base_url.strip().rstrip("/"))
        if not self.base_url:
            raise ValueError("Jira base URL is required")
        try:
            assert_safe_connector_url(self.base_url, context="JiraConfig base_url")
        except UnsafeConnectorURLError as exc:
            raise ValueError(str(exc)) from exc
        _validate_project_key(self.project_key)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "JiraConfig | None":
        """Config from the environment, or ``None`` when Jira is not (validly) configured."""
        source = os.environ if env is None else env
        values = {
            name: (source.get(name) or "").strip()
            for name in (ENV_BASE_URL, ENV_EMAIL, ENV_API_TOKEN, ENV_PROJECT_KEY)
        }
        if not all(values.values()):
            return None
        try:
            return cls(base_url=values[ENV_BASE_URL], email=values[ENV_EMAIL],
                       api_token=values[ENV_API_TOKEN], project_key=values[ENV_PROJECT_KEY])
        except ValueError as exc:
            log.warning("jira.config_invalid", reason=str(exc))
            return None


# -- payload ------------------------------------------------------------------

def _wiki_text(value: Any, limit: int = _VALUE_MAX) -> str:
    """Single-line text with Jira wiki-markup control characters escaped."""
    if isinstance(value, (list, tuple, set, frozenset)):
        value = "; ".join(str(item) for item in value)
    text = _CONTROL.sub(" ", "" if value is None else str(value)).strip()
    return _WIKI_SPECIALS.sub(r"\\\1", text[:limit])


def _label(prefix: str, value: Any) -> str | None:
    cleaned = _LABEL_UNSAFE.sub("_", str(value or "")).strip("_")[:50]
    return f"{prefix}-{cleaned}" if cleaned else None


def proposal_to_jira_issue(proposal: dict, *, project_key: str, issue_type: str = "Task") -> dict:
    """Jira REST v2 create-issue payload for a proposal.  Pure; performs no I/O."""
    _validate_project_key(project_key)
    summary = _CONTROL.sub(" ", f"[Ontology {proposal.get('kind', '')}] {proposal.get('proposed_value', '')}")
    lines = ["h3. Ontology change proposal awaiting curator review"]
    for label, key in _DESCRIPTION_FIELDS:
        text = _wiki_text(proposal.get(key))
        if text:
            lines.append(f"* *{label}:* {text}")
    labels = [_LABEL_BASE, _label("kind", proposal.get("kind")), _label("urgency", proposal.get("urgency"))]
    return {
        "fields": {
            "project": {"key": project_key},
            "issuetype": {"name": issue_type},
            "summary": summary.strip()[:_SUMMARY_MAX],
            "description": "\n".join(lines)[:_DESCRIPTION_MAX],
            "labels": [label for label in labels if label],
        }
    }


# -- client -------------------------------------------------------------------

class JiraClient:
    """Create-issue client.  ``client`` is injectable (tests use ``httpx.MockTransport``)."""

    def __init__(self, config: JiraConfig, client: httpx.AsyncClient | None = None):
        self._config = config
        self._client = client

    @property
    def project_key(self) -> str:
        return self._config.project_key

    def _scrub(self, text: str) -> str:
        cfg = self._config
        for secret in (cfg.api_token, cfg.email):
            text = text.replace(secret, "[redacted]")
        return text

    async def create_issue(self, payload: dict) -> str:
        """POST the payload once and return the new issue key; raise ``JiraError`` otherwise."""
        url = self._config.base_url + _ISSUE_PATH
        owned = self._client is None
        client = self._client or httpx.AsyncClient(follow_redirects=False)
        try:
            response = await client.post(
                url, json=payload, auth=(self._config.email, self._config.api_token),
                headers={"Accept": "application/json"}, timeout=self._config.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            log.warning("jira.timeout")
            raise JiraError("jira_timeout", "Jira did not respond in time; the ticket may or may not exist") from exc
        except httpx.HTTPError as exc:
            log.warning("jira.unreachable", error_type=type(exc).__name__)
            raise JiraError("jira_unreachable", f"could not reach Jira ({type(exc).__name__})") from exc
        finally:
            if owned:
                await client.aclose()
        return self._issue_key(response)

    def _issue_key(self, response: httpx.Response) -> str:
        if not response.is_success:
            snippet = self._scrub(response.text[:_ERROR_SNIPPET_MAX])
            log.warning("jira.request_failed", status=response.status_code)
            raise JiraError("jira_request_failed", f"Jira returned HTTP {response.status_code}: {snippet}".rstrip(": "),
                            status=response.status_code)
        try:
            key = response.json()["key"]
        except (ValueError, KeyError, TypeError) as exc:
            raise JiraError("jira_bad_response", "Jira response did not contain an issue key") from exc
        if not isinstance(key, str) or not key:
            raise JiraError("jira_bad_response", "Jira response did not contain an issue key")
        log.info("jira.issue_created", issue_key=key)
        return key
