"""Unit coverage for graphrag/core/connector_url_safety.py.

This is a scheme/format allow-list (http/https, well-formed host), not an
SSRF/private-IP blocklist -- see the module docstring for why loopback and
private addresses are deliberately still permitted (this repo's own demo
deployment legitimately targets 127.0.0.1). These tests pin exactly that
boundary: dangerous schemes and malformed URLs are rejected; legitimate
loopback/private http(s) endpoints are not.
"""

from __future__ import annotations

import pytest

from graphrag.core.connector_url_safety import (
    UnsafeConnectorURLError,
    assert_safe_connector_url,
)


class TestRejectedSchemes:
    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/plain;base64,SGVsbG8=",
            "gopher://example.com/",
            "ftp://example.com/",
            "ws://example.com/socket",
        ],
    )
    def test_dangerous_or_non_http_scheme_is_rejected(self, url: str) -> None:
        with pytest.raises(UnsafeConnectorURLError):
            assert_safe_connector_url(url, context="test")

    def test_scheme_relative_url_with_no_scheme_is_rejected(self) -> None:
        with pytest.raises(UnsafeConnectorURLError):
            assert_safe_connector_url("//example.com/path", context="test")


class TestMalformedURLs:
    def test_url_with_no_host_is_rejected(self) -> None:
        with pytest.raises(UnsafeConnectorURLError):
            assert_safe_connector_url("http:///no-host-here", context="test")

    def test_bare_path_with_no_scheme_is_rejected(self) -> None:
        with pytest.raises(UnsafeConnectorURLError):
            assert_safe_connector_url("not-a-url-at-all", context="test")


class TestLegitimateURLsAreNotBlocked:
    """The scope boundary: this guard must not break this repo's own
    documented local deployment shape."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.example.com/v1",
            "http://127.0.0.1:7200",
            "http://localhost:8000/sparql",
            "https://10.0.5.12:8443/api",  # private IP: deliberately allowed
        ],
    )
    def test_wellformed_http_or_https_url_is_accepted(self, url: str) -> None:
        assert_safe_connector_url(url, context="test")  # must not raise


class TestEmptyURLIsANoOp:
    def test_empty_string_does_not_raise(self) -> None:
        """Several call sites (SourceSystem.uri, for one) treat "" as "not
        configured" -- rejecting that would turn an optional field required."""
        assert_safe_connector_url("", context="test")


class TestErrorMessageNamesTheContext:
    def test_context_label_appears_in_the_error(self) -> None:
        with pytest.raises(UnsafeConnectorURLError, match="my-caller-label"):
            assert_safe_connector_url("file:///x", context="my-caller-label")
