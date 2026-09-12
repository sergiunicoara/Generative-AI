"""Scheme/format validation for connector-configured URLs.

Closes a gap a follow-up platform review named explicitly: no connector URL
anywhere in this codebase was validated at all -- ``TripleStoreTarget.base_url``
(``graphrag/graph/triplestore.py``), ``RESTAPISourceConfig.base_url``/
``token_url`` (``graphrag/ingestion/http_source.py``), and
``SourceSystem.uri`` (the one tenant-facing write surface for a URL, via
``POST /sources``) all accepted arbitrary strings, including dangerous
schemes such as ``file://``, ``gopher://`` or ``javascript:``.

What this deliberately does NOT do
-----------------------------------
This is a scheme/format allow-list, not a private-IP/SSRF network blocklist.
``TripleStoreTarget`` and ``RESTAPISourceConnector`` are operator-configured
today (see ``graphrag/graph/triplestore.py``'s module docstring and
``graphrag/ingestion/http_source.py``'s), never accepted from a tenant HTTP
request body, and this repo's own demo deployment legitimately targets
loopback (``compose.energy-demo.yaml`` binds GraphDB to
``127.0.0.1:7200``). Blocking private/loopback IPs here would break that
documented, intended deployment shape for a threat -- an attacker
controlling this configuration -- that doesn't exist at this boundary. If a
connector URL is ever accepted directly from an untrusted request in the
future, that call site needs its own SSRF review at that time; don't assume
this guard covers it.

What this DOES do: reject a URL that cannot possibly be a legitimate HTTP(S)
API/triplestore endpoint -- an unparseable string, a non-http(s) scheme, or a
missing host -- before it's used to build a request. That is real defense in
depth against a mistyped or maliciously substituted connector URL (e.g. an
operator pasting a ``file://`` path from a different tool, or a tenant-facing
form accepting ``javascript:`` for a field meant to hold an API base URL).
"""

from __future__ import annotations

from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class UnsafeConnectorURLError(ValueError):
    """Raised when a connector-configured URL cannot be a valid HTTP(S) endpoint."""


def assert_safe_connector_url(url: str, *, context: str) -> None:
    """Raise ``UnsafeConnectorURLError`` unless ``url`` is a well-formed http(s) URL.

    A no-op for a falsy ``url`` -- several call sites (``SourceSystem.uri``,
    for one) treat an empty string as "not configured", and rejecting that
    would turn an optional field into a required one.

    ``context`` is a short, caller-supplied label (e.g. ``"TripleStoreTarget
    base_url"``) so the raised error identifies which configured URL failed,
    not just that URL validation failed somewhere.
    """
    if not url:
        return
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise UnsafeConnectorURLError(f"{context}: {url!r} is not a parseable URL") from exc

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeConnectorURLError(
            f"{context}: {url!r} uses scheme {parts.scheme!r}; "
            f"only {sorted(_ALLOWED_SCHEMES)} are permitted"
        )
    if not parts.netloc or not parts.hostname:
        raise UnsafeConnectorURLError(f"{context}: {url!r} has no host")
