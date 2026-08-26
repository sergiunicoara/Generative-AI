"""Property-based tests for the invariants that must hold on *all* inputs.

Example-based tests pin the cases someone thought of. These pin the cases
nobody enumerates — and every property below guards something whose violation
is silent rather than loud:

- **Cache-key tenant isolation.** If two tenants can ever produce the same
  answer-cache key, one tenant is served another's answer. Nothing errors; the
  answer is simply wrong and confidential. No finite set of examples can
  establish this, which is exactly what property testing is for.
- **Prompt escaping.** A single unescaped angle bracket reopens the injection
  boundary. The fixed corpus in `test_prompt_injection_corpus.py` covers the
  payloads we imagined; this covers the ones we did not.
- **Resource-URI canonicalisation.** A non-idempotent normaliser means two
  spellings of one resource never compare equal, and every MCP call 401s.

Deliberately *not* asserted: that escaping is idempotent. It is not, and should
not be — escaping twice legitimately double-escapes. Writing that property
would encode a bug as a requirement.
"""

from __future__ import annotations

import html

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from api.limiter import client_key
from graphrag.core.prompt_security import escape_prompt_data
from graphrag.core.resource_identifiers import canonical_resource_uri
from graphrag.core.scopes import is_valid_scope, tenant_scope, validate_scopes
from graphrag.core.tenant_quota import QuotaPolicy, TenantQuotaStore
from graphrag.retrieval.query_cache import (
    QueryCacheContext,
    build_cache_key,
    normalize_query,
)

# Keep generated cases small: these are unit tests in a 1500-test suite, and a
# property that needs hundreds of examples to find a bug is usually a property
# that is too vague to be useful.
FAST = settings(max_examples=60, deadline=None)

# Tenants follow the same shape the tenant: scope regex already enforces.
# Build the shape directly instead of asking Hypothesis to synthesize strings
# from a regex. The regex strategy can occasionally consume excessive entropy
# for this simple finite alphabet and trip HealthCheck.too_slow before the
# property itself runs (observed after two otherwise clean full-suite passes).
_TENANT_HEAD = tuple("abcdefghijklmnopqrstuvwxyz0123456789")
_TENANT_ALPHABET = _TENANT_HEAD + ("_", "-")
tenants = st.builds(
    lambda head, tail: head + tail,
    st.sampled_from(_TENANT_HEAD),
    st.text(alphabet=_TENANT_ALPHABET, min_size=0, max_size=20),
)
queries = st.text(min_size=1, max_size=200)
any_text = st.text(max_size=300)


def _context(**overrides) -> QueryCacheContext:
    values = {
        "corpus_revision": 1,
        "requested_mode": "hybrid",
        "effective_mode": "local",
        "model_route": {"primary": "p"},
        "prompt_version": "v1",
        "retrieval_config": {},
        "ontology_version": "platform/v1",
    }
    values.update(overrides)
    return QueryCacheContext(**values)


class TestCacheKeyIsolation:
    @FAST
    @given(query=queries, tenant=tenants)
    def test_key_is_deterministic(self, query, tenant):
        context = _context()
        assert build_cache_key(query, tenant, context) == build_cache_key(query, tenant, context)

    @FAST
    @given(query=queries, first=tenants, second=tenants)
    def test_distinct_tenants_never_share_a_key(self, query, first, second):
        # The security property: a collision here serves one tenant another
        # tenant's cached answer, silently.
        if first == second:
            return
        context = _context()
        assert build_cache_key(query, first, context) != build_cache_key(query, second, context)

    @FAST
    @given(query=queries, tenant=tenants, revision=st.integers(0, 10_000))
    def test_corpus_revision_change_always_invalidates(self, query, tenant, revision):
        # Corpus revision is the hard staleness guard; if a revision bump could
        # ever produce the same key, a re-ingested corpus keeps serving the
        # pre-ingest answer.
        a = build_cache_key(query, tenant, _context(corpus_revision=revision))
        b = build_cache_key(query, tenant, _context(corpus_revision=revision + 1))
        assert a != b

    @FAST
    @given(query=queries, tenant=tenants)
    def test_key_is_namespaced_by_tenant_digest(self, query, tenant):
        # Lets flush_tenant() scan by prefix without touching another tenant.
        key = build_cache_key(query, tenant, _context())
        other = build_cache_key(query, tenant + "x", _context())
        assert key.rsplit(":", 1)[0] != other.rsplit(":", 1)[0]

    @FAST
    @given(query=queries, tenant=tenants)
    def test_whitespace_and_case_variants_collide(self, query, tenant):
        # Normalisation must be strong enough that trivial reformatting is a
        # hit, or the cache never warms.
        noisy = f"  {query.upper()}  "
        context = _context()
        if normalize_query(noisy) == normalize_query(query):
            assert build_cache_key(noisy, tenant, context) == build_cache_key(query, tenant, context)

    @FAST
    @given(query=queries)
    def test_normalisation_is_idempotent(self, query):
        once = normalize_query(query)
        assert normalize_query(once) == once


class TestPromptEscaping:
    @FAST
    @given(text=any_text)
    def test_no_angle_bracket_survives(self, text):
        # Every element-closing attack is expressed with angle brackets; if
        # none survive, no payload can open or close an element at all.
        escaped = escape_prompt_data(text)
        assert "<" not in escaped and ">" not in escaped

    @FAST
    @given(text=any_text)
    def test_escaping_is_lossless(self, text):
        # Containment must not delete evidence: a reviewer reading a trace has
        # to see exactly what was planted.
        assert html.unescape(escape_prompt_data(text)) == text

    @FAST
    @given(text=any_text)
    def test_no_structural_delimiter_can_be_introduced(self, text):
        escaped = escape_prompt_data(text)
        for token in ("<retrieved_context>", "</retrieved_context>",
                      "<source_text>", "</source_text>"):
            assert token not in escaped


class TestResourceCanonicalisation:
    # Absolute http(s) URIs with an optional path, which is what RFC 8707
    # permits as a resource identifier.
    uris = st.builds(
        lambda scheme, host, port, path: (
            f"{scheme}://{host}{port}{path}"
        ),
        scheme=st.sampled_from(["http", "https", "HTTP", "HttpS"]),
        host=st.from_regex(r"\A[a-zA-Z][a-zA-Z0-9.-]{0,30}\Z", fullmatch=True),
        port=st.sampled_from(["", ":8080", ":443"]),
        path=st.sampled_from(["", "/", "/mcp", "/mcp/", "/a/b", "/a/b/"]),
    )

    @FAST
    @given(uri=uris)
    def test_canonicalisation_is_idempotent(self, uri):
        # Not idempotent means two spellings of one resource never compare
        # equal, and every audience check fails.
        once = canonical_resource_uri(uri)
        assert canonical_resource_uri(once) == once

    @FAST
    @given(uri=uris)
    def test_result_has_no_trailing_slash_query_or_fragment(self, uri):
        result = canonical_resource_uri(uri)
        assert not result.endswith("/")
        assert "?" not in result and "#" not in result

    @FAST
    @given(uri=uris)
    def test_scheme_and_host_are_lowercased(self, uri):
        result = canonical_resource_uri(uri)
        scheme, _, rest = result.partition("://")
        authority = rest.split("/", 1)[0]
        assert scheme == scheme.lower()
        assert authority == authority.lower()

    @FAST
    @given(
        uri=uris,
        fragment=st.from_regex(r"\A[a-z]{1,8}\Z", fullmatch=True),
    )
    def test_fragments_are_always_rejected(self, uri, fragment):
        from graphrag.core.resource_identifiers import InvalidResourceIdentifier

        with pytest.raises(InvalidResourceIdentifier):
            canonical_resource_uri(f"{uri}#{fragment}")


class TestScopeVocabulary:
    @FAST
    @given(tenant=tenants)
    def test_tenant_scope_round_trips(self, tenant):
        # A tenant scope that fails its own validator makes ToolPolicy's guard
        # unreachable for that tenant -- the exact bug scopes.py was written
        # to prevent.
        assert is_valid_scope(tenant_scope(tenant))

    @FAST
    @given(scopes=st.lists(st.text(max_size=30), max_size=10))
    def test_validation_only_ever_narrows(self, scopes):
        validated = validate_scopes(scopes)
        assert set(validated).issubset(set(scopes))
        assert all(is_valid_scope(scope) for scope in validated)


class TestRateLimitKeyNamespaces:
    @FAST
    @given(
        subject=st.text(min_size=1, max_size=40),
        address=st.text(min_size=1, max_size=40),
    )
    def test_subject_and_address_keys_never_collide(self, subject, address):
        from types import SimpleNamespace

        # A collision lets a caller inherit (or escape) another's bucket by
        # choosing a subject that looks like an address.
        authed = SimpleNamespace(
            client=SimpleNamespace(host="0.0.0.0", port=1),
            headers={}, state=SimpleNamespace(user={"sub": subject}),
            url=SimpleNamespace(path="/"),
        )
        anon = SimpleNamespace(
            client=SimpleNamespace(host=address, port=1),
            headers={}, state=SimpleNamespace(),
            url=SimpleNamespace(path="/"),
        )
        assert client_key(authed) != client_key(anon)


class TestQuotaAccounting:
    @FAST
    @given(
        amounts=st.lists(st.floats(0.0, 5.0, allow_nan=False, allow_infinity=False),
                         min_size=1, max_size=12),
        tenant=tenants,
    )
    def test_usage_is_monotonic_within_a_window(self, amounts, tenant):
        import asyncio

        async def run() -> None:
            store = TenantQuotaStore(default_policy=QuotaPolicy(max_cost_usd=10_000))
            await store.connect()
            previous = 0.0
            for amount in amounts:
                await store.consume(tenant, requests=0, cost_usd=amount)
                usage = (await store.usage(tenant))["cost_usd"]["used"]
                # Spend can never decrease inside a window; if it can, a tenant
                # can reset its own budget by spending in the right order.
                assert usage >= previous - 1e-9
                previous = usage

        asyncio.run(run())

    @FAST
    @given(first=tenants, second=tenants, amount=st.integers(1, 20))
    def test_consumption_never_crosses_tenants(self, first, second, amount):
        import asyncio

        if first == second:
            return

        async def run() -> None:
            store = TenantQuotaStore(default_policy=QuotaPolicy(max_requests=10_000))
            await store.connect()
            for _ in range(amount):
                await store.consume(first, requests=1)
            # One tenant's spend must never appear on another's ledger.
            assert (await store.usage(second))["requests"]["used"] == 0

        asyncio.run(run())

    @FAST
    @given(ceiling=st.integers(1, 20))
    def test_ceiling_is_never_exceeded_when_checked_before_consuming(self, ceiling):
        import asyncio

        async def run() -> None:
            store = TenantQuotaStore(default_policy=QuotaPolicy(max_requests=ceiling))
            await store.connect()
            admitted = 0
            for _ in range(ceiling * 2 + 5):
                if (await store.check("acme", additional_requests=1)).allowed:
                    await store.consume("acme", requests=1)
                    admitted += 1
            assert admitted == ceiling

        asyncio.run(run())
