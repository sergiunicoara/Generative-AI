"""Golden-file compatibility test for the MCP capability registry.

Diffs the live registry's `contract_snapshot()` against a committed JSON
fixture (`tests/unit/contracts/mcp_capabilities_v1.json`). A capability's
shape (id, version, scopes, arg schema keys, aliases, ...) is part of the
platform's external contract with any MCP client that has already
registered it -- this test's whole purpose is to make an accidental
breaking change fail loudly, with an explicit diff, rather than silently
ship. Regenerate the fixture deliberately when a change is intentional
(see the module docstring below for how); never "fix" a failure here by
regenerating without reading why it changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp_server.capabilities import build_registry
from mcp_server.identity import CallerIdentity

FIXTURE_PATH = Path(__file__).parent / "contracts" / "mcp_capabilities_v1.json"


def _load_golden() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class TestContractSnapshotMatchesGoldenFile:
    def test_snapshot_matches_committed_fixture(self):
        golden = _load_golden()
        live = build_registry().contract_snapshot()
        golden_by_name = {c["qualified_name"]: c for c in golden}
        live_by_name = {c["qualified_name"]: c for c in live}

        removed = golden_by_name.keys() - live_by_name.keys()
        added = live_by_name.keys() - golden_by_name.keys()
        assert not removed, (
            f"capabilities removed from the registry (breaking change): {sorted(removed)}"
        )
        assert not added, (
            f"new capabilities not yet committed to the golden fixture: {sorted(added)} "
            "-- regenerate tests/unit/contracts/mcp_capabilities_v1.json"
        )
        for name, golden_entry in golden_by_name.items():
            assert live_by_name[name] == golden_entry, (
                f"capability {name!r} changed shape -- if intentional, "
                "regenerate tests/unit/contracts/mcp_capabilities_v1.json"
            )


class TestBackwardCompatibleReadCapabilities:
    """graph_stats is the compatibility-adapter proof case (Wave 4): same
    wire name and signature, with read entitlement required for resolution."""

    def test_graph_stats_resolvable_under_legacy_bare_name(self):
        registry = build_registry()
        spec = registry.resolve("graph_stats", CallerIdentity.from_claims(
            {"sub": "reader", "tenant": "aerospace", "scope": "read"},
        ))
        assert spec.qualified_name == "kg.graph.stats@1.0.0"
        assert spec.arg_schema.keys() == {"tenant"}

    def test_query_graph_facts_resolvable_under_qualified_name(self):
        registry = build_registry()
        spec = registry.resolve("kg.facts.query@1.0.0", CallerIdentity.from_claims(
            {"sub": "reader", "tenant": "aerospace", "scope": "read"},
        ))
        assert spec.required_scopes == ("read",)

    def test_every_read_capability_requires_read_scope(self):
        registry = build_registry()
        no_read = CallerIdentity.from_claims(
            {"sub": "worker", "tenant": "aerospace", "scope": "biz:write"},
        )
        for spec in registry.contract_snapshot():
            if spec["kind"] == "read":
                assert "read" in spec["required_scopes"]
                assert registry.resolve(spec["qualified_name"], no_read).reason == "missing_scope"


class TestWriteCapabilityRequiresBizWriteScope:
    def test_anonymous_caller_denied_missing_scope(self):
        registry = build_registry()
        result = registry.resolve("biz.workorder.create@1.0.0", CallerIdentity.anonymous())
        assert result.reason == "missing_scope"

    def test_caller_with_biz_write_can_resolve_it(self):
        registry = build_registry()
        identity = CallerIdentity(
            subject="agent-1", tenant="aerospace",
            scopes=frozenset({"biz:write"}), authenticated=True,
        )
        spec = registry.resolve("biz.workorder.create@1.0.0", identity)
        assert spec.required_scopes == ("biz:write",)
        assert spec.kind == "write"
        assert spec.requires_approval is True
