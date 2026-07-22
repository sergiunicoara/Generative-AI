"""Unit tests for Settings.retrieval_for() — per-tenant retrieval config merge.

The critical property is the *no-op guarantee*: with no override for a tenant,
retrieval_for returns the global retrieval config unchanged, so an empty
tenant_overrides block cannot alter any tenant's behavior. Overrides, when
present, win over the global default for that tenant only.
"""

from __future__ import annotations

from graphrag.core.config import Settings


def _settings_with_yaml(retrieval: dict) -> Settings:
    """A Settings instance whose YAML block is a controlled test fixture.

    Settings.__init__ loads the real settings.yml, so we overwrite _yaml
    afterward (it's set via object.__setattr__ in __init__ for the same reason
    — pydantic models are otherwise frozen to declared fields)."""
    s = Settings()
    object.__setattr__(s, "_yaml", {"retrieval": retrieval})
    return s


class TestRetrievalForNoOp:
    """With no matching override, the global config passes through untouched."""

    def test_no_tenant_overrides_key_returns_base_identity(self):
        base = {"local_top_k": 10, "rerank_top_k": 5}
        s = _settings_with_yaml(base)
        # Same object identity — no copy, guaranteed no mutation.
        assert s.retrieval_for("aerospace") is s.retrieval

    def test_empty_tenant_overrides_is_noop(self):
        s = _settings_with_yaml({"local_top_k": 10, "tenant_overrides": {}})
        result = s.retrieval_for("aerospace")
        assert result == {"local_top_k": 10}          # tenant_overrides stripped
        assert "tenant_overrides" not in result

    def test_tenant_absent_from_overrides_gets_base(self):
        s = _settings_with_yaml({
            "local_top_k": 10,
            "tenant_overrides": {"marketing": {"rerank_top_k": 8}},
        })
        # aerospace has no override → global defaults, tenant_overrides stripped
        assert s.retrieval_for("aerospace") == {"local_top_k": 10}

    def test_default_tenant_gets_base(self):
        s = _settings_with_yaml({"local_top_k": 10, "tenant_overrides": {}})
        assert s.retrieval_for() == {"local_top_k": 10}


class TestRetrievalForMerge:
    """A present override wins over the global default for that tenant only."""

    def test_override_wins_over_global(self):
        s = _settings_with_yaml({
            "local_top_k": 10,
            "rerank_top_k": 5,
            "tenant_overrides": {"aerospace": {"rerank_top_k": 8}},
        })
        result = s.retrieval_for("aerospace")
        assert result["rerank_top_k"] == 8      # tenant value wins
        assert result["local_top_k"] == 10      # non-overridden key preserved
        assert "tenant_overrides" not in result  # never leaks as a knob

    def test_override_adds_new_key(self):
        # A per-tenant-only knob (e.g. a Phase-2 gate defaulting off globally)
        s = _settings_with_yaml({
            "local_top_k": 10,
            "tenant_overrides": {"aerospace": {"authority_rank_weight": 0.3}},
        })
        assert s.retrieval_for("aerospace")["authority_rank_weight"] == 0.3

    def test_isolation_one_tenant_override_does_not_affect_another(self):
        s = _settings_with_yaml({
            "local_top_k": 10,
            "rerank_top_k": 5,
            "tenant_overrides": {"aerospace": {"rerank_top_k": 8}},
        })
        assert s.retrieval_for("aerospace")["rerank_top_k"] == 8
        assert s.retrieval_for("automotive")["rerank_top_k"] == 5   # untouched
        # And the global block itself is not mutated by the merge.
        assert s.retrieval["rerank_top_k"] == 5

    def test_merge_returns_a_copy_not_the_base(self):
        base_ret = {
            "local_top_k": 10,
            "tenant_overrides": {"aerospace": {"local_top_k": 20}},
        }
        s = _settings_with_yaml(base_ret)
        result = s.retrieval_for("aerospace")
        result["local_top_k"] = 99            # mutate the returned dict
        assert s.retrieval["local_top_k"] == 10   # global unaffected
