"""Tests for scripts/load_blazegraph.py after it became a thin wrapper over
graphrag.graph.triplestore.TripleStoreTarget -- the public load() function's
signature and behaviour must be unchanged for existing callers, even though
it became async internally (offloading to a real remote store)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import load_blazegraph  # noqa: E402


class TestLoadSignatureIsUnchanged:
    def test_parameter_names_and_defaults_match_the_pre_wrapper_contract(self):
        sig = inspect.signature(load_blazegraph.load)
        assert list(sig.parameters) == [
            "ttl_path", "endpoint", "namespace", "timeout", "context_path",
        ]
        assert sig.parameters["endpoint"].default == load_blazegraph.DEFAULT_ENDPOINT
        assert sig.parameters["namespace"].default == "kb"
        assert sig.parameters["context_path"].default == load_blazegraph.DEFAULT_CONTEXT_PATH

    def test_load_is_now_a_coroutine_function(self):
        # main() must therefore run it via asyncio.run -- this pins that the
        # conversion was deliberate, not an accidental signature drift.
        assert inspect.iscoroutinefunction(load_blazegraph.load)


class TestLoadDelegatesToTripleStoreTarget:
    @pytest.mark.asyncio
    async def test_missing_file_raises_before_any_network_call(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            await load_blazegraph.load(tmp_path / "nonexistent.ttl")

    @pytest.mark.asyncio
    async def test_posts_the_file_bytes_via_triplestoretarget(self, tmp_path):
        ttl = tmp_path / "export.ttl"
        ttl.write_bytes(b"<a> <b> <c> .")

        mock_target = AsyncMock()
        mock_target.load = AsyncMock(return_value=200)
        mock_target.load_url = "http://localhost:9999/bigdata/namespace/kb/sparql"

        with patch("load_blazegraph.TripleStoreTarget", return_value=mock_target) as mock_cls:
            status = await load_blazegraph.load(ttl, namespace="kb")

        assert status == 200
        mock_cls.assert_called_once_with(
            "blazegraph", load_blazegraph.DEFAULT_ENDPOINT, timeout=60.0,
            namespace="kb", context_path=load_blazegraph.DEFAULT_CONTEXT_PATH,
        )
        mock_target.load.assert_awaited_once_with(b"<a> <b> <c> .")
