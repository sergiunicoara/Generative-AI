"""Every ported src/graph/*.py module must import cleanly after the graphrag.* -> src.*
import rewrite (Increment 1 / Phase -1). This is the unblocking test: nothing else in
this suite (or any later phase) is meaningful if these modules can't even be imported.
"""

import importlib

import pytest

LEGACY_MODULES = [
    "src.graph.alias_registry",
    "src.graph.bitemporal",
    "src.graph.contradiction_detector",
    "src.graph.contradiction_strategies",
    "src.graph.domain_ontology",
    "src.graph.ontology_registry",
    "src.graph.review_queue",
    "src.graph.reification",
]


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_legacy_module_imports_cleanly(module_name):
    module = importlib.import_module(module_name)
    assert module is not None


def test_forked_core_modules_import_cleanly():
    import src.core.config  # noqa: F401
    import src.core.neo4j_client  # noqa: F401
    import src.core.retry  # noqa: F401
    import src.graph.inference_engine  # noqa: F401
    import src.graph.ontology_migration  # noqa: F401
