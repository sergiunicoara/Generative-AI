from __future__ import annotations

import importlib
import os
import yaml
from typing import Callable, Dict, List, Optional

from engines.base import BaseEngine


def _load_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _load_engine_config() -> Dict[str, dict]:
    """Read engine kwargs from config.yml so every engine gets the same params."""
    for path in ("config.yml", "config.yaml"):
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f).get("engines", {})
    return {}


_ENGINE_CONFIG: Dict[str, dict] = _load_engine_config()


def _make_factory(module_path: str, class_name: str, engine_name: str) -> Callable[[], BaseEngine]:
    """Return a zero-arg factory that passes config kwargs to the engine constructor."""
    kwargs = _ENGINE_CONFIG.get(engine_name, {})
    return lambda: _load_class(module_path, class_name)(**kwargs)


_ENGINE_FACTORIES: Dict[str, Callable[[], BaseEngine]] = {
    "qdrant":         _make_factory("engines.qdrant_engine",         "QdrantEngine",        "qdrant"),
    "elasticsearch":  _make_factory("engines.elasticsearch_engine",  "ElasticsearchEngine", "elasticsearch"),
    "pgvector":       _make_factory("engines.pgvector_engine",       "PgVectorEngine",      "pgvector"),
    "redis":          _make_factory("engines.redis_engine",          "RedisEngine",         "redis"),
}

# Registry used by get_engine_factory — maps name → (module, class)
_ENGINE_SPECS: Dict[str, tuple] = {
    "qdrant":        ("engines.qdrant_engine",        "QdrantEngine"),
    "elasticsearch": ("engines.elasticsearch_engine", "ElasticsearchEngine"),
    "pgvector":      ("engines.pgvector_engine",      "PgVectorEngine"),
    "redis":         ("engines.redis_engine",         "RedisEngine"),
}


def _try_create_engine(name: str) -> Optional[BaseEngine]:
    factory = _ENGINE_FACTORIES.get(name)
    if factory is None:
        available = ", ".join(sorted(_ENGINE_FACTORIES.keys()))
        print(f"Unknown engine '{name}'. Available engines: {available}")
        return None

    try:
        return factory()
    except ModuleNotFoundError as e:
        missing = getattr(e, "name", None) or str(e)
        print(f"Skipping engine '{name}': missing dependency '{missing}'")
        return None
    except Exception as e:
        print(f"Skipping engine '{name}': failed to initialize ({e})")
        return None


def get_engine(name: str) -> Optional[BaseEngine]:
    """Retrieves a single engine instance by its name."""
    return _try_create_engine(name)


def get_all_engines() -> Dict[str, BaseEngine]:
    engines: Dict[str, BaseEngine] = {}
    for name in sorted(_ENGINE_FACTORIES.keys()):
        eng = _try_create_engine(name)
        if eng is not None:
            engines[name] = eng
    return engines


def get_engines_by_names(names: List[str]) -> Dict[str, BaseEngine]:
    if not names:
        return get_all_engines()

    engines: Dict[str, BaseEngine] = {}
    for name in names:
        eng = _try_create_engine(name)
        if eng is not None:
            engines[name] = eng
    return engines


def get_engine_factory(name: str) -> Optional[Callable[..., BaseEngine]]:
    """Return a factory callable for the named engine that accepts **extra_kwargs.

    Extra kwargs are merged on top of the base config kwargs, allowing callers to
    pass e.g. ``skip_index=True`` to create a worker-thread client without touching
    the index:

        factory = get_engine_factory("qdrant")
        worker_eng = factory(skip_index=True)   # own connection, no index setup
    """
    spec = _ENGINE_SPECS.get(name)
    if spec is None:
        return None
    module_path, class_name = spec
    base_kwargs = _ENGINE_CONFIG.get(name, {})
    cls = _load_class(module_path, class_name)

    def factory(**extra_kwargs) -> BaseEngine:
        return cls(**{**base_kwargs, **extra_kwargs})

    return factory


def get_all_engine_factories() -> Dict[str, Callable[..., BaseEngine]]:
    """Return factories for all configured engines, keyed by engine name."""
    result = {}
    for name in sorted(_ENGINE_SPECS.keys()):
        f = get_engine_factory(name)
        if f is not None:
            result[name] = f
    return result
