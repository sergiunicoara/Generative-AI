from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Optional

from engines.base import BaseEngine


def _load_class(module_path: str, class_name: str):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


_ENGINE_FACTORIES: Dict[str, Callable[[], BaseEngine]] = {
    "qdrant": lambda: _load_class("engines.qdrant_engine", "QdrantEngine")(),
    "elasticsearch": lambda: _load_class("engines.elasticsearch_engine", "ElasticsearchEngine")(),
    "pgvector": lambda: _load_class("engines.pgvector_engine", "PgVectorEngine")(),
    "redis": lambda: _load_class("engines.redis_engine", "RedisEngine")(),
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