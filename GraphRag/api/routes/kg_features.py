"""Backward-compatibility shim — re-exports the combined KG router from api.routes.kg."""

from api.routes.kg.router import router  # noqa: F401

__all__ = ["router"]
