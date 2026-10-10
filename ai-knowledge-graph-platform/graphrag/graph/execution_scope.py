"""Per-operation guard rails for graph execution (plan Phase 7).

A guarded operation (MCP capability, agent tool) runs inside an
``execution_scope``. While it is active, every ``Neo4jClient.run`` call:

* uses a READ-access session when ``read_only`` (the server refuses writes, so
  a read operation cannot mutate data even if a code path tries to);
* carries a server-side transaction timeout (``neo4j.Query(timeout=...)``);
* stops reading once ``max_rows`` is exceeded and raises ``ResultTooLarge``
  instead of materialising an unbounded result.

Context variables propagate through ``await`` and, when copied explicitly
(``contextvars.copy_context().run``), into executor threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


class ResultTooLarge(RuntimeError):
    """A guarded operation produced more rows than its limit."""


@dataclass(frozen=True)
class ExecutionScope:
    read_only: bool = True
    timeout_s: float | None = None
    max_rows: int | None = None
    operation: str = ""


_SCOPE: ContextVar[ExecutionScope | None] = ContextVar("graph_execution_scope", default=None)


def current_scope() -> ExecutionScope | None:
    return _SCOPE.get()


@contextmanager
def execution_scope(*, read_only: bool = True, timeout_s: float | None = None,
                    max_rows: int | None = None, operation: str = ""):
    token = _SCOPE.set(ExecutionScope(read_only, timeout_s, max_rows, operation))
    try:
        yield _SCOPE.get()
    finally:
        _SCOPE.reset(token)
