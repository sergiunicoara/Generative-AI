"""Token usage ledger: measured prompt/completion tokens per kind of call, never invented.

A ``UsageLedger`` collects what providers report for each successful call. It is active only inside a
``ledger_scope`` (a context variable, so concurrent tasks do not mix), and the *kind* of call
(``extraction``, ``artifacts``, ``community``, ``embedding``, ``generation``, ``rewrite``, ``judge``) comes from
``kind_scope`` or from the reporter itself. Outside a scope every ``record_usage`` is a no-op, so production
paths cost nothing when nobody is measuring.

Rules that keep the numbers honest:

* a call whose provider returned no usage counts in ``n_calls`` and ``n_calls_without_usage`` and adds no tokens;
* ``totals`` returns token sums only when at least one call was seen and **every** call reported usage, otherwise
  ``None``: "unknown" is never turned into zero;
* ``cost_usd`` is the ``estimated_cost_usd`` of ``graphrag.observability.pricing`` summed over buckets, ``None`` as
  soon as one bucket has no price or is partly unmeasured (embeddings have no price row, so their cost is ``None``).
  It is an estimate at the table's off-peak rate, not a bill;
* only successful responses reach the ledger: a provider-internal retry that failed is not counted, so spend is
  slightly understated when retries happen.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

KINDS = ("extraction", "artifacts", "community", "embedding", "generation", "rewrite", "judge")
_UNSPECIFIED = "unspecified"

_ledger: ContextVar["UsageLedger | None"] = ContextVar("usage_ledger", default=None)
_kind: ContextVar[str] = ContextVar("usage_kind", default=_UNSPECIFIED)


@dataclass
class _Bucket:
    prompt: int = 0
    completion: int = 0
    n_calls: int = 0
    n_unmeasured: int = 0


class UsageLedger:
    """Thread-safe accumulator keyed by (kind, provider, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str, str], _Bucket] = {}

    def add(self, kind: str, provider: str | None, model: str | None,
            prompt_tokens: int | None, completion_tokens: int | None) -> None:
        key = (kind or _UNSPECIFIED, provider or "unknown", model or "unknown")
        with self._lock:
            bucket = self._buckets.setdefault(key, _Bucket())
            bucket.n_calls += 1
            if prompt_tokens is None or completion_tokens is None:
                bucket.n_unmeasured += 1          # a half-known call is treated as unknown, not as zero
            else:
                bucket.prompt += int(prompt_tokens)
                bucket.completion += int(completion_tokens)

    def _select(self, kind: str | None) -> list[tuple[tuple[str, str, str], _Bucket]]:
        with self._lock:
            return [(key, _Bucket(**vars(bucket))) for key, bucket in self._buckets.items()
                    if kind is None or key[0] == kind]

    def totals(self, kind: str | None = None) -> dict:
        """Token and cost totals of one kind (or of everything). Tokens and cost are None when not fully measured."""
        from graphrag.observability.pricing import estimated_cost_usd

        selected = self._select(kind)
        n_calls = sum(bucket.n_calls for _, bucket in selected)
        n_unmeasured = sum(bucket.n_unmeasured for _, bucket in selected)
        measured = n_calls > 0 and n_unmeasured == 0
        prompt = sum(bucket.prompt for _, bucket in selected) if measured else None
        completion = sum(bucket.completion for _, bucket in selected) if measured else None
        cost: float | None = None
        if measured:
            parts = [estimated_cost_usd(key[1], key[2], bucket.prompt, bucket.completion) for key, bucket in selected]
            cost = None if any(part is None for part in parts) else round(sum(parts), 6)
        return {"n_calls": n_calls, "n_calls_without_usage": n_unmeasured, "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion if measured else None, "cost_usd": cost}

    def snapshot(self) -> list[dict]:
        """Every bucket (kind, provider, model) with its counts, sorted, for provenance in result files."""
        return [{"kind": key[0], "provider": key[1], "model": key[2], "n_calls": bucket.n_calls,
                 "n_calls_without_usage": bucket.n_unmeasured, "prompt_tokens": bucket.prompt,
                 "completion_tokens": bucket.completion}
                for key, bucket in sorted(self._select(None))]


def current_ledger() -> UsageLedger | None:
    return _ledger.get()


def record_usage(*, provider: str | None, model: str | None, prompt_tokens: int | None,
                 completion_tokens: int | None, kind: str | None = None) -> None:
    """Report one successful call to the active ledger; a no-op outside ``ledger_scope``."""
    ledger = _ledger.get()
    if ledger is None:
        return
    ledger.add(kind or _kind.get(), provider, model, prompt_tokens, completion_tokens)


@contextmanager
def ledger_scope(ledger: UsageLedger, kind: str | None = None):
    """Activate ``ledger`` (and optionally a default kind) for the calls made inside, including child tasks."""
    ledger_token = _ledger.set(ledger)
    kind_token = _kind.set(kind) if kind else None
    try:
        yield ledger
    finally:
        if kind_token is not None:
            _kind.reset(kind_token)
        _ledger.reset(ledger_token)


@contextmanager
def kind_scope(kind: str):
    """Change only the kind of the calls made inside (for example ``rewrite`` inside ``generation``)."""
    token = _kind.set(kind)
    try:
        yield
    finally:
        _kind.reset(token)
