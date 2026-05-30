"""Conversational session context — resolves query ambiguity across turns.

Problem solved
--------------
Stateless retrieval fails on follow-up questions:
  Turn 1: "What rockets did SpaceX launch?"
  Turn 2: "What was their reusability record?"  ← "their" is ambiguous

Without context, Turn 2 retrieves random chunks about reusability.
With context, "their" resolves to SpaceX + Falcon 9 from Turn 1.

Architecture
------------
SessionContext maintains a sliding window of the last N turns per
session_id.  Before retrieval, it rewrites ambiguous queries by
injecting the most recently referenced entities.

Disambiguation signals:
  - Pronouns: they, their, it, its, he, she, his, her, this, that
  - Vague references: "the company", "the system", "the rocket"
  - Implicit continuations: "what about...", "and also...", "how about..."

Persistence
-----------
Session turns are now stored via SessionStore (see session_store.py).
When configured with a Redis URL the history survives process restarts.
With no Redis the behaviour is identical to the previous in-memory dict.

Public async interface:
  await ctx.record_turn(...)
  await ctx.enrich_query(...)
  await ctx.get_recent_entities(...)
  await ctx.get_recent_chunks(...)
  await ctx.clear_session(...)
  await ctx.session_summary(...)
"""

from __future__ import annotations

import re
from collections import deque

import structlog

from graphrag.core.models import SessionTurn
from graphrag.retrieval.session_store import SessionStore, get_session_store

log = structlog.get_logger(__name__)

# Triggers that suggest the query references something from prior context
_AMBIGUITY_SIGNALS = [
    r"\bthey\b", r"\btheir\b", r"\bthem\b",
    r"\bit\b",   r"\bits\b",
    r"\bhe\b",   r"\bhis\b",   r"\bhim\b",
    r"\bshe\b",  r"\bher\b",
    r"\bthis\b", r"\bthat\b",  r"\bthose\b", r"\bthese\b",
    r"\bthe company\b", r"\bthe system\b", r"\bthe rocket\b",
    r"\bthe product\b", r"\bthe organization\b",
    r"\bwhat about\b",  r"\bhow about\b",   r"\band also\b",
]

_AMBIGUITY_RE = re.compile("|".join(_AMBIGUITY_SIGNALS), re.IGNORECASE)

SESSION_MAX_TURNS = 10   # sliding window size per session


class SessionContext:
    """
    Multi-turn session context backed by a persistent SessionStore.

    Usage (async)::

        ctx = SessionContext()
        enriched = await ctx.enrich_query(session_id, "What about their engines?")
        # → "What about SpaceX's engines? (context: SpaceX, Falcon 9)"
        await ctx.record_turn(session_id, question, answer, entities, chunks)
    """

    def __init__(self, store: SessionStore | None = None):
        self._store = store or get_session_store()

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _get_turns(self, session_id: str) -> deque[SessionTurn]:
        return await self._store.load_turns(session_id)

    # ── Write ──────────────────────────────────────────────────────────────────

    async def record_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        referenced_entities: list[str],
        referenced_chunks: list[str],
    ) -> None:
        """Append a completed turn to the session history."""
        if not session_id:
            return
        turn = SessionTurn(
            question=question,
            answer=answer,
            referenced_entities=referenced_entities,
            referenced_chunks=referenced_chunks,
        )
        await self._store.save_turn(session_id, turn)
        log.debug(
            "session_context.turn_recorded",
            session_id=session_id,
            entities=referenced_entities,
            backend="redis" if self._store.is_redis_backed() else "memory",
        )

    # ── Read / enrich ──────────────────────────────────────────────────────────

    async def enrich_query(self, session_id: str, question: str) -> str:
        """
        If the question contains ambiguity signals, append the most
        recently referenced entities as explicit context.

        Returns the (possibly enriched) query string.
        """
        if not session_id:
            return question

        turns = await self._get_turns(session_id)
        if not turns:
            return question

        if not _AMBIGUITY_RE.search(question):
            return question

        # Collect entities from the last 3 turns, most recent first
        recent_entities: list[str] = []
        for turn in reversed(list(turns)):
            for entity in turn.referenced_entities:
                if entity not in recent_entities:
                    recent_entities.append(entity)
            if len(recent_entities) >= 5:
                break

        if not recent_entities:
            return question

        context_hint = ", ".join(recent_entities[:5])
        enriched = f"{question} (context: {context_hint})"
        log.info(
            "session_context.query_enriched",
            session_id=session_id,
            original=question,
            enriched=enriched,
        )
        return enriched

    async def get_recent_entities(self, session_id: str, n: int = 5) -> list[str]:
        """Return the N most recently referenced unique entity names."""
        if not session_id:
            return []
        turns = await self._get_turns(session_id)
        seen: list[str] = []
        for turn in reversed(list(turns)):
            for entity in turn.referenced_entities:
                if entity not in seen:
                    seen.append(entity)
            if len(seen) >= n:
                break
        return seen[:n]

    async def get_recent_chunks(self, session_id: str, n: int = 10) -> list[str]:
        """Return the N most recently retrieved chunk IDs."""
        if not session_id:
            return []
        turns = await self._get_turns(session_id)
        seen: list[str] = []
        for turn in reversed(list(turns)):
            for chunk_id in turn.referenced_chunks:
                if chunk_id not in seen:
                    seen.append(chunk_id)
            if len(seen) >= n:
                break
        return seen[:n]

    async def clear_session(self, session_id: str) -> None:
        """Remove all history for a session (e.g. user logs out)."""
        await self._store.clear(session_id)

    async def session_summary(self, session_id: str) -> dict:
        """Return a summary of the session for debugging."""
        turns = await self._get_turns(session_id)
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "recent_entities": await self.get_recent_entities(session_id),
            "recent_chunks": await self.get_recent_chunks(session_id, 5),
            "backend": "redis" if self._store.is_redis_backed() else "memory",
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

_session_context: SessionContext | None = None


def get_session_context() -> SessionContext:
    global _session_context
    if _session_context is None:
        _session_context = SessionContext()
    return _session_context
