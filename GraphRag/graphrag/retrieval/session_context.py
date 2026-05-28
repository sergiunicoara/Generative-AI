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

Sessions are in-memory (dict) — they reset on process restart.
For persistent sessions, swap the dict for Redis or a DB table.
"""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime, timezone

import structlog

from graphrag.core.models import SessionTurn

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
    In-memory multi-turn session context.

    Usage::

        ctx = SessionContext()
        enriched_query = ctx.enrich_query(session_id, "What about their engines?")
        # → "What about SpaceX's engines? (context: SpaceX, Falcon 9)"
        ctx.record_turn(session_id, question, answer, entities, chunks)
    """

    def __init__(self):
        # session_id → deque of SessionTurn
        self._sessions: dict[str, deque[SessionTurn]] = {}

    def _get_turns(self, session_id: str) -> deque[SessionTurn]:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=SESSION_MAX_TURNS)
        return self._sessions[session_id]

    def record_turn(
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
        turns = self._get_turns(session_id)
        turns.append(
            SessionTurn(
                question=question,
                answer=answer,
                referenced_entities=referenced_entities,
                referenced_chunks=referenced_chunks,
            )
        )
        log.debug(
            "session_context.turn_recorded",
            session_id=session_id,
            entities=referenced_entities,
            total_turns=len(turns),
        )

    def enrich_query(self, session_id: str, question: str) -> str:
        """
        If the question contains ambiguity signals, append the most
        recently referenced entities as explicit context.

        Returns the (possibly enriched) query string.
        """
        if not session_id:
            return question

        turns = self._get_turns(session_id)
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

    def get_recent_entities(self, session_id: str, n: int = 5) -> list[str]:
        """Return the N most recently referenced unique entity names."""
        if not session_id:
            return []
        turns = self._get_turns(session_id)
        seen: list[str] = []
        for turn in reversed(list(turns)):
            for entity in turn.referenced_entities:
                if entity not in seen:
                    seen.append(entity)
            if len(seen) >= n:
                break
        return seen[:n]

    def get_recent_chunks(self, session_id: str, n: int = 10) -> list[str]:
        """Return the N most recently retrieved chunk IDs."""
        if not session_id:
            return []
        turns = self._get_turns(session_id)
        seen: list[str] = []
        for turn in reversed(list(turns)):
            for chunk_id in turn.referenced_chunks:
                if chunk_id not in seen:
                    seen.append(chunk_id)
            if len(seen) >= n:
                break
        return seen[:n]

    def clear_session(self, session_id: str) -> None:
        """Remove all history for a session (e.g. user logs out)."""
        self._sessions.pop(session_id, None)

    def session_summary(self, session_id: str) -> dict:
        """Return a summary of the session for debugging."""
        turns = self._get_turns(session_id)
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "recent_entities": self.get_recent_entities(session_id),
            "recent_chunks": self.get_recent_chunks(session_id, 5),
        }


# ── Module-level singleton ─────────────────────────────────────────────────────

_session_context: SessionContext | None = None


def get_session_context() -> SessionContext:
    global _session_context
    if _session_context is None:
        _session_context = SessionContext()
    return _session_context
