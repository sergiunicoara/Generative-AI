"""Abstract Google ADK Agent base with shared tool registration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class BaseGraphRAGAgent(ABC):
    """
    Thin wrapper around the Google ADK Agent.
    Subclasses register tools via `_tools()` and implement `run()`.
    """

    def __init__(self, name: str):
        self.name = name
        self._agent = self._build_agent()

    def _build_agent(self):
        try:
            from google.adk.agents import Agent
            from graphrag.core.config import get_settings

            cfg = get_settings()
            return Agent(
                name=self.name,
                model=self._model(),
                tools=self._tools(),
                instruction=self._instruction(),
            )
        except ImportError:
            log.warning("google_adk.not_installed — running in tool-only mode")
            return None

    @abstractmethod
    def _model(self) -> str:
        """Return the Gemini model ID to use."""

    @abstractmethod
    def _tools(self) -> list:
        """Return list of ADK FunctionTool objects."""

    @abstractmethod
    def _instruction(self) -> str:
        """Return the agent system instruction."""

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        """Execute the agent's primary task."""
