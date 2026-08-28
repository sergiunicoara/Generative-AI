"""Stable, JSON-serializable records shared by every evaluator."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TraceNode(BaseModel):
    """One chronological node, retaining source-specific detail in ``attributes``."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: Literal["user", "agent", "tool", "state", "error"]
    text: str
    name: str | None = None
    latency_ms: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class EvaluationCase(BaseModel):
    """A normalized trajectory and its independently asserted expected outcome."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    source: Literal["historical", "synthetic"]
    synthetic: bool = False
    source_reference: str
    trajectory: list[TraceNode]
    expected_failure: bool
    expected_failure_types: list[str] = Field(default_factory=list)
    expected_refusal: bool | None = None
    role: str | None = None
    criteria: list[str] = Field(default_factory=list)
    notes: str | None = None
    expected_tool_behavior: dict[str, Any] = Field(default_factory=dict)
    expected_policy_behavior: dict[str, Any] = Field(default_factory=dict)
    redaction_applied: bool = False
    source_payload: dict[str, Any] = Field(default_factory=dict)


class EvaluatorResult(BaseModel):
    """Comparable evaluator output; unavailable fields remain ``None``, never invented."""

    model_config = ConfigDict(extra="forbid")

    evaluator: str
    case_id: str
    status: Literal["ok", "skipped", "error"]
    detected_failure: bool | None = None
    failure_types: list[str] = Field(default_factory=list)
    score: float | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
