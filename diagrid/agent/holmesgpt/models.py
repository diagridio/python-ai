# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""IO contracts for the HolmesGPT durable-execution integration.

Activities communicate with the workflow through these Pydantic models.
The workflow allocates monotonically-increasing event sequence numbers
and hands them to activities, so the per-instance event tape is dense
and totally ordered even when tools fan out in parallel.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Workflow-level
# ---------------------------------------------------------------------------


class InvestigationInput(BaseModel):
    """Input passed to ``investigation_workflow`` when scheduled.

    ``messages`` and ``tools`` are pre-rendered by the runner so that the
    workflow itself stays deterministic (no prompt building inside the
    workflow body).

    ``trace_context`` carries an injected OpenTelemetry propagator carrier
    (W3C ``traceparent``/``tracestate`` etc.) so that activities can attach
    their spans to the calling process's trace.
    """

    model_config = ConfigDict(extra="ignore")

    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    max_steps: int = 10
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    request_context: Optional[Dict[str, Any]] = None
    trace_context: Optional[Dict[str, str]] = None


class InvestigationOutput(BaseModel):
    """Final output of ``investigation_workflow``."""

    model_config = ConfigDict(extra="ignore")

    final: Optional[Dict[str, Any]] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None  # "completed" | "max_steps_reached" | error string
    total_iterations: int = 0


# ---------------------------------------------------------------------------
# Activity contracts
# ---------------------------------------------------------------------------


class LLMCallInput(BaseModel):
    """Input to the ``call_llm`` activity."""

    model_config = ConfigDict(extra="ignore")

    instance_id: str
    seq_base: int
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    response_format: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    trace_context: Optional[Dict[str, str]] = None


class LLMCallOutput(BaseModel):
    """Output of the ``call_llm`` activity.

    ``messages`` is the (possibly compacted) message tape the activity used
    for the LLM call. When HolmesGPT's compaction pass replaces the message
    history, the workflow adopts the new tape for the next iteration.
    """

    model_config = ConfigDict(extra="ignore")

    assistant_message: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    response_id: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    compaction: Optional[Dict[str, Any]] = None


class ToolCallInput(BaseModel):
    """Input to the ``invoke_tool`` activity.

    ``previous_tool_calls`` enables HolmesGPT's repeated-tool-call safeguard:
    if this tool with these exact params has already run earlier in the
    investigation, ``prevent_overly_repeated_tool_call`` short-circuits the
    invocation with an error. Each entry has shape
    ``{"tool_name": str, "result": {"params": dict}}`` to match
    ``holmes.core.safeguards``'s expectation.
    """

    model_config = ConfigDict(extra="ignore")

    instance_id: str
    seq_base: int
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    user_approved: bool = False
    session_approved_prefixes: List[str] = Field(default_factory=list)
    request_context: Optional[Dict[str, Any]] = None
    previous_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trace_context: Optional[Dict[str, str]] = None


class ToolCallOutput(BaseModel):
    """Output of the ``invoke_tool`` activity.

    ``raw_result`` is the full ``StructuredToolResult.model_dump(mode="json")``
    from HolmesGPT — the workflow only inspects ``status`` and ``data_str``,
    but the raw payload is preserved for downstream consumers.
    """

    model_config = ConfigDict(extra="ignore")

    tool_call_id: str
    tool_name: str
    status: str
    invocation: Optional[str] = None
    data_str: Optional[str] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    raw_result: Dict[str, Any] = Field(default_factory=dict)


class RecordEventInput(BaseModel):
    """Input to the ``record_event`` activity (used for workflow-level events)."""

    model_config = ConfigDict(extra="ignore")

    instance_id: str
    seq: int
    event: str
    data: Dict[str, Any] = Field(default_factory=dict)
    trace_context: Optional[Dict[str, str]] = None
