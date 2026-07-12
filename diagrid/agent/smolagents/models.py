# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"id": self.id, "name": self.name, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(id=data["id"], name=data["name"], args=data.get("args", {}))


@dataclass
class ChatEntry:
    """A single plain role/content conversation turn.

    smolagents flattens tool calls and their observations into plain
    ``assistant``/``user`` text turns before sending them to the model (see
    ``tool_role_conversions`` in ``smolagents.models``) — there is no
    ``"tool"`` role on the wire. Keeping history in this same flattened shape
    means workflow state doesn't need to round-trip smolagents' own
    ``AgentMemory``/``ActionStep`` objects.
    """

    role: str
    """One of ``"system"``, ``"user"``, ``"assistant"``."""

    content: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatEntry":
        """Create from dictionary."""
        return cls(role=data["role"], content=data["content"])

    def to_model_message(self) -> dict[str, Any]:
        """Build the dict shape ``Model.generate()`` expects.

        ``content`` must be a list of content blocks, not a bare string —
        ``get_clean_message_list()`` merges consecutive same-role turns and
        that merge path asserts ``isinstance(content, list)``. A real agent
        transcript hits this (an assistant "thought" turn is immediately
        followed by an assistant "tool call" turn), so it isn't optional.
        """
        return {"role": self.role, "content": [{"type": "text", "text": self.content}]}


@dataclass
class ToolDefinition:
    """Serializable tool schema (smolagents ``Tool.inputs`` shape)."""

    name: str
    description: str
    inputs: dict[str, Any]
    output_type: str = "string"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "output_type": self.output_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            inputs=data.get("inputs") or {},
            output_type=data.get("output_type", "string"),
        )


@dataclass
class AgentConfig:
    """Serializable agent configuration.

    ``model_key`` looks up the live ``Model`` instance in the process-local
    registry (``diagrid.agent.smolagents.workflow``) — an arbitrary
    pre-configured ``Model`` (OpenAI, LiteLLM, a custom subclass, ...) can't
    generally be reconstructed from a name alone.
    """

    name: str
    model_key: str
    tool_definitions: list[ToolDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model_key": self.model_key,
            "tool_definitions": [td.to_dict() for td in self.tool_definitions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            model_key=data["model_key"],
            tool_definitions=[
                ToolDefinition.from_dict(td) for td in data.get("tool_definitions", [])
            ],
        )


@dataclass
class AgentWorkflowInput:
    """Input for the agent workflow."""

    agent_config: AgentConfig
    messages: list[ChatEntry]
    session_id: str
    iteration: int = 0
    max_iterations: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "session_id": self.session_id,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentWorkflowInput":
        """Create from dictionary."""
        return cls(
            agent_config=AgentConfig.from_dict(data["agent_config"]),
            messages=[ChatEntry.from_dict(m) for m in data["messages"]],
            session_id=data["session_id"],
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 20),
        )


@dataclass
class AgentWorkflowOutput:
    """Output from the agent workflow."""

    final_answer: Optional[str]
    messages: list[ChatEntry]
    iterations: int
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_answer": self.final_answer,
            "messages": [m.to_dict() for m in self.messages],
            "iterations": self.iterations,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentWorkflowOutput":
        """Create from dictionary."""
        return cls(
            final_answer=data.get("final_answer"),
            messages=[ChatEntry.from_dict(m) for m in data["messages"]],
            iterations=data["iterations"],
            status=data.get("status", "completed"),
            error=data.get("error"),
        )


@dataclass
class CallModelInput:
    """Input for the call_model activity."""

    agent_config: AgentConfig
    messages: list[ChatEntry]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallModelInput":
        """Create from dictionary."""
        return cls(
            agent_config=AgentConfig.from_dict(data["agent_config"]),
            messages=[ChatEntry.from_dict(m) for m in data["messages"]],
        )


@dataclass
class CallModelOutput:
    """Output from the call_model activity."""

    content: Optional[str]
    tool_calls: list[ToolCall]
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallModelOutput":
        """Create from dictionary."""
        return cls(
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            error=data.get("error"),
        )


@dataclass
class ExecuteToolInput:
    """Input for the execute_tool activity."""

    tool_call: ToolCall

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"tool_call": self.tool_call.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolInput":
        """Create from dictionary."""
        return cls(tool_call=ToolCall.from_dict(data["tool_call"]))


@dataclass
class ExecuteToolOutput:
    """Output from the execute_tool activity."""

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolOutput":
        """Create from dictionary."""
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            content=data["content"],
            is_error=data.get("is_error", False),
        )
