# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

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

    def to_langchain_tool_call(self) -> dict[str, Any]:
        """Build the ``langchain_core`` tool-call dict shape.

        ``BaseTool.invoke()`` only returns an auto-built ``ToolMessage`` when
        the input dict has ``type == "tool_call"`` *and* an ``id`` — this is
        a literal string check (``langchain_core/tools/base.py``), not a
        duck-typed one, so the key must be present verbatim.
        """
        return {
            "name": self.name,
            "args": self.args,
            "id": self.id,
            "type": "tool_call",
        }


@dataclass
class Message:
    """A serializable chat message.

    Simplified stand-in for ``langchain_core.messages.BaseMessage`` subtypes
    so conversation history can be stored as plain JSON in Dapr workflow
    state. Activities convert to/from real ``langchain_core`` message
    objects at the point they call the model or a tool.
    """

    role: str
    """One of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``."""

    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    """Set only on ``role="tool"`` messages — the ``ToolCall.id`` it answers."""
    name: Optional[str] = None
    """Set only on ``role="tool"`` messages — the tool name that produced it."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class ToolDefinition:
    """Serializable tool schema (OpenAI function-calling shape)."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters") or {},
        )


@dataclass
class AgentConfig:
    """Serializable agent configuration.

    ``model_key`` is a lookup into the process-local model registry
    (``diagrid.agent.langchain.workflow``) rather than a model id string —
    unlike a single provider string (e.g. Gemini's), an arbitrary
    pre-configured ``BaseChatModel`` instance can't generally be
    reconstructed from a name, so the live object is registered once by the
    runner and fetched by key inside the activity.
    """

    name: str
    model_key: str
    system_prompt: Optional[str] = None
    tool_definitions: list[ToolDefinition] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model_key": self.model_key,
            "system_prompt": self.system_prompt,
            "tool_definitions": [td.to_dict() for td in self.tool_definitions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            model_key=data["model_key"],
            system_prompt=data.get("system_prompt"),
            tool_definitions=[
                ToolDefinition.from_dict(td) for td in data.get("tool_definitions", [])
            ],
        )


@dataclass
class AgentWorkflowInput:
    """Input for the agent workflow."""

    agent_config: AgentConfig
    messages: list[Message]
    session_id: str
    iteration: int = 0
    max_iterations: int = 25

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
            messages=[Message.from_dict(m) for m in data["messages"]],
            session_id=data["session_id"],
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 25),
        )


@dataclass
class AgentWorkflowOutput:
    """Output from the agent workflow."""

    final_response: Optional[str]
    messages: list[Message]
    iterations: int
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_response": self.final_response,
            "messages": [m.to_dict() for m in self.messages],
            "iterations": self.iterations,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentWorkflowOutput":
        """Create from dictionary."""
        return cls(
            final_response=data.get("final_response"),
            messages=[Message.from_dict(m) for m in data["messages"]],
            iterations=data["iterations"],
            status=data.get("status", "completed"),
            error=data.get("error"),
        )


@dataclass
class CallLlmInput:
    """Input for the call_llm activity."""

    agent_config: AgentConfig
    messages: list[Message]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallLlmInput":
        """Create from dictionary."""
        return cls(
            agent_config=AgentConfig.from_dict(data["agent_config"]),
            messages=[Message.from_dict(m) for m in data["messages"]],
        )


@dataclass
class CallLlmOutput:
    """Output from the call_llm activity."""

    message: Message
    is_final: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message": self.message.to_dict(),
            "is_final": self.is_final,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallLlmOutput":
        """Create from dictionary."""
        return cls(
            message=Message.from_dict(data["message"]),
            is_final=data["is_final"],
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
