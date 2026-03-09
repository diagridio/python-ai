# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    MODEL = "model"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any
    error: Optional[str] = None


@dataclass
class Message:
    """A serializable message in the conversation.

    This is a simplified representation of ADK's Content/Event structure
    that can be serialized for Dapr workflow state.
    """

    role: MessageRole
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "args": tc.args}
                for tc in self.tool_calls
            ],
            "tool_results": [
                {
                    "tool_call_id": tr.tool_call_id,
                    "tool_name": tr.tool_name,
                    "result": tr.result,
                    "error": tr.error,
                }
                for tr in self.tool_results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=[
                ToolCall(id=tc["id"], name=tc["name"], args=tc["args"])
                for tc in data.get("tool_calls", [])
            ],
            tool_results=[
                ToolResult(
                    tool_call_id=tr["tool_call_id"],
                    tool_name=tr["tool_name"],
                    result=tr["result"],
                    error=tr.get("error"),
                )
                for tr in data.get("tool_results", [])
            ],
        )


@dataclass
class ToolDefinition:
    """Serializable tool definition."""

    name: str
    description: str
    parameters: Optional[dict[str, Any]] = None

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
            parameters=data.get("parameters"),
        )


@dataclass
class AgentConfig:
    """Serializable agent configuration."""

    name: str
    model: str
    system_instruction: Optional[str] = None
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    component_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "model": self.model,
            "system_instruction": self.system_instruction,
            "tool_definitions": [td.to_dict() for td in self.tool_definitions],
            "component_name": self.component_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            model=data["model"],
            system_instruction=data.get("system_instruction"),
            tool_definitions=[
                ToolDefinition.from_dict(td) for td in data.get("tool_definitions", [])
            ],
            component_name=data.get("component_name"),
        )


@dataclass
class AgentWorkflowInput:
    """Input for the agent workflow."""

    agent_config: AgentConfig
    messages: list[Message]
    session_id: str
    user_id: Optional[str] = None
    app_name: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "session_id": self.session_id,
            "user_id": self.user_id,
            "app_name": self.app_name,
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
            user_id=data.get("user_id"),
            app_name=data.get("app_name"),
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 100),
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
    agent_name: str
    session_id: str
    user_id: Optional[str] = None
    app_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call": {
                "id": self.tool_call.id,
                "name": self.tool_call.name,
                "args": self.tool_call.args,
            },
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "app_name": self.app_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolInput":
        """Create from dictionary."""
        tc = data["tool_call"]
        return cls(
            tool_call=ToolCall(id=tc["id"], name=tc["name"], args=tc["args"]),
            agent_name=data["agent_name"],
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            app_name=data.get("app_name"),
        )


@dataclass
class ExecuteToolOutput:
    """Output from the execute_tool activity."""

    tool_result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_result": {
                "tool_call_id": self.tool_result.tool_call_id,
                "tool_name": self.tool_result.tool_name,
                "result": self.tool_result.result,
                "error": self.tool_result.error,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolOutput":
        """Create from dictionary."""
        tr = data["tool_result"]
        return cls(
            tool_result=ToolResult(
                tool_call_id=tr["tool_call_id"],
                tool_name=tr["tool_name"],
                result=tr["result"],
                error=tr.get("error"),
            ),
        )
