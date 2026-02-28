"""Serializable data models for Pydantic AI Dapr Workflow integration."""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            args=data["args"],
        )


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            result=data["result"],
            error=data.get("error"),
        )


@dataclass
class Message:
    """A serializable message in the conversation.

    This is a simplified representation that can be serialized for Dapr workflow state.
    """

    role: MessageRole
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_call_id: Optional[str] = None  # For tool response messages
    name: Optional[str] = None  # Tool name for tool response messages

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data.get("content"),
            tool_calls=[ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            tool_results=[
                ToolResult.from_dict(tr) for tr in data.get("tool_results", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
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
    system_prompt: str
    model: str
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    component_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "tool_definitions": [td.to_dict() for td in self.tool_definitions],
            "component_name": self.component_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            model=data["model"],
            tool_definitions=[
                ToolDefinition.from_dict(td) for td in data.get("tool_definitions", [])
            ],
            component_name=data.get("component_name"),
        )


@dataclass
class AgentWorkflowInput:
    """Input for the Pydantic AI workflow."""

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
    """Output from the Pydantic AI workflow."""

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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call": self.tool_call.to_dict(),
            "agent_name": self.agent_name,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolInput":
        """Create from dictionary."""
        return cls(
            tool_call=ToolCall.from_dict(data["tool_call"]),
            agent_name=data["agent_name"],
            session_id=data["session_id"],
        )


@dataclass
class ExecuteToolOutput:
    """Output from the execute_tool activity."""

    tool_result: ToolResult

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_result": self.tool_result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolOutput":
        """Create from dictionary."""
        return cls(
            tool_result=ToolResult.from_dict(data["tool_result"]),
        )
