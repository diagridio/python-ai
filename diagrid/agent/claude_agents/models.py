# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Serializable data models for Claude Agent SDK Dapr Workflow integration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    args: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "args": self.args}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(id=data["id"], name=data["name"], args=data["args"])


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class Message:
    """A serializable conversation message.

    Mirrors the Anthropic Messages API shape so the workflow can checkpoint
    the conversation state between activities.
    """

    role: MessageRole
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
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
    """Serializable tool definition (name + JSON Schema)."""

    name: str
    description: str
    parameters: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDefinition":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters"),
        )


@dataclass
class AgentConfig:
    """Serializable Claude agent configuration."""

    name: str
    system_prompt: str
    model: str
    tool_definitions: list[ToolDefinition] = field(default_factory=list)
    max_tokens: int = 4096
    component_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "model": self.model,
            "tool_definitions": [td.to_dict() for td in self.tool_definitions],
            "max_tokens": self.max_tokens,
            "component_name": self.component_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        return cls(
            name=data["name"],
            system_prompt=data["system_prompt"],
            model=data["model"],
            tool_definitions=[
                ToolDefinition.from_dict(td) for td in data.get("tool_definitions", [])
            ],
            max_tokens=data.get("max_tokens", 4096),
            component_name=data.get("component_name"),
        )


@dataclass
class AgentWorkflowInput:
    """Input for the Claude agent workflow."""

    agent_config: AgentConfig
    messages: list[Message]
    session_id: str
    iteration: int = 0
    max_iterations: int = 25

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "session_id": self.session_id,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentWorkflowInput":
        return cls(
            agent_config=AgentConfig.from_dict(data["agent_config"]),
            messages=[Message.from_dict(m) for m in data["messages"]],
            session_id=data["session_id"],
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 25),
        )


@dataclass
class AgentWorkflowOutput:
    """Output from the Claude agent workflow."""

    final_response: Optional[str]
    messages: list[Message]
    iterations: int
    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_response": self.final_response,
            "messages": [m.to_dict() for m in self.messages],
            "iterations": self.iterations,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentWorkflowOutput":
        return cls(
            final_response=data.get("final_response"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            iterations=data["iterations"],
            status=data.get("status", "completed"),
            error=data.get("error"),
        )


@dataclass
class CallLlmInput:
    """Input for the call_llm activity (one Anthropic API call)."""

    agent_config: AgentConfig
    messages: list[Message]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_config": self.agent_config.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallLlmInput":
        return cls(
            agent_config=AgentConfig.from_dict(data["agent_config"]),
            messages=[Message.from_dict(m) for m in data["messages"]],
        )


@dataclass
class CallLlmOutput:
    """Output from the call_llm activity."""

    message: Message
    is_final: bool
    stop_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "is_final": self.is_final,
            "stop_reason": self.stop_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CallLlmOutput":
        return cls(
            message=Message.from_dict(data["message"]),
            is_final=data["is_final"],
            stop_reason=data.get("stop_reason"),
            error=data.get("error"),
        )


@dataclass
class ExecuteToolInput:
    """Input for the execute_tool activity."""

    tool_call: ToolCall
    agent_name: str
    session_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call": self.tool_call.to_dict(),
            "agent_name": self.agent_name,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolInput":
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
        return {"tool_result": self.tool_result.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecuteToolOutput":
        return cls(tool_result=ToolResult.from_dict(data["tool_result"]))
