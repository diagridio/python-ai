# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Canonical message types for the Dapr Conversation API adapter.

These types serve as the lingua franca between framework-specific message
formats and the Dapr Conversation API. Each framework adapter converts its
native messages to/from these types when routing LLM calls through Dapr.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChatRole(str, Enum):
    """Role of a message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: str  # JSON string

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatToolCall":
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
        )


@dataclass
class ChatToolDefinition:
    """A tool definition to pass to the LLM."""

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
    def from_dict(cls, data: dict[str, Any]) -> "ChatToolDefinition":
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters"),
        )


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: ChatRole
    content: Optional[str] = None
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=ChatRole(data["role"]),
            content=data.get("content"),
            tool_calls=[
                ChatToolCall.from_dict(tc) for tc in data.get("tool_calls", [])
            ],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class ChatResponse:
    """Response from a chat completion call."""

    content: Optional[str] = None
    tool_calls: list[ChatToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    context_id: Optional[str] = None

    @property
    def is_final(self) -> bool:
        """Whether this is a final response (no tool calls)."""
        return len(self.tool_calls) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason,
            "context_id": self.context_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatResponse":
        return cls(
            content=data.get("content"),
            tool_calls=[
                ChatToolCall.from_dict(tc) for tc in data.get("tool_calls", [])
            ],
            finish_reason=data.get("finish_reason", "stop"),
            context_id=data.get("context_id"),
        )
