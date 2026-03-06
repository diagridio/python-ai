# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for Pydantic AI workflow and activities."""

import asyncio
import json
import pytest

from diagrid.agent.pydantic_ai.workflow import (
    register_tool,
    get_registered_tool,
    get_tool_definition,
    clear_tool_registry,
    _execute_tool,
    _build_system_prompt,
)
from diagrid.agent.pydantic_ai.models import (
    AgentConfig,
    ToolDefinition,
)


class TestToolRegistry:
    def setup_method(self):
        """Clear registry before each test."""
        clear_tool_registry()

    def test_register_and_get_tool(self):
        def my_tool(x: str) -> str:
            return f"result: {x}"

        tool_def = ToolDefinition(name="my_tool", description="A test tool")
        register_tool("my_tool", my_tool, tool_def)

        retrieved = get_registered_tool("my_tool")
        assert retrieved is my_tool

        retrieved_def = get_tool_definition("my_tool")
        assert retrieved_def is not None
        assert retrieved_def.name == "my_tool"

    def test_get_nonexistent_tool(self):
        result = get_registered_tool("nonexistent")
        assert result is None

    def test_clear_registry(self):
        def my_tool():
            pass

        register_tool("my_tool", my_tool)
        assert get_registered_tool("my_tool") is not None

        clear_tool_registry()
        assert get_registered_tool("my_tool") is None

    def test_register_without_definition(self):
        def my_tool():
            pass

        register_tool("my_tool", my_tool)
        assert get_registered_tool("my_tool") is my_tool
        assert get_tool_definition("my_tool") is None


class TestExecuteTool:
    def test_execute_callable(self):
        def add(a: int, b: int) -> int:
            return a + b

        result = _execute_tool(add, {"a": 1, "b": 2})
        assert result == 3

    def test_execute_async_callable(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = _execute_tool(async_add, {"a": 3, "b": 4})
        assert result == 7

    def test_execute_non_callable_raises(self):
        """Non-callable should raise TypeError."""

        class NotCallable:
            pass

        with pytest.raises(TypeError, match="not callable"):
            _execute_tool(NotCallable(), {})


class TestBuildSystemPrompt:
    def test_returns_system_prompt(self):
        agent_config = AgentConfig(
            name="research_assistant",
            system_prompt="You help users find information. Be concise.",
            model="gpt-4",
        )

        prompt = _build_system_prompt(agent_config)

        assert prompt == "You help users find information. Be concise."

    def test_empty_system_prompt(self):
        agent_config = AgentConfig(
            name="assistant",
            system_prompt="",
            model="gpt-4",
        )

        prompt = _build_system_prompt(agent_config)
        assert prompt == ""
