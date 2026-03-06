# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for workflow and activities."""

import pytest

from diagrid.agent.crewai.workflow import (
    register_tool,
    get_registered_tool,
    get_tool_definition,
    clear_tool_registry,
    _execute_tool,
    _build_system_prompt,
)
from diagrid.agent.crewai.models import (
    AgentConfig,
    TaskConfig,
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


class TestExecuteTool:
    def test_execute_callable(self):
        def add(a: int, b: int) -> int:
            return a + b

        result = _execute_tool(add, {"a": 1, "b": 2})
        assert result == 3

    def test_execute_tool_with_run_method(self):
        class MyTool:
            def run(self, query: str) -> str:
                return f"searched: {query}"

        tool = MyTool()
        result = _execute_tool(tool, {"query": "test"})
        assert result == "searched: test"

    def test_execute_tool_with_private_run_method(self):
        class MyTool:
            def _run(self, query: str) -> str:
                return f"searched: {query}"

        tool = MyTool()
        result = _execute_tool(tool, {"query": "test"})
        assert result == "searched: test"

    def test_execute_tool_with_invoke_method(self):
        class MyTool:
            def invoke(self, args: dict) -> str:
                return f"invoked with: {args}"

        tool = MyTool()
        result = _execute_tool(tool, {"query": "test"})
        assert "invoked with" in result


class TestBuildSystemPrompt:
    def test_default_prompt(self):
        agent_config = AgentConfig(
            role="Research Assistant",
            goal="Help users find information",
            backstory="An expert researcher",
            model="gpt-4",
        )
        task_config = TaskConfig(
            description="Find info about AI",
            expected_output="A summary",
        )

        prompt = _build_system_prompt(agent_config, task_config)

        assert "Research Assistant" in prompt
        assert "Help users find information" in prompt
        assert "An expert researcher" in prompt
        assert "Find info about AI" in prompt
        assert "A summary" in prompt

    def test_custom_template(self):
        agent_config = AgentConfig(
            role="Assistant",
            goal="Help",
            backstory="Expert",
            model="gpt-4",
            system_template="You are {role}. Goal: {goal}. Task: {task_description}",
        )
        task_config = TaskConfig(
            description="Do the thing",
            expected_output="Result",
        )

        prompt = _build_system_prompt(agent_config, task_config)

        assert prompt == "You are Assistant. Goal: Help. Task: Do the thing"

    def test_prompt_with_context(self):
        agent_config = AgentConfig(
            role="Assistant",
            goal="Help",
            backstory="Expert",
            model="gpt-4",
        )
        task_config = TaskConfig(
            description="Do the thing",
            expected_output="Result",
            context="Previous results showed X",
        )

        prompt = _build_system_prompt(agent_config, task_config)

        assert "Previous results showed X" in prompt
