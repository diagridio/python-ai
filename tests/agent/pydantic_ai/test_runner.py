# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for Pydantic AI DaprWorkflowAgentRunner."""

from unittest import mock

import pytest

from diagrid.agent.pydantic_ai.runner import DaprWorkflowAgentRunner
from diagrid.agent.pydantic_ai.models import ToolDefinition


class FakeToolInfo:
    """Simulates a Pydantic AI tool info object."""

    def __init__(
        self, name, description="", parameters_json_schema=None, function=None
    ):
        self.name = name
        self.description = description
        self.parameters_json_schema = parameters_json_schema
        self.function = function


class FakeAgent:
    """Simulates a pydantic_ai.Agent for unit testing."""

    def __init__(
        self,
        model=None,
        name=None,
        _system_prompts=None,
        _function_tools=None,
    ):
        self.model = model
        self.name = name
        self._system_prompts = _system_prompts or []
        self._function_tools = _function_tools or {}


class TestCreateToolDefinition:
    def _make_runner_partial(self):
        runner = object.__new__(DaprWorkflowAgentRunner)
        runner._agent = FakeAgent()
        return runner

    def test_basic_tool(self):
        runner = self._make_runner_partial()
        tool = FakeToolInfo(name="search", description="Search the web")
        td = runner._create_tool_definition(tool, "search")
        assert td.name == "search"
        assert td.description == "Search the web"
        assert td.parameters is None

    def test_tool_with_schema(self):
        runner = self._make_runner_partial()
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        tool = FakeToolInfo(
            name="search",
            description="Search",
            parameters_json_schema=schema,
        )
        td = runner._create_tool_definition(tool, "search")
        assert td.parameters == schema

    def test_tool_with_callable_schema(self):
        runner = self._make_runner_partial()
        schema = {"type": "object", "properties": {}}
        tool = FakeToolInfo(
            name="search",
            description="Search",
            parameters_json_schema=lambda: schema,
        )
        td = runner._create_tool_definition(tool, "search")
        assert td.parameters == schema

    def test_tool_with_broken_callable_schema(self):
        runner = self._make_runner_partial()

        def broken():
            raise ValueError("broken")

        tool = FakeToolInfo(
            name="test",
            description="test",
            parameters_json_schema=broken,
        )
        td = runner._create_tool_definition(tool, "test")
        assert td.parameters is None


class TestGetAgentConfig:
    def _make_runner_partial(self, agent):
        runner = object.__new__(DaprWorkflowAgentRunner)
        runner._agent = agent
        return runner

    def test_basic_config(self):
        agent = FakeAgent(
            model="openai:gpt-4o",
            name="my-agent",
            _system_prompts=["You are helpful."],
        )
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert config.name == "my-agent"
        assert config.system_prompt == "You are helpful."
        assert config.model == "openai:gpt-4o"

    def test_no_name_defaults(self):
        agent = FakeAgent(model="openai:gpt-4o", name=None)
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert config.name == "pydantic-ai-agent"

    def test_multiple_system_prompts(self):
        agent = FakeAgent(
            model="openai:gpt-4o",
            name="test",
            _system_prompts=["Part one.", "Part two."],
        )
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert config.system_prompt == "Part one.\nPart two."

    def test_callable_system_prompt(self):
        agent = FakeAgent(
            model="openai:gpt-4o",
            name="test",
            _system_prompts=[lambda: "Dynamic prompt"],
        )
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert config.system_prompt == "Dynamic prompt"

    def test_with_tools(self):
        def search(query: str) -> str:
            return query

        tool_info = FakeToolInfo(
            name="search",
            description="Search tool",
            function=search,
        )
        agent = FakeAgent(
            model="openai:gpt-4o",
            name="test",
            _function_tools={"search": tool_info},
        )
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert len(config.tool_definitions) == 1
        assert config.tool_definitions[0].name == "search"

    def test_model_non_string(self):
        """Model objects should be converted to string."""

        class FakeModel:
            def __str__(self):
                return "anthropic:claude-3-5-sonnet"

        agent = FakeAgent(model=FakeModel(), name="test")
        runner = self._make_runner_partial(agent)
        config = runner._get_agent_config()
        assert config.model == "anthropic:claude-3-5-sonnet"


class TestRegisterAgentTools:
    def test_registers_tools_from_function_tools(self):
        from diagrid.agent.pydantic_ai.workflow import (
            get_registered_tool,
            clear_tool_registry,
        )

        def my_func(x: str) -> str:
            return x

        tool_info = FakeToolInfo(
            name="my_func",
            description="A function",
            function=my_func,
        )
        agent = FakeAgent(_function_tools={"my_func": tool_info})

        runner = object.__new__(DaprWorkflowAgentRunner)
        runner._agent = agent
        runner._register_agent_tools()

        assert get_registered_tool("my_func") is my_func

        clear_tool_registry()

    def test_registers_tool_info_as_fallback(self):
        from diagrid.agent.pydantic_ai.workflow import (
            get_registered_tool,
            clear_tool_registry,
        )

        tool_info = FakeToolInfo(name="raw_tool", description="No function attr")
        # Remove the function attribute to test fallback
        del tool_info.function

        agent = FakeAgent(_function_tools={"raw_tool": tool_info})

        runner = object.__new__(DaprWorkflowAgentRunner)
        runner._agent = agent
        runner._register_agent_tools()

        assert get_registered_tool("raw_tool") is tool_info

        clear_tool_registry()
