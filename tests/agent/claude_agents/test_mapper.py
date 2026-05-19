# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the Claude Agents metadata mapper."""

import unittest
from unittest import mock

from diagrid.agent.core.metadata.mapping.claude_agents import ClaudeAgentsMapper
from diagrid.agent.core.types import SupportedFrameworks


def make_runner(
    *,
    name="claude-agent",
    system_prompt="be helpful",
    model="claude-sonnet-4-6",
    tools=None,
    max_iterations=25,
):
    """Build a minimal duck-typed stand-in for ``DaprWorkflowAgentRunner``.

    The mapper reads private attrs off the runner; we only construct what it
    actually touches so the test stays decoupled from runtime/Dapr init.
    """
    runner = mock.Mock(spec=[])
    runner._name = name
    runner._system_prompt = system_prompt
    runner._model = model
    runner._tools = tools if tools is not None else []
    runner._max_iterations = max_iterations
    return runner


class TestClaudeAgentsMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = ClaudeAgentsMapper()

    def test_maps_basic_runner(self):
        runner = make_runner()
        result = self.mapper.map_agent_metadata(runner, "edge")

        self.assertEqual(result.version, "edge")
        self.assertEqual(result.agent.role, "claude-agent")
        self.assertEqual(result.agent.system_prompt, "be helpful")
        self.assertEqual(result.agent.framework, SupportedFrameworks.CLAUDE_AGENTS)
        self.assertEqual(result.agent.max_iterations, 25)
        self.assertEqual(result.name, "claude-agents-claude-agent")

    def test_llm_metadata_routes_to_anthropic_messages(self):
        runner = make_runner(model="claude-haiku-4-5-20251001")
        result = self.mapper.map_agent_metadata(runner, "edge")

        assert result.llm is not None
        self.assertEqual(result.llm.provider, "anthropic")
        self.assertEqual(result.llm.api, "messages")
        self.assertEqual(result.llm.model, "claude-haiku-4-5-20251001")
        self.assertEqual(result.llm.client, "claude_agent_sdk")

    def test_memory_short_term_is_dapr_workflow(self):
        result = self.mapper.map_agent_metadata(make_runner(), "edge")
        assert result.memory is not None
        assert result.memory.short_term is not None
        self.assertEqual(result.memory.short_term.type, "DaprWorkflow")

    def test_uses_runner_provided_name(self):
        runner = make_runner(name="ignored")
        result = self.mapper.map_agent_metadata(
            runner, "edge", name="explicit-override"
        )
        self.assertEqual(result.name, "explicit-override")

    def test_tools_metadata_extracted_from_handler_objects(self):
        """Tools created via ``claude_agent_sdk.tool`` expose ``name`` and
        ``description`` attributes; the mapper must surface both."""
        tool = mock.Mock()
        tool.name = "search"
        tool.description = "Search the web"

        runner = make_runner(tools=[tool])
        result = self.mapper.map_agent_metadata(runner, "edge")

        assert result.tools is not None
        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].name, "search")
        self.assertEqual(result.tools[0].description, "Search the web")
        self.assertEqual(result.agent.tool_choice, "auto")

    def test_tool_choice_none_when_no_tools(self):
        result = self.mapper.map_agent_metadata(make_runner(tools=[]), "edge")
        self.assertIsNone(result.agent.tool_choice)
        self.assertEqual(result.tools, [])

    def test_instructions_omitted_when_system_prompt_empty(self):
        result = self.mapper.map_agent_metadata(make_runner(system_prompt=""), "edge")
        self.assertIsNone(result.agent.instructions)
        self.assertIsNone(result.agent.system_prompt)

    def test_metadata_dict_carries_framework_and_model(self):
        runner = make_runner(name="weather", model="claude-sonnet-4-6")
        result = self.mapper.map_agent_metadata(runner, "edge")

        assert result.agent.metadata is not None
        self.assertEqual(result.agent.metadata["framework"], "claude-agents")
        self.assertEqual(result.agent.metadata["name"], "weather")
        self.assertEqual(result.agent.metadata["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
