"""Tests for Pydantic AI metadata mapper."""

import unittest
from unittest import mock

from diagrid.agent.core.metadata.mapping.pydantic_ai import PydanticAIMapper


class TestPydanticAIMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = PydanticAIMapper()

    # ------------------------------------------------------------------
    # Model extraction
    # ------------------------------------------------------------------

    def test_model_string_with_colon(self):
        """Model as 'provider:model' string should split correctly."""
        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.llm.model, "openai:gpt-4o")
        self.assertEqual(result.llm.provider, "openai")

    def test_model_string_without_colon(self):
        """Model as plain string should have unknown provider."""
        agent = mock.Mock()
        agent.model = "gpt-4o-mini"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.llm.model, "gpt-4o-mini")
        self.assertEqual(result.llm.provider, "unknown")

    def test_model_object_with_model_id(self):
        """Model object with model_id attribute."""
        model_obj = mock.Mock()
        model_obj.model_id = "gpt-4o"
        model_obj.model_name = None
        model_obj.system = "openai"

        agent = mock.Mock()
        agent.model = model_obj
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.llm.model, "gpt-4o")
        self.assertEqual(result.llm.provider, "openai")

    def test_model_object_with_model_name(self):
        """Model object with model_name attribute (no model_id)."""
        model_obj = mock.Mock(spec=[])
        model_obj.model_name = "claude-3-opus"
        model_obj.system = "anthropic"

        agent = mock.Mock()
        agent.model = model_obj
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.llm.model, "claude-3-opus")
        self.assertEqual(result.llm.provider, "anthropic")

    def test_model_none(self):
        """Model as None should return unknown."""
        agent = mock.Mock()
        agent.model = None
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.llm.model, "unknown")
        self.assertEqual(result.llm.provider, "unknown")

    # ------------------------------------------------------------------
    # Tool extraction
    # ------------------------------------------------------------------

    def test_tools_via_function_toolset(self):
        """Tools via _function_toolset.tools (new API)."""
        tool_info = mock.Mock()
        tool_info.description = "A test tool"
        tool_info.function_schema = None

        toolset = mock.Mock()
        toolset.tools = {"my_tool": tool_info}

        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = toolset

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].tool_name, "my_tool")
        self.assertEqual(result.tools[0].tool_description, "A test tool")

    def test_tools_via_function_tools_fallback(self):
        """Tools via _function_tools (old API fallback)."""
        tool_info = mock.Mock()
        tool_info.description = "Legacy tool"
        tool_info.function_schema = None

        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {"old_tool": tool_info}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(len(result.tools), 1)
        self.assertEqual(result.tools[0].tool_name, "old_tool")
        self.assertEqual(result.tools[0].tool_description, "Legacy tool")

    def test_tools_with_json_schema(self):
        """Tools with function_schema.json_schema should serialize to tool_args."""
        func_schema = mock.Mock()
        func_schema.json_schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        tool_info = mock.Mock()
        tool_info.description = "Schema tool"
        tool_info.function_schema = func_schema

        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {"schema_tool": tool_info}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(len(result.tools), 1)
        self.assertIn('"type": "object"', result.tools[0].tool_args)

    # ------------------------------------------------------------------
    # System prompt extraction
    # ------------------------------------------------------------------

    def test_system_prompt_from_system_prompts(self):
        """System prompt from _system_prompts tuple."""
        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ("You are helpful.", "Be concise.")
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.agent.system_prompt, "You are helpful.\nBe concise.")

    def test_system_prompt_from_instructions_fallback(self):
        """System prompt from _instructions when _system_prompts is empty."""
        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = ["Instruction one", "Instruction two"]
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.agent.system_prompt, "Instruction one\nInstruction two")

    def test_system_prompt_empty_when_none_set(self):
        """System prompt is empty string when neither _system_prompts nor _instructions set."""
        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = []
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertIsNone(result.agent.system_prompt)

    def test_system_prompt_skips_callables_in_instructions(self):
        """Callable entries in _instructions should be skipped."""
        agent = mock.Mock()
        agent.model = "openai:gpt-4o"
        agent.name = "test-agent"
        agent._system_prompts = ()
        agent._instructions = [lambda: "dynamic", "Static instruction"]
        agent._function_toolset = None
        agent._function_tools = {}

        result = self.mapper.map_agent_metadata(agent, "edge")

        self.assertEqual(result.agent.system_prompt, "Static instruction")


if __name__ == "__main__":
    unittest.main()
