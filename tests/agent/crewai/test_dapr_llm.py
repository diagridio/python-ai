# -*- coding: utf-8 -*-

"""
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from unittest import mock

from diagrid.agent.crewai.models import (
    AgentConfig,
    CallLlmInput,
    CallLlmOutput,
    Message,
    MessageRole,
    TaskConfig,
    ToolCall,
    ToolDefinition,
)
from diagrid.agent.crewai.workflow import _call_llm_via_dapr


class TestCrewAICallLlmViaDapr(unittest.TestCase):
    """Test _call_llm_via_dapr for CrewAI framework."""

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_basic_chat(self, mock_dapr_cls):
        """Test a simple chat without tool calls."""
        # Mock response
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "The answer is 42."
        mock_choice.message.tool_calls = []

        mock_output = mock.MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = mock.MagicMock()
        mock_response.context_id = None
        mock_response.outputs = [mock_output]

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        llm_input = CallLlmInput(
            agent_config=AgentConfig(
                role="Researcher",
                goal="Find information",
                backstory="Expert researcher",
                model="",
                component_name="llm-provider",
            ),
            task_config=TaskConfig(
                description="What is 6*7?",
                expected_output="A number",
            ),
            messages=[
                Message(role=MessageRole.USER, content="What is 6*7?"),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertTrue(output.is_final)
        self.assertEqual(output.message.content, "The answer is 42.")
        self.assertEqual(output.message.role, MessageRole.ASSISTANT)

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_chat_with_tool_calls(self, mock_dapr_cls):
        """Test chat that returns tool calls."""
        mock_tc_func = mock.MagicMock()
        mock_tc_func.name = "search"
        mock_tc_func.arguments = '{"query": "weather"}'

        mock_tc = mock.MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function = mock_tc_func

        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "tool_calls"
        mock_choice.message.content = ""
        mock_choice.message.tool_calls = [mock_tc]

        mock_output = mock.MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = mock.MagicMock()
        mock_response.context_id = None
        mock_response.outputs = [mock_output]

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        llm_input = CallLlmInput(
            agent_config=AgentConfig(
                role="Researcher",
                goal="Find information",
                backstory="Expert",
                model="",
                tool_definitions=[
                    ToolDefinition(
                        name="search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                component_name="llm-provider",
            ),
            task_config=TaskConfig(
                description="Find the weather",
                expected_output="Weather info",
            ),
            messages=[
                Message(role=MessageRole.USER, content="What's the weather?"),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertFalse(output.is_final)
        self.assertEqual(len(output.message.tool_calls), 1)
        self.assertEqual(output.message.tool_calls[0].name, "search")
        self.assertEqual(output.message.tool_calls[0].args, {"query": "weather"})

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_message_conversion_with_history(self, mock_dapr_cls):
        """Test that all message types are converted correctly."""
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Done."
        mock_choice.message.tool_calls = []

        mock_output = mock.MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = mock.MagicMock()
        mock_response.context_id = None
        mock_response.outputs = [mock_output]

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        llm_input = CallLlmInput(
            agent_config=AgentConfig(
                role="Researcher",
                goal="Help",
                backstory="Expert",
                model="",
                component_name="llm-provider",
            ),
            task_config=TaskConfig(
                description="task",
                expected_output="output",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
                Message(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="search", args={"q": "test"})],
                ),
                Message(
                    role=MessageRole.TOOL,
                    content="result",
                    tool_call_id="c1",
                    name="search",
                ),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)
        self.assertTrue(output.is_final)

        # Verify that converse_alpha2 was called with properly converted messages
        call_args = mock_dapr_cls.return_value.converse_alpha2.call_args
        inputs = call_args.kwargs.get("inputs") or call_args[1].get("inputs")
        # inputs is a list with one ConversationInputAlpha2 wrapping the messages
        self.assertEqual(len(inputs), 1)
        # system + user + assistant + tool = 4 messages
        self.assertEqual(len(inputs[0].messages), 4)


class TestCrewAIComponentName(unittest.TestCase):
    """Test component_name in CrewAI AgentConfig serialization."""

    def test_component_name_serialization(self):
        config = AgentConfig(
            role="test",
            goal="test",
            backstory="test",
            model="gpt-4",
            component_name="my-llm",
        )
        data = config.to_dict()
        self.assertEqual(data["component_name"], "my-llm")

        config2 = AgentConfig.from_dict(data)
        self.assertEqual(config2.component_name, "my-llm")

    def test_component_name_none(self):
        config = AgentConfig(
            role="test",
            goal="test",
            backstory="test",
            model="gpt-4",
        )
        data = config.to_dict()
        self.assertIsNone(data["component_name"])

        config2 = AgentConfig.from_dict(data)
        self.assertIsNone(config2.component_name)


if __name__ == "__main__":
    unittest.main()
