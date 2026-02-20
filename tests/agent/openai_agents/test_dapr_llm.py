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

from diagrid.agent.openai_agents.models import (
    AgentConfig,
    CallLlmInput,
    CallLlmOutput,
    Message,
    MessageRole,
    ToolDefinition,
)
from diagrid.agent.openai_agents.workflow import _call_llm_via_dapr


class TestOpenAICallLlmViaDapr(unittest.TestCase):
    """Test _call_llm_via_dapr for OpenAI Agents framework."""

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_basic_chat(self, mock_dapr_cls):
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Hello!"
        mock_choice.message.tool_calls = []

        mock_output = mock.MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = mock.MagicMock()
        mock_response.context_id = None
        mock_response.outputs = [mock_output]

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        llm_input = CallLlmInput(
            agent_config=AgentConfig(
                name="test_agent",
                instructions="Be helpful",
                model="",
                component_name="llm-provider",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertTrue(output.is_final)
        self.assertEqual(output.message.content, "Hello!")

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_chat_with_tool_calls(self, mock_dapr_cls):
        mock_tc_func = mock.MagicMock()
        mock_tc_func.name = "get_weather"
        mock_tc_func.arguments = '{"city": "Tokyo"}'

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
                name="test_agent",
                instructions="Help with weather",
                model="",
                tool_definitions=[
                    ToolDefinition(
                        name="get_weather",
                        description="Get weather",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                component_name="llm-provider",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Weather in Tokyo?"),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertFalse(output.is_final)
        self.assertEqual(len(output.message.tool_calls), 1)
        self.assertEqual(output.message.tool_calls[0].name, "get_weather")


class TestOpenAIComponentName(unittest.TestCase):
    """Test component_name in OpenAI AgentConfig serialization."""

    def test_component_name_serialization(self):
        config = AgentConfig(
            name="test",
            instructions="test",
            model="gpt-4",
            component_name="my-llm",
        )
        data = config.to_dict()
        self.assertEqual(data["component_name"], "my-llm")

        config2 = AgentConfig.from_dict(data)
        self.assertEqual(config2.component_name, "my-llm")

    def test_component_name_none(self):
        config = AgentConfig(
            name="test",
            instructions="test",
            model="gpt-4",
        )
        data = config.to_dict()
        self.assertIsNone(data["component_name"])

        config2 = AgentConfig.from_dict(data)
        self.assertIsNone(config2.component_name)


if __name__ == "__main__":
    unittest.main()
