# -*- coding: utf-8 -*-

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1


import unittest
from unittest import mock

from diagrid.agent.adk.models import (
    AgentConfig,
    CallLlmInput,
    CallLlmOutput,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from diagrid.agent.adk.workflow import _call_llm_via_dapr
from diagrid.agent.core.chat.client import _chat_client_cache


class TestADKCallLlmViaDapr(unittest.TestCase):
    """Test _call_llm_via_dapr for ADK framework."""

    def setUp(self):
        _chat_client_cache.clear()

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_basic_chat(self, mock_dapr_cls):
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Hello from ADK!"
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
                model="",
                system_instruction="Be helpful",
                component_name="llm-provider",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Hi"),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertTrue(output.is_final)
        self.assertEqual(output.message.content, "Hello from ADK!")
        # ADK uses MODEL role
        self.assertEqual(output.message.role, MessageRole.MODEL)

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_tool_results_on_user_message(self, mock_dapr_cls):
        """ADK puts tool results on USER messages - verify conversion."""
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Based on the search..."
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
                model="",
                component_name="llm-provider",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Search for weather"),
                Message(
                    role=MessageRole.MODEL,
                    tool_calls=[
                        ToolCall(id="c1", name="search", args={"q": "weather"})
                    ],
                ),
                # ADK-specific: tool results on USER message
                Message(
                    role=MessageRole.USER,
                    tool_results=[
                        ToolResult(
                            tool_call_id="c1",
                            tool_name="search",
                            result="Sunny today",
                        )
                    ],
                ),
            ],
        )

        result = _call_llm_via_dapr(llm_input)
        output = CallLlmOutput.from_dict(result)

        self.assertTrue(output.is_final)
        # Verify converse_alpha2 was called
        mock_dapr_cls.return_value.converse_alpha2.assert_called_once()

        # Verify the tool result was converted to a TOOL message
        call_args = mock_dapr_cls.return_value.converse_alpha2.call_args
        inputs = call_args.kwargs.get("inputs") or call_args[1].get("inputs")
        # inputs is a list with one ConversationInputAlpha2 wrapping the messages
        self.assertEqual(len(inputs), 1)
        # system_instruction not set in this case, so: user + model(assistant) + tool = 3
        self.assertEqual(len(inputs[0].messages), 3)


class TestADKComponentName(unittest.TestCase):
    """Test component_name in ADK AgentConfig serialization."""

    def test_component_name_serialization(self):
        config = AgentConfig(
            name="test",
            model="gemini-2.0-flash",
            component_name="my-llm",
        )
        data = config.to_dict()
        self.assertEqual(data["component_name"], "my-llm")

        config2 = AgentConfig.from_dict(data)
        self.assertEqual(config2.component_name, "my-llm")


if __name__ == "__main__":
    unittest.main()
