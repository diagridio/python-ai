# -*- coding: utf-8 -*-

# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1


import unittest
from collections import namedtuple
from unittest import mock

from diagrid.agent.core.chat.client import DaprChatClient, _DEFAULT_COMPONENT_NAME
from diagrid.agent.core.chat.types import (
    ChatMessage,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
)


# Mimic Dapr's RegisteredComponents namedtuple
RegisteredComponents = namedtuple(
    "RegisteredComponents", ["name", "type", "version", "capabilities"]
)


def _make_metadata_response(components):
    """Create a mock metadata response."""
    resp = mock.MagicMock()
    resp.registered_components = [
        RegisteredComponents(
            name=c["name"],
            type=c["type"],
            version=c.get("version", "v1"),
            capabilities=c.get("capabilities", []),
        )
        for c in components
    ]
    return resp


class TestComponentResolution(unittest.TestCase):
    """Test DaprChatClient component auto-resolution."""

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_explicit_component_name(self, mock_dapr_cls):
        client = DaprChatClient(component_name="my-llm")
        self.assertEqual(client.component_name, "my-llm")
        # get_metadata should NOT be called
        mock_dapr_cls.return_value.get_metadata.assert_not_called()

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_auto_detect_llm_provider(self, mock_dapr_cls):
        mock_dapr_cls.return_value.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "statestore", "type": "state.redis"},
                {"name": "llm-provider", "type": "conversation.openai"},
            ]
        )
        client = DaprChatClient()
        self.assertEqual(client.component_name, "llm-provider")

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_auto_detect_single_component(self, mock_dapr_cls):
        mock_dapr_cls.return_value.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "my-custom-llm", "type": "conversation.anthropic"},
            ]
        )
        client = DaprChatClient()
        self.assertEqual(client.component_name, "my-custom-llm")

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_fallback_to_default(self, mock_dapr_cls):
        mock_dapr_cls.return_value.get_metadata.return_value = _make_metadata_response(
            [
                {"name": "statestore", "type": "state.redis"},
            ]
        )
        client = DaprChatClient()
        self.assertEqual(client.component_name, _DEFAULT_COMPONENT_NAME)

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_fallback_on_metadata_error(self, mock_dapr_cls):
        mock_dapr_cls.return_value.get_metadata.side_effect = Exception("no sidecar")
        client = DaprChatClient()
        self.assertEqual(client.component_name, _DEFAULT_COMPONENT_NAME)


class TestMessageConversion(unittest.TestCase):
    """Test ChatMessage -> Dapr ConversationMessage conversion."""

    def test_system_message(self):
        msg = ChatMessage(role=ChatRole.SYSTEM, content="You are helpful.")
        dapr_msg = DaprChatClient._to_dapr_message(msg)
        self.assertIsNotNone(dapr_msg.of_system)
        self.assertEqual(dapr_msg.of_system.content[0].text, "You are helpful.")

    def test_user_message(self):
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        dapr_msg = DaprChatClient._to_dapr_message(msg)
        self.assertIsNotNone(dapr_msg.of_user)
        self.assertEqual(dapr_msg.of_user.content[0].text, "Hello")

    def test_assistant_message_with_tool_calls(self):
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatToolCall(id="c1", name="search", arguments='{"q": "test"}')
            ],
        )
        dapr_msg = DaprChatClient._to_dapr_message(msg)
        self.assertIsNotNone(dapr_msg.of_assistant)
        self.assertEqual(len(dapr_msg.of_assistant.tool_calls), 1)
        self.assertEqual(dapr_msg.of_assistant.tool_calls[0].id, "c1")
        self.assertEqual(dapr_msg.of_assistant.tool_calls[0].function.name, "search")

    def test_assistant_message_with_content(self):
        msg = ChatMessage(role=ChatRole.ASSISTANT, content="Sure!")
        dapr_msg = DaprChatClient._to_dapr_message(msg)
        self.assertIsNotNone(dapr_msg.of_assistant)
        self.assertEqual(len(dapr_msg.of_assistant.content), 1)
        self.assertEqual(dapr_msg.of_assistant.content[0].text, "Sure!")

    def test_tool_message(self):
        msg = ChatMessage(
            role=ChatRole.TOOL,
            content="result data",
            tool_call_id="c1",
            name="search",
        )
        dapr_msg = DaprChatClient._to_dapr_message(msg)
        self.assertIsNotNone(dapr_msg.of_tool)
        self.assertEqual(dapr_msg.of_tool.tool_id, "c1")
        self.assertEqual(dapr_msg.of_tool.name, "search")
        self.assertEqual(dapr_msg.of_tool.content[0].text, "result data")


class TestToolConversion(unittest.TestCase):
    """Test ChatToolDefinition -> Dapr ConversationTools conversion."""

    def test_basic_tool(self):
        td = ChatToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        dapr_tool = DaprChatClient._to_dapr_tool(td)
        self.assertEqual(dapr_tool.function.name, "search")
        self.assertEqual(dapr_tool.function.description, "Search the web")
        self.assertIsNotNone(dapr_tool.function.parameters)


class TestChat(unittest.TestCase):
    """Test DaprChatClient.chat() with mocked converse_alpha2."""

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_chat_final_response(self, mock_dapr_cls):
        # Mock the response
        mock_choice = mock.MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "Hello from Dapr!"
        mock_choice.message.tool_calls = []

        mock_output = mock.MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = mock.MagicMock()
        mock_response.context_id = "ctx-1"
        mock_response.outputs = [mock_output]

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        client = DaprChatClient(component_name="test-llm")
        response = client.chat(messages=[ChatMessage(role=ChatRole.USER, content="Hi")])

        self.assertEqual(response.content, "Hello from Dapr!")
        self.assertTrue(response.is_final)
        self.assertEqual(response.context_id, "ctx-1")
        self.assertEqual(response.finish_reason, "stop")

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_chat_with_tool_calls(self, mock_dapr_cls):
        mock_tc_func = mock.MagicMock()
        mock_tc_func.name = "search"
        mock_tc_func.arguments = '{"q": "test"}'

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

        client = DaprChatClient(component_name="test-llm")
        response = client.chat(
            messages=[ChatMessage(role=ChatRole.USER, content="Search for test")],
            tools=[
                ChatToolDefinition(
                    name="search",
                    description="Search",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )

        self.assertFalse(response.is_final)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "call_1")
        self.assertEqual(response.tool_calls[0].name, "search")

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_chat_empty_response(self, mock_dapr_cls):
        mock_response = mock.MagicMock()
        mock_response.context_id = None
        mock_response.outputs = []

        mock_dapr_cls.return_value.converse_alpha2.return_value = mock_response

        client = DaprChatClient(component_name="test-llm")
        response = client.chat(messages=[ChatMessage(role=ChatRole.USER, content="Hi")])

        self.assertIsNone(response.content)
        self.assertTrue(response.is_final)


class TestClientLifecycle(unittest.TestCase):
    """Test DaprChatClient connection lifecycle."""

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_close(self, mock_dapr_cls):
        client = DaprChatClient(component_name="test-llm")
        client.close()
        mock_dapr_cls.return_value.close.assert_called_once()

    @mock.patch("diagrid.agent.core.chat.client.DaprClient")
    def test_lazy_resolution_only_once(self, mock_dapr_cls):
        mock_dapr_cls.return_value.get_metadata.return_value = _make_metadata_response(
            [{"name": "llm-provider", "type": "conversation.openai"}]
        )
        client = DaprChatClient()
        _ = client.component_name
        _ = client.component_name
        # get_metadata should only be called once
        mock_dapr_cls.return_value.get_metadata.assert_called_once()


if __name__ == "__main__":
    unittest.main()
