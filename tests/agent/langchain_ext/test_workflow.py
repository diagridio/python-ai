# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest

from diagrid.agent.langchain.models import Message, ToolCall
from diagrid.agent.langchain.workflow import (
    _to_langchain_messages,
    clear_registries,
    get_registered_model,
    get_registered_tool,
    register_model,
    register_tool,
)


class MockTool:
    """Mock tool for testing the registry."""

    def __init__(self, name: str):
        self.name = name


class TestModelRegistry(unittest.TestCase):
    """Tests for the model registry functions."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_register_and_get_model(self):
        model = object()
        register_model("my-agent", model)

        self.assertIs(get_registered_model("my-agent"), model)

    def test_get_nonexistent_model(self):
        self.assertIsNone(get_registered_model("nonexistent"))

    def test_clear_registries_clears_model(self):
        register_model("my-agent", object())
        clear_registries()

        self.assertIsNone(get_registered_model("my-agent"))


class TestToolRegistry(unittest.TestCase):
    """Tests for the tool registry functions."""

    def setUp(self):
        clear_registries()

    def tearDown(self):
        clear_registries()

    def test_register_and_get_tool(self):
        tool = MockTool("test_tool")
        register_tool("test_tool", tool)

        retrieved = get_registered_tool("test_tool")
        self.assertEqual(retrieved, tool)
        self.assertEqual(retrieved.name, "test_tool")

    def test_get_nonexistent_tool(self):
        self.assertIsNone(get_registered_tool("nonexistent"))

    def test_clear_registry(self):
        register_tool("tool1", MockTool("tool1"))
        register_tool("tool2", MockTool("tool2"))

        clear_registries()

        self.assertIsNone(get_registered_tool("tool1"))
        self.assertIsNone(get_registered_tool("tool2"))

    def test_overwrite_tool(self):
        register_tool("my_tool", MockTool("original"))
        self.assertEqual(get_registered_tool("my_tool").name, "original")

        register_tool("my_tool", MockTool("replacement"))
        self.assertEqual(get_registered_tool("my_tool").name, "replacement")


class TestToLangchainMessages(unittest.TestCase):
    """Tests for the internal Message -> langchain_core message conversion."""

    def test_converts_all_roles(self):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What's the weather?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})
                ],
            ),
            Message(
                role="tool",
                content="Sunny",
                tool_call_id="call_1",
                name="get_weather",
            ),
        ]

        lc_messages = _to_langchain_messages(messages)

        self.assertIsInstance(lc_messages[0], SystemMessage)
        self.assertIsInstance(lc_messages[1], HumanMessage)
        self.assertIsInstance(lc_messages[2], AIMessage)
        self.assertEqual(lc_messages[2].tool_calls[0]["name"], "get_weather")
        self.assertIsInstance(lc_messages[3], ToolMessage)
        self.assertEqual(lc_messages[3].tool_call_id, "call_1")

    def test_unknown_role_raises(self):
        with self.assertRaises(ValueError):
            _to_langchain_messages([Message(role="bogus", content="x")])


if __name__ == "__main__":
    unittest.main()
