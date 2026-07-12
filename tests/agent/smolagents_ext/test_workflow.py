# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest

from diagrid.agent.smolagents.models import ExecuteToolOutput, ToolCall
from diagrid.agent.smolagents.workflow import (
    _coerce_args,
    _format_observation_text,
    _format_tool_calls_text,
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


class TestCoerceArgs(unittest.TestCase):
    """Tests for the _coerce_args helper.

    ``ChatMessageToolCallFunction.arguments`` may come back as a JSON string
    or an already-parsed dict depending on provider/model.
    """

    def test_dict_passthrough(self):
        self.assertEqual(_coerce_args({"city": "Tokyo"}), {"city": "Tokyo"})

    def test_json_string_parsed(self):
        self.assertEqual(_coerce_args('{"city": "Tokyo"}'), {"city": "Tokyo"})

    def test_invalid_json_string_returns_empty_dict(self):
        self.assertEqual(_coerce_args("not json"), {})

    def test_non_dict_json_returns_empty_dict(self):
        self.assertEqual(_coerce_args("[1, 2, 3]"), {})

    def test_other_type_returns_empty_dict(self):
        self.assertEqual(_coerce_args(None), {})
        self.assertEqual(_coerce_args(42), {})


class TestFormatHelpers(unittest.TestCase):
    """Tests for the text-formatting helpers used to build conversation history."""

    def test_format_tool_calls_text_includes_name_and_args(self):
        text = _format_tool_calls_text(
            [ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})]
        )
        self.assertIn("Calling tools:", text)
        self.assertIn("get_weather", text)
        self.assertIn("Tokyo", text)

    def test_format_observation_text_includes_tool_name_and_content(self):
        text = _format_observation_text(
            [
                ExecuteToolOutput(
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    content="Sunny, 25C",
                )
            ]
        )
        self.assertIn("Observation:", text)
        self.assertIn("get_weather", text)
        self.assertIn("Sunny, 25C", text)


if __name__ == "__main__":
    unittest.main()
