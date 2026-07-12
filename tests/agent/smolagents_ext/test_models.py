# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest

from diagrid.agent.smolagents.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallModelInput,
    CallModelOutput,
    ChatEntry,
    ExecuteToolInput,
    ExecuteToolOutput,
    ToolCall,
    ToolDefinition,
)


class TestToolCall(unittest.TestCase):
    """Tests for ToolCall dataclass."""

    def test_serialization(self):
        tc = ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})
        restored = ToolCall.from_dict(tc.to_dict())

        self.assertEqual(restored.id, "call_1")
        self.assertEqual(restored.name, "get_weather")
        self.assertEqual(restored.args, {"city": "Tokyo"})


class TestChatEntry(unittest.TestCase):
    """Tests for ChatEntry dataclass."""

    def test_serialization(self):
        entry = ChatEntry(role="user", content="Hello, world!")
        restored = ChatEntry.from_dict(entry.to_dict())

        self.assertEqual(restored.role, "user")
        self.assertEqual(restored.content, "Hello, world!")

    def test_to_model_message_wraps_content_in_block_list(self):
        entry = ChatEntry(role="assistant", content="Calling tools:\n[...]")
        message = entry.to_model_message()

        self.assertEqual(message["role"], "assistant")
        self.assertIsInstance(message["content"], list)
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "Calling tools:\n[...]")


class TestToolDefinition(unittest.TestCase):
    """Tests for ToolDefinition dataclass."""

    def test_serialization(self):
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get the weather for a city",
            inputs={"city": {"type": "string", "description": "The city"}},
            output_type="string",
        )
        restored = ToolDefinition.from_dict(tool_def.to_dict())

        self.assertEqual(restored.name, "get_weather")
        self.assertEqual(restored.inputs["city"]["type"], "string")
        self.assertEqual(restored.output_type, "string")


class TestAgentConfig(unittest.TestCase):
    """Tests for AgentConfig dataclass."""

    def test_serialization(self):
        config = AgentConfig(
            name="weather_agent",
            model_key="WeatherAgent",
            tool_definitions=[
                ToolDefinition(
                    name="get_weather", description="Get weather", inputs={}
                ),
            ],
        )
        restored = AgentConfig.from_dict(config.to_dict())

        self.assertEqual(restored.name, "weather_agent")
        self.assertEqual(restored.model_key, "WeatherAgent")
        self.assertEqual(len(restored.tool_definitions), 1)


class TestAgentWorkflowInput(unittest.TestCase):
    """Tests for AgentWorkflowInput dataclass."""

    def test_serialization(self):
        workflow_input = AgentWorkflowInput(
            agent_config=AgentConfig(name="test_agent", model_key="TestAgent"),
            messages=[ChatEntry(role="user", content="Hello!")],
            session_id="session-123",
            iteration=5,
            max_iterations=50,
        )
        restored = AgentWorkflowInput.from_dict(workflow_input.to_dict())

        self.assertEqual(restored.session_id, "session-123")
        self.assertEqual(restored.iteration, 5)
        self.assertEqual(restored.max_iterations, 50)
        self.assertEqual(len(restored.messages), 1)


class TestAgentWorkflowOutput(unittest.TestCase):
    """Tests for AgentWorkflowOutput dataclass."""

    def test_success(self):
        output = AgentWorkflowOutput(
            final_answer="The weather in Tokyo is sunny.",
            messages=[ChatEntry(role="user", content="What's the weather?")],
            iterations=2,
            status="completed",
        )
        restored = AgentWorkflowOutput.from_dict(output.to_dict())

        self.assertEqual(restored.final_answer, "The weather in Tokyo is sunny.")
        self.assertEqual(restored.iterations, 2)
        self.assertIsNone(restored.error)

    def test_error(self):
        output = AgentWorkflowOutput(
            final_answer=None,
            messages=[],
            iterations=0,
            status="error",
            error="model error",
        )
        restored = AgentWorkflowOutput.from_dict(output.to_dict())

        self.assertIsNone(restored.final_answer)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.error, "model error")


class TestCallModelInputOutput(unittest.TestCase):
    """Tests for CallModelInput/CallModelOutput dataclasses."""

    def test_input_serialization(self):
        model_input = CallModelInput(
            agent_config=AgentConfig(name="test_agent", model_key="TestAgent"),
            messages=[ChatEntry(role="user", content="Hello!")],
        )
        restored = CallModelInput.from_dict(model_input.to_dict())

        self.assertEqual(restored.agent_config.name, "test_agent")
        self.assertEqual(len(restored.messages), 1)

    def test_output_serialization(self):
        model_output = CallModelOutput(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})
            ],
        )
        restored = CallModelOutput.from_dict(model_output.to_dict())

        self.assertEqual(len(restored.tool_calls), 1)
        self.assertEqual(restored.tool_calls[0].name, "get_weather")


class TestExecuteToolInputOutput(unittest.TestCase):
    """Tests for ExecuteToolInput/ExecuteToolOutput dataclasses."""

    def test_input_serialization(self):
        tool_input = ExecuteToolInput(
            tool_call=ToolCall(
                id="call_123", name="get_weather", args={"city": "Tokyo"}
            ),
        )
        restored = ExecuteToolInput.from_dict(tool_input.to_dict())

        self.assertEqual(restored.tool_call.id, "call_123")
        self.assertEqual(restored.tool_call.args, {"city": "Tokyo"})

    def test_output_serialization(self):
        tool_output = ExecuteToolOutput(
            tool_call_id="call_123",
            tool_name="get_weather",
            content="Sunny, 25C",
        )
        restored = ExecuteToolOutput.from_dict(tool_output.to_dict())

        self.assertEqual(restored.tool_call_id, "call_123")
        self.assertFalse(restored.is_error)


if __name__ == "__main__":
    unittest.main()
