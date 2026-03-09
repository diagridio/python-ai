# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest

from diagrid.agent.adk.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class TestMessage(unittest.TestCase):
    """Tests for Message dataclass."""

    def test_message_serialization(self):
        """Test Message to_dict and from_dict."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello, world!",
        )
        data = msg.to_dict()
        restored = Message.from_dict(data)

        self.assertEqual(restored.role, MessageRole.USER)
        self.assertEqual(restored.content, "Hello, world!")
        self.assertEqual(restored.tool_calls, [])
        self.assertEqual(restored.tool_results, [])

    def test_message_with_tool_calls(self):
        """Test Message with tool calls."""
        msg = Message(
            role=MessageRole.MODEL,
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"}),
                ToolCall(id="call_2", name="get_time", args={"timezone": "JST"}),
            ],
        )
        data = msg.to_dict()
        restored = Message.from_dict(data)

        self.assertEqual(restored.role, MessageRole.MODEL)
        self.assertIsNone(restored.content)
        self.assertEqual(len(restored.tool_calls), 2)
        self.assertEqual(restored.tool_calls[0].name, "get_weather")
        self.assertEqual(restored.tool_calls[1].args, {"timezone": "JST"})

    def test_message_with_tool_results(self):
        """Test Message with tool results."""
        msg = Message(
            role=MessageRole.USER,
            tool_results=[
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="get_weather",
                    result="Sunny, 25\u00b0C",
                ),
                ToolResult(
                    tool_call_id="call_2",
                    tool_name="get_time",
                    result=None,
                    error="Timezone not found",
                ),
            ],
        )
        data = msg.to_dict()
        restored = Message.from_dict(data)

        self.assertEqual(len(restored.tool_results), 2)
        self.assertEqual(restored.tool_results[0].result, "Sunny, 25\u00b0C")
        self.assertIsNone(restored.tool_results[0].error)
        self.assertEqual(restored.tool_results[1].error, "Timezone not found")


class TestToolDefinition(unittest.TestCase):
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition_serialization(self):
        """Test ToolDefinition to_dict and from_dict."""
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get the weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        )
        data = tool_def.to_dict()
        restored = ToolDefinition.from_dict(data)

        self.assertEqual(restored.name, "get_weather")
        self.assertEqual(restored.description, "Get the weather for a city")
        self.assertIsNotNone(restored.parameters)
        self.assertEqual(restored.parameters["type"], "object")


class TestAgentConfig(unittest.TestCase):
    """Tests for AgentConfig dataclass."""

    def test_agent_config_serialization(self):
        """Test AgentConfig to_dict and from_dict."""
        config = AgentConfig(
            name="weather_agent",
            model="gemini-2.0-flash",
            system_instruction="You are a helpful weather assistant.",
            tool_definitions=[
                ToolDefinition(
                    name="get_weather",
                    description="Get weather for a city",
                ),
            ],
        )
        data = config.to_dict()
        restored = AgentConfig.from_dict(data)

        self.assertEqual(restored.name, "weather_agent")
        self.assertEqual(restored.model, "gemini-2.0-flash")
        self.assertEqual(
            restored.system_instruction, "You are a helpful weather assistant."
        )
        self.assertEqual(len(restored.tool_definitions), 1)


class TestAgentWorkflowInput(unittest.TestCase):
    """Tests for AgentWorkflowInput dataclass."""

    def test_workflow_input_serialization(self):
        """Test AgentWorkflowInput to_dict and from_dict."""
        workflow_input = AgentWorkflowInput(
            agent_config=AgentConfig(
                name="test_agent",
                model="gemini-2.0-flash",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Hello!"),
            ],
            session_id="session-123",
            user_id="user-456",
            iteration=5,
            max_iterations=50,
        )
        data = workflow_input.to_dict()
        restored = AgentWorkflowInput.from_dict(data)

        self.assertEqual(restored.session_id, "session-123")
        self.assertEqual(restored.user_id, "user-456")
        self.assertEqual(restored.iteration, 5)
        self.assertEqual(restored.max_iterations, 50)
        self.assertEqual(len(restored.messages), 1)


class TestAgentWorkflowOutput(unittest.TestCase):
    """Tests for AgentWorkflowOutput dataclass."""

    def test_workflow_output_success(self):
        """Test AgentWorkflowOutput for successful completion."""
        output = AgentWorkflowOutput(
            final_response="The weather in Tokyo is sunny.",
            messages=[
                Message(role=MessageRole.USER, content="What's the weather?"),
                Message(
                    role=MessageRole.MODEL, content="The weather in Tokyo is sunny."
                ),
            ],
            iterations=2,
            status="completed",
        )
        data = output.to_dict()
        restored = AgentWorkflowOutput.from_dict(data)

        self.assertEqual(restored.final_response, "The weather in Tokyo is sunny.")
        self.assertEqual(restored.iterations, 2)
        self.assertEqual(restored.status, "completed")
        self.assertIsNone(restored.error)

    def test_workflow_output_error(self):
        """Test AgentWorkflowOutput for error case."""
        output = AgentWorkflowOutput(
            final_response=None,
            messages=[],
            iterations=0,
            status="error",
            error="LLM API error",
        )
        data = output.to_dict()
        restored = AgentWorkflowOutput.from_dict(data)

        self.assertIsNone(restored.final_response)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.error, "LLM API error")


class TestCallLlmInput(unittest.TestCase):
    """Tests for CallLlmInput dataclass."""

    def test_call_llm_input_serialization(self):
        """Test CallLlmInput to_dict and from_dict."""
        llm_input = CallLlmInput(
            agent_config=AgentConfig(
                name="test_agent",
                model="gemini-2.0-flash",
            ),
            messages=[
                Message(role=MessageRole.USER, content="Hello!"),
            ],
        )
        data = llm_input.to_dict()
        restored = CallLlmInput.from_dict(data)

        self.assertEqual(restored.agent_config.name, "test_agent")
        self.assertEqual(len(restored.messages), 1)


class TestExecuteToolInput(unittest.TestCase):
    """Tests for ExecuteToolInput dataclass."""

    def test_execute_tool_input_serialization(self):
        """Test ExecuteToolInput to_dict and from_dict."""
        tool_input = ExecuteToolInput(
            tool_call=ToolCall(
                id="call_123",
                name="get_weather",
                args={"city": "Tokyo"},
            ),
            agent_name="weather_agent",
            session_id="session-123",
            user_id="user-456",
        )
        data = tool_input.to_dict()
        restored = ExecuteToolInput.from_dict(data)

        self.assertEqual(restored.tool_call.id, "call_123")
        self.assertEqual(restored.tool_call.name, "get_weather")
        self.assertEqual(restored.tool_call.args, {"city": "Tokyo"})
        self.assertEqual(restored.agent_name, "weather_agent")
        self.assertEqual(restored.session_id, "session-123")


if __name__ == "__main__":
    unittest.main()
