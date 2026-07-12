# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

import unittest

from diagrid.agent.langchain.models import (
    AgentConfig,
    AgentWorkflowInput,
    AgentWorkflowOutput,
    CallLlmInput,
    CallLlmOutput,
    ExecuteToolInput,
    ExecuteToolOutput,
    Message,
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

    def test_to_langchain_tool_call(self):
        tc = ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"})
        lc_call = tc.to_langchain_tool_call()

        self.assertEqual(lc_call["type"], "tool_call")
        self.assertEqual(lc_call["id"], "call_1")
        self.assertEqual(lc_call["name"], "get_weather")
        self.assertEqual(lc_call["args"], {"city": "Tokyo"})


class TestMessage(unittest.TestCase):
    """Tests for Message dataclass."""

    def test_message_serialization(self):
        msg = Message(role="user", content="Hello, world!")
        data = msg.to_dict()
        restored = Message.from_dict(data)

        self.assertEqual(restored.role, "user")
        self.assertEqual(restored.content, "Hello, world!")
        self.assertEqual(restored.tool_calls, [])
        self.assertIsNone(restored.tool_call_id)

    def test_message_with_tool_calls(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="get_weather", args={"city": "Tokyo"}),
                ToolCall(id="call_2", name="get_time", args={"timezone": "JST"}),
            ],
        )
        restored = Message.from_dict(msg.to_dict())

        self.assertEqual(restored.role, "assistant")
        self.assertIsNone(restored.content)
        self.assertEqual(len(restored.tool_calls), 2)
        self.assertEqual(restored.tool_calls[0].name, "get_weather")
        self.assertEqual(restored.tool_calls[1].args, {"timezone": "JST"})

    def test_tool_message(self):
        msg = Message(
            role="tool",
            content="Sunny, 25C",
            tool_call_id="call_1",
            name="get_weather",
        )
        restored = Message.from_dict(msg.to_dict())

        self.assertEqual(restored.role, "tool")
        self.assertEqual(restored.tool_call_id, "call_1")
        self.assertEqual(restored.name, "get_weather")


class TestToolDefinition(unittest.TestCase):
    """Tests for ToolDefinition dataclass."""

    def test_serialization(self):
        tool_def = ToolDefinition(
            name="get_weather",
            description="Get the weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        restored = ToolDefinition.from_dict(tool_def.to_dict())

        self.assertEqual(restored.name, "get_weather")
        self.assertEqual(restored.parameters["type"], "object")


class TestAgentConfig(unittest.TestCase):
    """Tests for AgentConfig dataclass."""

    def test_serialization(self):
        config = AgentConfig(
            name="weather_agent",
            model_key="WeatherAgent",
            system_prompt="You are a helpful weather assistant.",
            tool_definitions=[
                ToolDefinition(
                    name="get_weather", description="Get weather", parameters={}
                ),
            ],
        )
        restored = AgentConfig.from_dict(config.to_dict())

        self.assertEqual(restored.name, "weather_agent")
        self.assertEqual(restored.model_key, "WeatherAgent")
        self.assertEqual(restored.system_prompt, "You are a helpful weather assistant.")
        self.assertEqual(len(restored.tool_definitions), 1)


class TestAgentWorkflowInput(unittest.TestCase):
    """Tests for AgentWorkflowInput dataclass."""

    def test_serialization(self):
        workflow_input = AgentWorkflowInput(
            agent_config=AgentConfig(name="test_agent", model_key="TestAgent"),
            messages=[Message(role="user", content="Hello!")],
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
            final_response="The weather in Tokyo is sunny.",
            messages=[Message(role="user", content="What's the weather?")],
            iterations=2,
            status="completed",
        )
        restored = AgentWorkflowOutput.from_dict(output.to_dict())

        self.assertEqual(restored.final_response, "The weather in Tokyo is sunny.")
        self.assertEqual(restored.iterations, 2)
        self.assertIsNone(restored.error)

    def test_error(self):
        output = AgentWorkflowOutput(
            final_response=None,
            messages=[],
            iterations=0,
            status="error",
            error="LLM API error",
        )
        restored = AgentWorkflowOutput.from_dict(output.to_dict())

        self.assertIsNone(restored.final_response)
        self.assertEqual(restored.status, "error")
        self.assertEqual(restored.error, "LLM API error")


class TestCallLlmInputOutput(unittest.TestCase):
    """Tests for CallLlmInput/CallLlmOutput dataclasses."""

    def test_input_serialization(self):
        llm_input = CallLlmInput(
            agent_config=AgentConfig(name="test_agent", model_key="TestAgent"),
            messages=[Message(role="user", content="Hello!")],
        )
        restored = CallLlmInput.from_dict(llm_input.to_dict())

        self.assertEqual(restored.agent_config.name, "test_agent")
        self.assertEqual(len(restored.messages), 1)

    def test_output_serialization(self):
        llm_output = CallLlmOutput(
            message=Message(role="assistant", content="Hi there!"),
            is_final=True,
        )
        restored = CallLlmOutput.from_dict(llm_output.to_dict())

        self.assertEqual(restored.message.content, "Hi there!")
        self.assertTrue(restored.is_final)


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
