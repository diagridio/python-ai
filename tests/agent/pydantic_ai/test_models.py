# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for Pydantic AI data models."""

import json
import pytest

from diagrid.agent.pydantic_ai.models import (
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


class TestToolCall:
    def test_to_dict(self):
        tc = ToolCall(id="call_123", name="search", args={"query": "test"})
        result = tc.to_dict()
        assert result == {
            "id": "call_123",
            "name": "search",
            "args": {"query": "test"},
        }

    def test_from_dict(self):
        data = {"id": "call_123", "name": "search", "args": {"query": "test"}}
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.args == {"query": "test"}

    def test_roundtrip(self):
        original = ToolCall(id="call_123", name="search", args={"query": "test"})
        roundtripped = ToolCall.from_dict(original.to_dict())
        assert original.id == roundtripped.id
        assert original.name == roundtripped.name
        assert original.args == roundtripped.args


class TestToolResult:
    def test_to_dict(self):
        tr = ToolResult(
            tool_call_id="call_123",
            tool_name="search",
            result="Found 10 results",
        )
        result = tr.to_dict()
        assert result["tool_call_id"] == "call_123"
        assert result["tool_name"] == "search"
        assert result["result"] == "Found 10 results"
        assert result["error"] is None

    def test_from_dict_with_error(self):
        data = {
            "tool_call_id": "call_123",
            "tool_name": "search",
            "result": None,
            "error": "Tool not found",
        }
        tr = ToolResult.from_dict(data)
        assert tr.error == "Tool not found"
        assert tr.result is None


class TestMessage:
    def test_user_message(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["tool_calls"] == []

    def test_assistant_message_with_tool_calls(self):
        tc = ToolCall(id="call_123", name="search", args={"query": "test"})
        msg = Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[tc],
        )
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "search"

    def test_tool_message(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="Result: success",
            tool_call_id="call_123",
            name="search",
        )
        data = msg.to_dict()
        assert data["role"] == "tool"
        assert data["tool_call_id"] == "call_123"
        assert data["name"] == "search"

    def test_roundtrip(self):
        tc = ToolCall(id="call_123", name="search", args={"query": "test"})
        original = Message(
            role=MessageRole.ASSISTANT,
            content="Let me search",
            tool_calls=[tc],
        )
        roundtripped = Message.from_dict(original.to_dict())
        assert original.role == roundtripped.role
        assert original.content == roundtripped.content
        assert len(original.tool_calls) == len(roundtripped.tool_calls)


class TestToolDefinition:
    def test_to_dict(self):
        td = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        data = td.to_dict()
        assert data["name"] == "search"
        assert data["description"] == "Search the web"
        assert data["parameters"]["type"] == "object"

    def test_from_dict_no_parameters(self):
        data = {"name": "get_time", "description": "Get current time"}
        td = ToolDefinition.from_dict(data)
        assert td.name == "get_time"
        assert td.parameters is None

    def test_roundtrip(self):
        original = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {}},
        )
        roundtripped = ToolDefinition.from_dict(original.to_dict())
        assert original.name == roundtripped.name
        assert original.description == roundtripped.description
        assert original.parameters == roundtripped.parameters


class TestAgentConfig:
    def test_to_dict(self):
        config = AgentConfig(
            name="research_assistant",
            system_prompt="You help users find information.",
            model="gpt-4o-mini",
            tool_definitions=[
                ToolDefinition(name="search", description="Search the web"),
            ],
        )
        data = config.to_dict()
        assert data["name"] == "research_assistant"
        assert data["system_prompt"] == "You help users find information."
        assert data["model"] == "gpt-4o-mini"
        assert len(data["tool_definitions"]) == 1

    def test_from_dict(self):
        data = {
            "name": "assistant",
            "system_prompt": "Help users.",
            "model": "gpt-4",
            "tool_definitions": [],
        }
        config = AgentConfig.from_dict(data)
        assert config.name == "assistant"
        assert config.system_prompt == "Help users."
        assert config.model == "gpt-4"

    def test_roundtrip(self):
        original = AgentConfig(
            name="my_agent",
            system_prompt="Be helpful.",
            model="gpt-4o",
            tool_definitions=[
                ToolDefinition(name="tool1", description="A tool"),
            ],
        )
        roundtripped = AgentConfig.from_dict(original.to_dict())
        assert original.name == roundtripped.name
        assert original.system_prompt == roundtripped.system_prompt
        assert original.model == roundtripped.model
        assert len(original.tool_definitions) == len(roundtripped.tool_definitions)


class TestAgentWorkflowInput:
    def test_json_serializable(self):
        """Test that the entire input can be JSON serialized."""
        agent_config = AgentConfig(
            name="assistant",
            system_prompt="Help users.",
            model="gpt-4",
        )
        workflow_input = AgentWorkflowInput(
            agent_config=agent_config,
            messages=[Message(role=MessageRole.USER, content="Hello")],
            session_id="test-session",
        )

        # Should not raise
        json_str = json.dumps(workflow_input.to_dict())
        assert json_str is not None

        # Should roundtrip
        parsed = AgentWorkflowInput.from_dict(json.loads(json_str))
        assert parsed.session_id == "test-session"
        assert parsed.agent_config.name == "assistant"

    def test_defaults(self):
        agent_config = AgentConfig(
            name="test",
            system_prompt="test",
            model="gpt-4",
        )
        workflow_input = AgentWorkflowInput(
            agent_config=agent_config,
            messages=[],
            session_id="s1",
        )
        assert workflow_input.iteration == 0
        assert workflow_input.max_iterations == 25


class TestAgentWorkflowOutput:
    def test_completed_output(self):
        output = AgentWorkflowOutput(
            final_response="The answer is 42",
            messages=[],
            iterations=3,
            status="completed",
        )
        data = output.to_dict()
        assert data["final_response"] == "The answer is 42"
        assert data["iterations"] == 3
        assert data["status"] == "completed"

    def test_error_output(self):
        output = AgentWorkflowOutput(
            final_response=None,
            messages=[],
            iterations=1,
            status="error",
            error="Something went wrong",
        )
        data = output.to_dict()
        assert data["error"] == "Something went wrong"


class TestCallLlmInput:
    def test_to_dict(self):
        agent_config = AgentConfig(
            name="assistant",
            system_prompt="Help users.",
            model="gpt-4",
        )
        llm_input = CallLlmInput(
            agent_config=agent_config,
            messages=[],
        )
        data = llm_input.to_dict()
        assert "agent_config" in data
        assert "messages" in data

    def test_roundtrip(self):
        agent_config = AgentConfig(
            name="test",
            system_prompt="test",
            model="gpt-4",
        )
        original = CallLlmInput(
            agent_config=agent_config,
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        roundtripped = CallLlmInput.from_dict(original.to_dict())
        assert roundtripped.agent_config.name == "test"
        assert len(roundtripped.messages) == 1


class TestCallLlmOutput:
    def test_final_response(self):
        output = CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT, content="Done"),
            is_final=True,
        )
        data = output.to_dict()
        assert data["is_final"] is True
        assert data["message"]["content"] == "Done"

    def test_with_tool_calls(self):
        tc = ToolCall(id="call_1", name="search", args={"q": "test"})
        output = CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT, tool_calls=[tc]),
            is_final=False,
        )
        data = output.to_dict()
        assert data["is_final"] is False
        assert len(data["message"]["tool_calls"]) == 1


class TestExecuteToolInput:
    def test_to_dict(self):
        tc = ToolCall(id="call_123", name="search", args={"query": "test"})
        tool_input = ExecuteToolInput(
            tool_call=tc,
            agent_name="assistant",
            session_id="test-session",
        )
        data = tool_input.to_dict()
        assert data["tool_call"]["id"] == "call_123"
        assert data["agent_name"] == "assistant"
        assert data["session_id"] == "test-session"

    def test_roundtrip(self):
        tc = ToolCall(id="call_1", name="tool1", args={})
        original = ExecuteToolInput(
            tool_call=tc,
            agent_name="agent",
            session_id="s1",
        )
        roundtripped = ExecuteToolInput.from_dict(original.to_dict())
        assert roundtripped.tool_call.id == "call_1"
        assert roundtripped.agent_name == "agent"


class TestExecuteToolOutput:
    def test_to_dict(self):
        tr = ToolResult(
            tool_call_id="call_1",
            tool_name="search",
            result="found it",
        )
        output = ExecuteToolOutput(tool_result=tr)
        data = output.to_dict()
        assert data["tool_result"]["result"] == "found it"

    def test_roundtrip(self):
        tr = ToolResult(
            tool_call_id="call_1",
            tool_name="tool1",
            result={"key": "value"},
        )
        original = ExecuteToolOutput(tool_result=tr)
        roundtripped = ExecuteToolOutput.from_dict(original.to_dict())
        assert roundtripped.tool_result.tool_call_id == "call_1"
        assert roundtripped.tool_result.result == {"key": "value"}
