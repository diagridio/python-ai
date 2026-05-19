# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for claude_agents data models."""

import json

from diagrid.agent.claude_agents.models import (
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
        tc = ToolCall(id="toolu_123", name="search", args={"query": "test"})
        assert tc.to_dict() == {
            "id": "toolu_123",
            "name": "search",
            "args": {"query": "test"},
        }

    def test_from_dict(self):
        data = {"id": "toolu_123", "name": "search", "args": {"query": "test"}}
        tc = ToolCall.from_dict(data)
        assert tc.id == "toolu_123"
        assert tc.name == "search"
        assert tc.args == {"query": "test"}

    def test_roundtrip(self):
        original = ToolCall(id="toolu_123", name="search", args={"query": "test"})
        roundtripped = ToolCall.from_dict(original.to_dict())
        assert original.id == roundtripped.id
        assert original.name == roundtripped.name
        assert original.args == roundtripped.args


class TestToolResult:
    def test_to_dict(self):
        tr = ToolResult(
            tool_call_id="toolu_123",
            tool_name="search",
            result="Found 10 results",
        )
        data = tr.to_dict()
        assert data["tool_call_id"] == "toolu_123"
        assert data["tool_name"] == "search"
        assert data["result"] == "Found 10 results"
        assert data["error"] is None

    def test_from_dict_with_error(self):
        tr = ToolResult.from_dict(
            {
                "tool_call_id": "toolu_123",
                "tool_name": "search",
                "result": None,
                "error": "Tool not found",
            }
        )
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
        tc = ToolCall(id="toolu_123", name="search", args={"query": "test"})
        msg = Message(role=MessageRole.ASSISTANT, content=None, tool_calls=[tc])
        data = msg.to_dict()
        assert data["role"] == "assistant"
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "search"

    def test_tool_message(self):
        msg = Message(
            role=MessageRole.TOOL,
            content="Result: success",
            tool_call_id="toolu_123",
            name="search",
        )
        data = msg.to_dict()
        assert data["role"] == "tool"
        assert data["tool_call_id"] == "toolu_123"
        assert data["name"] == "search"

    def test_roundtrip_assistant_with_tool_call(self):
        tc = ToolCall(id="toolu_123", name="search", args={"query": "test"})
        original = Message(
            role=MessageRole.ASSISTANT,
            content="Let me search",
            tool_calls=[tc],
        )
        roundtripped = Message.from_dict(original.to_dict())
        assert original.role == roundtripped.role
        assert original.content == roundtripped.content
        assert len(original.tool_calls) == len(roundtripped.tool_calls)
        assert roundtripped.tool_calls[0].id == "toolu_123"


class TestToolDefinition:
    def test_to_dict_with_parameters(self):
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
        td = ToolDefinition.from_dict(
            {"name": "get_time", "description": "Get the time"}
        )
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
    def test_to_dict_defaults(self):
        ac = AgentConfig(
            name="weather-agent",
            system_prompt="You are a weather assistant.",
            model="claude-sonnet-4-6",
        )
        data = ac.to_dict()
        assert data["name"] == "weather-agent"
        assert data["system_prompt"] == "You are a weather assistant."
        assert data["model"] == "claude-sonnet-4-6"
        assert data["max_tokens"] == 4096
        assert data["tool_definitions"] == []

    def test_to_dict_with_tools(self):
        ac = AgentConfig(
            name="agent",
            system_prompt="prompt",
            model="claude-haiku-4-5-20251001",
            tool_definitions=[ToolDefinition(name="t1", description="d1")],
            max_tokens=2048,
        )
        data = ac.to_dict()
        assert len(data["tool_definitions"]) == 1
        assert data["max_tokens"] == 2048

    def test_roundtrip(self):
        original = AgentConfig(
            name="my-agent",
            system_prompt="be helpful",
            model="claude-sonnet-4-6",
            tool_definitions=[ToolDefinition(name="search", description="Search")],
            max_tokens=1024,
        )
        roundtripped = AgentConfig.from_dict(original.to_dict())
        assert original.name == roundtripped.name
        assert original.system_prompt == roundtripped.system_prompt
        assert original.model == roundtripped.model
        assert original.max_tokens == roundtripped.max_tokens
        assert len(original.tool_definitions) == len(roundtripped.tool_definitions)


class TestAgentWorkflowInput:
    def _make_input(self) -> AgentWorkflowInput:
        return AgentWorkflowInput(
            agent_config=AgentConfig(
                name="agent",
                system_prompt="prompt",
                model="claude-sonnet-4-6",
            ),
            messages=[Message(role=MessageRole.USER, content="Hello")],
            session_id="test-session",
        )

    def test_json_serializable(self):
        wfi = self._make_input()
        # Should not raise and must round-trip through JSON
        json_str = json.dumps(wfi.to_dict())
        parsed = AgentWorkflowInput.from_dict(json.loads(json_str))
        assert parsed.session_id == "test-session"
        assert parsed.agent_config.name == "agent"
        assert parsed.messages[0].content == "Hello"

    def test_defaults(self):
        wfi = AgentWorkflowInput(
            agent_config=AgentConfig(
                name="a", system_prompt="b", model="claude-sonnet-4-6"
            ),
            messages=[],
            session_id="s",
        )
        assert wfi.iteration == 0
        assert wfi.max_iterations == 25


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
            error="boom",
        )
        data = output.to_dict()
        assert data["error"] == "boom"

    def test_roundtrip(self):
        original = AgentWorkflowOutput(
            final_response="done",
            messages=[Message(role=MessageRole.ASSISTANT, content="done")],
            iterations=2,
            status="completed",
        )
        roundtripped = AgentWorkflowOutput.from_dict(original.to_dict())
        assert original.final_response == roundtripped.final_response
        assert original.iterations == roundtripped.iterations
        assert original.status == roundtripped.status
        assert roundtripped.messages[0].content == "done"


class TestCallLlmInput:
    def test_roundtrip(self):
        original = CallLlmInput(
            agent_config=AgentConfig(
                name="agent", system_prompt="prompt", model="claude-sonnet-4-6"
            ),
            messages=[Message(role=MessageRole.USER, content="Hi")],
        )
        roundtripped = CallLlmInput.from_dict(original.to_dict())
        assert roundtripped.agent_config.name == "agent"
        assert len(roundtripped.messages) == 1


class TestCallLlmOutput:
    def test_final_response(self):
        output = CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT, content="Done"),
            is_final=True,
            stop_reason="end_turn",
        )
        data = output.to_dict()
        assert data["is_final"] is True
        assert data["stop_reason"] == "end_turn"

    def test_with_tool_calls(self):
        tc = ToolCall(id="toolu_1", name="search", args={"q": "test"})
        output = CallLlmOutput(
            message=Message(role=MessageRole.ASSISTANT, tool_calls=[tc]),
            is_final=False,
            stop_reason="tool_use",
        )
        data = output.to_dict()
        assert data["is_final"] is False
        assert len(data["message"]["tool_calls"]) == 1
        assert data["stop_reason"] == "tool_use"


class TestExecuteToolInput:
    def test_roundtrip(self):
        tc = ToolCall(id="toolu_1", name="tool1", args={})
        original = ExecuteToolInput(tool_call=tc, agent_name="agent", session_id="s1")
        roundtripped = ExecuteToolInput.from_dict(original.to_dict())
        assert roundtripped.tool_call.id == "toolu_1"
        assert roundtripped.agent_name == "agent"


class TestExecuteToolOutput:
    def test_roundtrip(self):
        tr = ToolResult(
            tool_call_id="toolu_1",
            tool_name="tool1",
            result={"key": "value"},
        )
        original = ExecuteToolOutput(tool_result=tr)
        roundtripped = ExecuteToolOutput.from_dict(original.to_dict())
        assert roundtripped.tool_result.tool_call_id == "toolu_1"
        assert roundtripped.tool_result.result == {"key": "value"}
