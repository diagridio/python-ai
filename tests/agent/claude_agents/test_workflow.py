# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Tests for the claude_agents workflow + activities."""

import pytest

from diagrid.agent.claude_agents.models import (
    AgentConfig,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
)
from diagrid.agent.claude_agents.workflow import (
    _build_anthropic_messages,
    _build_anthropic_tools,
    _execute_tool,
    _serialize_tool_result,
    clear_tool_registry,
    get_registered_tool,
    get_tool_definition,
    register_tool,
)


class TestToolRegistry:
    def setup_method(self):
        clear_tool_registry()

    def teardown_method(self):
        clear_tool_registry()

    def test_register_and_get_tool(self):
        def my_tool(x: str) -> str:
            return f"result: {x}"

        tool_def = ToolDefinition(name="my_tool", description="A test tool")
        register_tool("my_tool", my_tool, tool_def)

        retrieved = get_registered_tool("my_tool")
        assert retrieved is my_tool

        retrieved_def = get_tool_definition("my_tool")
        assert retrieved_def is not None
        assert retrieved_def.name == "my_tool"

    def test_get_nonexistent_tool(self):
        assert get_registered_tool("nonexistent") is None

    def test_clear_registry(self):
        def my_tool():
            return None

        register_tool("my_tool", my_tool)
        assert get_registered_tool("my_tool") is not None

        clear_tool_registry()
        assert get_registered_tool("my_tool") is None

    def test_register_without_definition(self):
        def my_tool():
            return None

        register_tool("my_tool", my_tool)
        assert get_registered_tool("my_tool") is my_tool
        assert get_tool_definition("my_tool") is None


class TestExecuteTool:
    def test_execute_sync_callable(self):
        def add(a: int, b: int) -> int:
            return a + b

        result = _execute_tool(add, {"a": 1, "b": 2})
        assert result == 3

    def test_execute_async_callable(self):
        async def async_add(a: int, b: int) -> int:
            return a + b

        result = _execute_tool(async_add, {"a": 3, "b": 4})
        assert result == 7

    def test_execute_sdk_mcp_tool_handler(self):
        """An SdkMcpTool from claude_agent_sdk.tool exposes an async handler;
        the activity must invoke it via that handler attribute."""

        async def handler(args):
            return {"content": [{"type": "text", "text": f"hello {args['name']}"}]}

        class FakeMcpTool:
            def __init__(self):
                self.name = "greet"
                self.description = "Greet someone"
                self.input_schema = {"name": str}
                self.handler = handler

        result = _execute_tool(FakeMcpTool(), {"name": "world"})
        assert result == {"content": [{"type": "text", "text": "hello world"}]}

    def test_execute_non_callable_raises(self):
        class NotCallable:
            pass

        with pytest.raises(TypeError, match="not callable"):
            _execute_tool(NotCallable(), {})


class TestSerializeToolResult:
    def test_serialize_primitives(self):
        assert _serialize_tool_result("text") == "text"
        assert _serialize_tool_result(42) == 42
        assert _serialize_tool_result(3.14) == 3.14
        assert _serialize_tool_result(True) is True
        assert _serialize_tool_result(None) is None
        assert _serialize_tool_result([1, 2]) == [1, 2]
        assert _serialize_tool_result({"k": "v"}) == {"k": "v"}

    def test_serialize_pydantic_model(self):
        class FakeModel:
            def model_dump(self):
                return {"k": "v"}

        assert _serialize_tool_result(FakeModel()) == {"k": "v"}

    def test_serialize_to_dict_object(self):
        class HasToDict:
            def to_dict(self):
                return {"a": 1}

        assert _serialize_tool_result(HasToDict()) == {"a": 1}

    def test_serialize_unknown_falls_back_to_str(self):
        class NoDumper:
            def __repr__(self) -> str:
                return "<no-dumper>"

        assert _serialize_tool_result(NoDumper()) == "<no-dumper>"


class TestBuildAnthropicMessages:
    def test_user_message(self):
        msgs = _build_anthropic_messages(
            [Message(role=MessageRole.USER, content="Hello")]
        )
        assert msgs == [{"role": "user", "content": "Hello"}]

    def test_assistant_message_with_tool_use(self):
        tc = ToolCall(id="toolu_1", name="search", args={"q": "x"})
        msgs = _build_anthropic_messages(
            [
                Message(
                    role=MessageRole.ASSISTANT,
                    content="Let me search.",
                    tool_calls=[tc],
                )
            ]
        )
        assert msgs[0]["role"] == "assistant"
        blocks = msgs[0]["content"]
        assert {"type": "text", "text": "Let me search."} in blocks
        assert any(
            b.get("type") == "tool_use" and b.get("id") == "toolu_1" for b in blocks
        )

    def test_tool_results_merged_into_user_turn(self):
        """Multiple consecutive tool messages must collapse into one user
        message whose content is a list of tool_result blocks — that's the
        Anthropic Messages API contract."""
        msgs = _build_anthropic_messages(
            [
                Message(
                    role=MessageRole.TOOL,
                    content="result A",
                    tool_call_id="toolu_a",
                ),
                Message(
                    role=MessageRole.TOOL,
                    content="result B",
                    tool_call_id="toolu_b",
                ),
            ]
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert len(msgs[0]["content"]) == 2
        assert msgs[0]["content"][0]["type"] == "tool_result"
        assert msgs[0]["content"][0]["tool_use_id"] == "toolu_a"
        assert msgs[0]["content"][1]["tool_use_id"] == "toolu_b"

    def test_full_conversation_ordering(self):
        tc = ToolCall(id="toolu_1", name="t", args={})
        msgs = _build_anthropic_messages(
            [
                Message(role=MessageRole.USER, content="hi"),
                Message(role=MessageRole.ASSISTANT, content="ok", tool_calls=[tc]),
                Message(role=MessageRole.TOOL, content="42", tool_call_id="toolu_1"),
                Message(role=MessageRole.ASSISTANT, content="done"),
            ]
        )
        # user, assistant(tool_use), user(tool_result), assistant
        assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


class TestBuildAnthropicTools:
    def test_passes_through_existing_schema(self):
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        td = ToolDefinition(name="search", description="Search", parameters=schema)
        tools = _build_anthropic_tools([td])
        assert tools == [
            {"name": "search", "description": "Search", "input_schema": schema}
        ]

    def test_defaults_to_empty_object_schema(self):
        td = ToolDefinition(
            name="get_time", description="Get the time", parameters=None
        )
        tools = _build_anthropic_tools([td])
        assert tools == [
            {
                "name": "get_time",
                "description": "Get the time",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]


class TestAgentConfigWiring:
    def test_agent_config_carries_defaults(self):
        ac = AgentConfig(name="agent", system_prompt="hello", model="claude-sonnet-4-6")
        # max_tokens must default sensibly so the activity always sends a value
        assert ac.max_tokens == 4096
