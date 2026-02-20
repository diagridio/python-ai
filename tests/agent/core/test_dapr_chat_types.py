# -*- coding: utf-8 -*-

"""
Copyright 2026 The Dapr Authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest

from diagrid.agent.core.chat.types import (
    ChatMessage,
    ChatResponse,
    ChatRole,
    ChatToolCall,
    ChatToolDefinition,
)


class TestChatRole(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ChatRole.SYSTEM.value, "system")
        self.assertEqual(ChatRole.USER.value, "user")
        self.assertEqual(ChatRole.ASSISTANT.value, "assistant")
        self.assertEqual(ChatRole.TOOL.value, "tool")


class TestChatToolCall(unittest.TestCase):
    def test_roundtrip(self):
        tc = ChatToolCall(id="call_1", name="search", arguments='{"q": "hello"}')
        data = tc.to_dict()
        tc2 = ChatToolCall.from_dict(data)
        self.assertEqual(tc2.id, "call_1")
        self.assertEqual(tc2.name, "search")
        self.assertEqual(tc2.arguments, '{"q": "hello"}')


class TestChatToolDefinition(unittest.TestCase):
    def test_roundtrip(self):
        td = ChatToolDefinition(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        data = td.to_dict()
        td2 = ChatToolDefinition.from_dict(data)
        self.assertEqual(td2.name, "search")
        self.assertEqual(td2.description, "Search the web")
        self.assertIsNotNone(td2.parameters)

    def test_roundtrip_no_parameters(self):
        td = ChatToolDefinition(name="ping", description="Ping")
        data = td.to_dict()
        td2 = ChatToolDefinition.from_dict(data)
        self.assertIsNone(td2.parameters)


class TestChatMessage(unittest.TestCase):
    def test_user_message_roundtrip(self):
        msg = ChatMessage(role=ChatRole.USER, content="Hello")
        data = msg.to_dict()
        msg2 = ChatMessage.from_dict(data)
        self.assertEqual(msg2.role, ChatRole.USER)
        self.assertEqual(msg2.content, "Hello")
        self.assertEqual(msg2.tool_calls, [])

    def test_assistant_message_with_tool_calls(self):
        msg = ChatMessage(
            role=ChatRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatToolCall(id="c1", name="search", arguments='{"q": "test"}'),
                ChatToolCall(id="c2", name="fetch", arguments='{"url": "http://x"}'),
            ],
        )
        data = msg.to_dict()
        msg2 = ChatMessage.from_dict(data)
        self.assertEqual(msg2.role, ChatRole.ASSISTANT)
        self.assertIsNone(msg2.content)
        self.assertEqual(len(msg2.tool_calls), 2)
        self.assertEqual(msg2.tool_calls[0].name, "search")
        self.assertEqual(msg2.tool_calls[1].name, "fetch")

    def test_tool_message_roundtrip(self):
        msg = ChatMessage(
            role=ChatRole.TOOL,
            content="result data",
            tool_call_id="c1",
            name="search",
        )
        data = msg.to_dict()
        msg2 = ChatMessage.from_dict(data)
        self.assertEqual(msg2.role, ChatRole.TOOL)
        self.assertEqual(msg2.tool_call_id, "c1")
        self.assertEqual(msg2.name, "search")

    def test_system_message_roundtrip(self):
        msg = ChatMessage(role=ChatRole.SYSTEM, content="You are helpful.")
        data = msg.to_dict()
        msg2 = ChatMessage.from_dict(data)
        self.assertEqual(msg2.role, ChatRole.SYSTEM)
        self.assertEqual(msg2.content, "You are helpful.")


class TestChatResponse(unittest.TestCase):
    def test_final_response(self):
        resp = ChatResponse(content="Hello!", finish_reason="stop")
        self.assertTrue(resp.is_final)

    def test_tool_call_response(self):
        resp = ChatResponse(
            content=None,
            tool_calls=[
                ChatToolCall(id="c1", name="search", arguments='{"q": "test"}')
            ],
            finish_reason="tool_calls",
        )
        self.assertFalse(resp.is_final)

    def test_roundtrip(self):
        resp = ChatResponse(
            content="result",
            tool_calls=[
                ChatToolCall(id="c1", name="search", arguments='{"q": "test"}')
            ],
            finish_reason="tool_calls",
            context_id="ctx-123",
        )
        data = resp.to_dict()
        resp2 = ChatResponse.from_dict(data)
        self.assertEqual(resp2.content, "result")
        self.assertEqual(len(resp2.tool_calls), 1)
        self.assertEqual(resp2.finish_reason, "tool_calls")
        self.assertEqual(resp2.context_id, "ctx-123")

    def test_default_values(self):
        resp = ChatResponse()
        self.assertIsNone(resp.content)
        self.assertEqual(resp.tool_calls, [])
        self.assertEqual(resp.finish_reason, "stop")
        self.assertIsNone(resp.context_id)
        self.assertTrue(resp.is_final)


if __name__ == "__main__":
    unittest.main()
