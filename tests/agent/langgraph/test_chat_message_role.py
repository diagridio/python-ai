# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A ``ChatMessage`` has a ``role`` of its own, and is still a message dump.

Its dump holds ``type: "chat"`` and ``role: "critic"``. A ``{"role", "content"}``
dict (no ``type``) is the shorthand ``convert_to_messages`` reads, and must keep
working; a dump with a ``type`` is validated as the message class.
"""

from typing import Any

from langchain_core.messages import ChatMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import _as_message, _serialize_value

from .test_runner import _RunnerTestCase, _run


def _graph() -> Any:
    graph = StateGraph(MessagesState)
    graph.add_node(
        "critic", lambda s: {"messages": [ChatMessage(content="c", role="critic")]}
    )
    graph.add_edge(START, "critic")
    graph.add_edge("critic", END)
    return graph.compile()


class TestChatMessageRole(_RunnerTestCase):
    def test_a_chat_message_dump_is_a_chat_message_again(self):
        back = _as_message(_serialize_value(ChatMessage(content="c", role="critic")))
        self.assertIsInstance(back, ChatMessage)
        self.assertEqual((back.role, back.content), ("critic", "c"))

    def test_a_role_dict_is_still_the_shorthand(self):
        back = _as_message({"role": "user", "content": "hi"})
        self.assertEqual((back.type, back.content), ("human", "hi"))

    def test_a_node_may_write_a_chat_message_to_an_add_messages_channel(self):
        compiled = _graph()
        native = compiled.invoke({"messages": [("user", "hi")]})["messages"]
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="chat-role"),
            {"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(out.status, "completed", out.error)
        got = [(m["type"], m["content"], m.get("role")) for m in out.output["messages"]]
        want = [(m.type, m.content, getattr(m, "role", None)) for m in native]
        self.assertEqual(got, want)
