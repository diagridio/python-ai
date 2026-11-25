# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Messages reach a node as LangGraph hands them over, on any add_messages channel."""

from typing import Annotated, Any, List

from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _clear_dapr_registration, _run, _RunnerTestCase


def _seen_graph(channel: str, schema: Any) -> StateGraph:
    """A one-node graph whose output ``seen`` lists what the node got on ``channel``."""

    def node(state: Any) -> dict:
        got = state[channel]
        return {"seen": [f"{type(m).__name__}:{m.content}" for m in got]}

    graph = StateGraph(schema)
    graph.add_node("a", node)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    return graph


class _Seen(MessagesState):
    seen: List[str]


class _History(TypedDict, total=False):
    history: Annotated[List[AnyMessage], add_messages]
    seen: List[str]


class TestMessageInputShapes(_RunnerTestCase):
    def _both(self, graph: StateGraph, graph_input: dict):
        _clear_dapr_registration()
        compiled = graph.compile()
        native = compiled.invoke(graph_input)
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name="shapes"), graph_input)
        self.assertEqual(out.status, "completed", out.error)
        return native, out

    def test_every_shape_add_messages_accepts_reaches_the_node_as_a_message(self):
        shapes = {
            "string": "hi",
            "pair": ("user", "hi"),
            "message": HumanMessage("hi"),
            "list of string": ["hi"],
            "dict": {"role": "user", "content": "hi"},
            "list of pair": [("user", "hi")],
        }
        for label, value in shapes.items():
            with self.subTest(label):
                native, out = self._both(
                    _seen_graph("messages", _Seen), {"messages": value}
                )
                self.assertEqual(out.output["seen"], native["seen"])
                self.assertEqual(out.output["seen"], ["HumanMessage:hi"])

    def test_a_channel_not_named_messages_reaches_the_node_as_messages(self):
        native, out = self._both(
            _seen_graph("history", _History), {"history": HumanMessage("hi")}
        )
        self.assertEqual(out.output["seen"], native["seen"])
        self.assertEqual(out.output["seen"], ["HumanMessage:hi"])

    def test_the_output_messages_of_such_a_channel_have_an_id(self):
        _, out = self._both(
            _seen_graph("history", _History), {"history": [HumanMessage("hi")]}
        )
        ids = [m.get("id") for m in out.output.get("history", [])]
        self.assertTrue(ids and all(isinstance(i, str) and i for i in ids), ids)
