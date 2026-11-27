# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A reducer channel nobody has written holds its default, as it does natively."""

import operator
from typing import Annotated, Any, Dict

from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _run, _RunnerTestCase


def _merge(old: dict, new: dict) -> dict:
    return {**old, **new}


class _State(TypedDict, total=False):
    docs: Annotated[list, operator.add]
    n: Annotated[int, operator.add]
    d: Annotated[dict, _merge]
    q: str


class TestUnwrittenReducerChannels(_RunnerTestCase):
    def _both(self, graph: StateGraph, graph_input: Dict[str, Any]):
        compiled = graph.compile()
        native = compiled.invoke(graph_input)
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="defaults"), graph_input
        )
        return native, out

    def test_a_node_reads_a_channel_nobody_wrote(self):
        graph = StateGraph(_State)
        graph.add_node("a", lambda s: {"docs": s["docs"] + ["a"], "n": s["n"] + 1})
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        native, out = self._both(graph, {"q": "x"})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["docs"], native["docs"])
        self.assertEqual(out.output["n"], native["n"])

    def test_a_router_reads_a_channel_nobody_wrote(self):
        graph = StateGraph(_State)
        graph.add_node("a", lambda s: {"q": "y"})
        graph.add_node("b", lambda s: {"docs": ["b"]})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", lambda s: "b" if not s["docs"] else END)
        graph.add_edge("b", END)
        native, out = self._both(graph, {"q": "x"})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["docs"], native["docs"])

    def test_a_messages_graph_started_without_messages(self):
        graph = StateGraph(MessagesState)
        graph.add_node(
            "a", lambda s: {"messages": [("assistant", str(len(s["messages"])))]}
        )
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        native, out = self._both(graph, {})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(
            [m["content"] for m in out.output["messages"]],
            [m.content for m in native["messages"]],
        )

    def test_the_output_has_the_channels_nobody_wrote(self):
        graph = StateGraph(_State)
        graph.add_node("a", lambda s: {"q": "y"})
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        native, out = self._both(graph, {"q": "x"})
        self.assertEqual(native, {"docs": [], "n": 0, "d": {}, "q": "y"})
        self.assertEqual(out.output, native)
