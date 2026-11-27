# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A write of Overwrite replaces a reducer channel's value, as in LangGraph."""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Overwrite

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .golden_harness import _Log
from .test_runner import _run, _RunnerTestCase


class TestOverwrite(_RunnerTestCase):
    def _both(self, graph: StateGraph, graph_input: dict):
        compiled = graph.compile()
        native = compiled.invoke(graph_input)
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name="over"), graph_input)
        self.assertEqual(out.status, "completed", out.error)
        return native, out

    def test_overwrite_replaces_the_value_of_a_reducer_channel(self):
        graph = StateGraph(_Log)
        graph.add_node("a", lambda s: {"log": ["a"]})
        graph.add_node("b", lambda s: {"log": Overwrite(["reset"])})
        graph.add_node("c", lambda s: {"log": ["c"]})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)
        native, out = self._both(graph, {})
        self.assertEqual(native["log"], ["reset", "c"])
        self.assertEqual(out.output["log"], native["log"])

    def test_overwrite_of_messages_replaces_the_history(self):
        graph = StateGraph(MessagesState)
        graph.add_node("a", lambda s: {"messages": [AIMessage(content="a")]})
        graph.add_node(
            "b", lambda s: {"messages": Overwrite([AIMessage(content="summary")])}
        )
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        native, out = self._both(graph, {"messages": [("user", "hi")]})
        self.assertEqual(
            [m["content"] for m in out.output["messages"]],
            [m.content for m in native["messages"]],
        )
        self.assertEqual([m["content"] for m in out.output["messages"]], ["summary"])
