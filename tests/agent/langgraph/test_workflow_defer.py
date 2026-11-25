# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``add_node(..., defer=True)``: the node runs once, when nothing else is pending."""

import operator
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import GraphConfig

from .test_runner import _clear_dapr_registration, _run, _RunnerTestCase


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _node(name: str) -> Any:
    return lambda state: {"log": [name]}


class TestDeferredNode(_RunnerTestCase):
    def _both(self, graph: StateGraph, name: str):
        _clear_dapr_registration()
        compiled = graph.compile()
        native = compiled.invoke({"log": []})
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name=name), {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        return native["log"], out.output["log"]

    def test_a_deferred_node_reached_from_branches_of_different_length_runs_once(self):
        graph = StateGraph(_Log)
        for name in ("a", "b", "b2"):
            graph.add_node(name, _node(name))
        graph.add_node("fin", _node("fin"), defer=True)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("b", "b2")
        graph.add_edge("a", "fin")
        graph.add_edge("b2", "fin")
        graph.add_edge("fin", END)
        native, ours = self._both(graph, "defer-uneven")
        self.assertEqual(native, ["a", "b", "b2", "fin"])
        self.assertEqual(ours, native)

    def test_what_a_deferred_node_starts_goes_on(self):
        graph = StateGraph(_Log)
        for name in ("a", "b", "b2", "after"):
            graph.add_node(name, _node(name))
        graph.add_node("fin", _node("fin"), defer=True)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge("b", "b2")
        graph.add_edge("a", "fin")
        graph.add_edge("b2", "fin")
        graph.add_edge("fin", "after")
        graph.add_edge("after", END)
        native, ours = self._both(graph, "defer-after")
        self.assertEqual(native, ["a", "b", "b2", "fin", "after"])
        self.assertEqual(ours, native)

    def test_a_deferred_join_waits_for_the_rest_too(self):
        graph = StateGraph(_Log)
        for name in ("a", "b", "c", "c2", "c3"):
            graph.add_node(name, _node(name))
        graph.add_node("fin", _node("fin"), defer=True)
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(START, "c")
        graph.add_edge("c", "c2")
        graph.add_edge("c2", "c3")
        graph.add_edge(["a", "b"], "fin")
        graph.add_edge("c3", END)
        graph.add_edge("fin", END)
        native, ours = self._both(graph, "defer-join")
        self.assertEqual(native, ["a", "b", "c", "c2", "c3", "fin"])
        self.assertEqual(ours, native)

    def test_a_graph_without_a_deferred_node_has_none_in_its_config(self):
        _clear_dapr_registration()
        graph = StateGraph(_Log)
        graph.add_node("a", _node("a"))
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="defer-none")
        self.assertEqual(runner.graph_config.deferred, [])
        self.assertEqual(
            GraphConfig.from_dict(runner.graph_config.to_dict()).deferred, []
        )
