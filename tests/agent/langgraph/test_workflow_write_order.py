# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Parallel nodes apply their writes in LangGraph's task order, not edge order."""

import itertools
import operator
from typing import Any, List

from langchain_core.messages import AIMessageChunk
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _clear_dapr_registration, _run, _RunnerTestCase
from .golden_harness import _both, _Golden


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _native_and_ours(graph: StateGraph, graph_input: dict, name: str):
    _clear_dapr_registration()
    compiled = graph.compile()
    native = compiled.invoke(graph_input)
    out = _run(DaprWorkflowGraphRunner(graph=compiled, name=name), graph_input)
    return native, out


class TestWriteOrder(_RunnerTestCase):
    def _fan_out(self) -> StateGraph:
        graph = StateGraph(_Log)
        for name in ("start", "zeta", "alpha", "mid"):
            graph.add_node(name, lambda s, name=name: {"log": [name]})
        graph.add_edge(START, "start")
        for name in ("zeta", "alpha", "mid"):
            graph.add_edge("start", name)
        return graph

    def test_parallel_nodes_apply_in_name_order(self):
        native, out = _native_and_ours(self._fan_out(), {"log": []}, "order")
        self.assertEqual(native["log"], ["start", "alpha", "mid", "zeta"])
        self.assertEqual(out.output["log"], native["log"])

    def test_the_order_of_the_edges_does_not_change_it(self):
        # the edges come from a set: their order differs between processes
        for permutation in itertools.permutations(range(3)):
            with self.subTest(permutation=permutation):
                _clear_dapr_registration()
                compiled = self._fan_out().compile()
                runner = DaprWorkflowGraphRunner(graph=compiled, name="order-perm")
                direct = [e for e in runner.graph_config.edges if e.source == "start"]
                others = [e for e in runner.graph_config.edges if e.source != "start"]
                runner.graph_config.edges = [
                    *others,
                    *(direct[i] for i in permutation),
                ]
                out = _run(runner, {"log": []})
                self.assertEqual(out.output["log"], ["start", "alpha", "mid", "zeta"])

    def test_send_tasks_apply_after_the_pull_tasks_in_index_order(self):
        def fan(state: _Log) -> Command:
            return Command(goto=[Send("w", {"i": i}) for i in (3, 1, 2)])

        graph = StateGraph(_Log)
        graph.add_node("fan", fan)
        graph.add_node("w", lambda arg: {"log": [f"w{arg['i']}"]})
        graph.add_node("z", lambda s: {"log": ["z"]})
        graph.add_edge(START, "fan")
        graph.add_edge("fan", "z")
        graph.add_edge("w", END)
        native, out = _native_and_ours(graph, {"log": []}, "order-send")
        self.assertEqual(native["log"], ["z", "w3", "w1", "w2"])
        self.assertEqual(out.output["log"], native["log"])
