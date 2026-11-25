# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A conditional edge runs once per task, on the state with only that task's writes."""

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


_seen: List[Any] = []


class TestRouterPerTask(_RunnerTestCase):
    def setUp(self):
        super().setUp()
        _seen.clear()

    def _routers_saw(self, graph: StateGraph, name: str):
        """What a router saw natively and in the runner, and the two outputs."""
        _clear_dapr_registration()
        compiled = graph.compile()
        _seen.clear()
        native = compiled.invoke({"log": []})
        native_seen = sorted(_seen)
        _seen.clear()
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name=name), {"log": []})
        self.assertEqual(out.status, "completed", out.error)
        return native_seen, sorted(_seen), native, out.output

    def test_a_router_sees_only_its_own_tasks_writes(self):
        def router(state: _Log) -> str:
            _seen.append(list(state["log"]))
            return "x"

        graph = StateGraph(_Log)
        graph.add_node("a", lambda s: {"log": ["a"]})
        graph.add_node("b", lambda s: {"log": ["b", "b"]})
        graph.add_node("x", lambda s: {"log": ["x"]})
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_conditional_edges("a", router, ["x"])
        graph.add_edge("b", END)
        graph.add_edge("x", END)
        native_seen, seen, native, out = self._routers_saw(graph, "router-parallel")
        self.assertEqual(native_seen, [["a"]])
        self.assertEqual(seen, native_seen)
        self.assertEqual(out["log"], native["log"])

    def test_a_router_runs_once_per_send_task(self):
        def router(state: _Log) -> str:
            _seen.append(list(state["log"]))
            return END

        graph = StateGraph(_Log)
        graph.add_node(
            "fan", lambda s: Command(goto=[Send("w", {"i": i}) for i in (1, 2, 3)])
        )
        graph.add_node("w", lambda arg: {"log": [f"w{arg['i']}"]})
        graph.add_edge(START, "fan")
        graph.add_conditional_edges("w", router, [END])
        native_seen, seen, _, _ = self._routers_saw(graph, "router-send")
        self.assertEqual(native_seen, [["w1"], ["w2"], ["w3"]])
        self.assertEqual(seen, native_seen)
