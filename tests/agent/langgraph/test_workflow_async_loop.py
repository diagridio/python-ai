# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Async nodes run on one event loop that outlives the activity.

A chat model keeps an async HTTP client (a connection pool bound to the loop it
was first used on). A loop made and closed for each activity would fail the
next activity that reuses the client with ``Event loop is closed``.
"""

import asyncio
from typing import Any, List, TypedDict

from langchain_core.runnables.config import var_child_runnable_config
from langgraph.graph import END, START, StateGraph
from langgraph.utils.runnable import RunnableCallable

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _run, _RunnerTestCase


class _State(TypedDict, total=False):
    n: int


class TestAsyncNodeLoop(_RunnerTestCase):
    def _graph(self, loops: List[Any], node_count: int = 3) -> DaprWorkflowGraphRunner:
        graph = StateGraph(_State)
        names = []
        for i in range(node_count):

            async def node(state: _State) -> dict:
                loops.append(asyncio.get_running_loop())
                return {"n": i}

            graph.add_node(f"n{i}", node)
            names.append(f"n{i}")
        graph.add_edge(START, names[0])
        for a, b in zip(names, names[1:]):
            graph.add_edge(a, b)
        graph.add_edge(names[-1], END)
        return DaprWorkflowGraphRunner(graph=graph.compile(), name="loop")

    def test_every_activity_runs_on_the_same_open_loop(self):
        loops: List[Any] = []
        runner = self._graph(loops)
        out = _run(runner, {"n": 0})
        self.assertEqual(out.status, "completed")
        self.assertEqual(len(loops), 3)
        self.assertEqual(len({id(loop) for loop in loops}), 1)
        self.assertFalse(loops[0].is_closed())

    def test_a_dual_entry_node_shares_the_loop_too(self):
        loops: List[Any] = []

        def sync_fn(state: _State) -> dict:
            return {"n": 1}

        async def async_fn(state: _State) -> dict:
            loops.append(asyncio.get_running_loop())
            return {"n": 1}

        graph = StateGraph(_State)
        graph.add_node("a", RunnableCallable(sync_fn, async_fn))
        graph.add_node("b", RunnableCallable(sync_fn, async_fn))
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="loop-dual")
        _run(runner, {"n": 0})
        self.assertEqual(len(loops), 2)
        self.assertIs(loops[0], loops[1])

    def test_the_activitys_ambient_config_reaches_the_coroutine(self):
        seen: List[Any] = []

        async def node(state: _State) -> dict:
            seen.append(var_child_runnable_config.get())
            return {"n": 1}

        graph = StateGraph(_State)
        graph.add_node("a", node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="loop-ctx")
        _run(runner, {"n": 0})
        self.assertEqual(len(seen), 1)
        self.assertIsNotNone(seen[0])
        # the caller's context is untouched by what ran on the loop
        self.assertIsNone(var_child_runnable_config.get())
