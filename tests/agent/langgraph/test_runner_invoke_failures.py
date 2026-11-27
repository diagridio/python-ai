# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``invoke`` raises for a run that did not finish, as native ``invoke`` does."""

import asyncio
import operator
from typing import Any, Optional

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph.streaming import GraphRunError

from .golden_harness import _chain, _FakeDaprTest, _runner


class TestInvokeFailures(_FakeDaprTest):
    def _invoke(self, runner: Any, config: Optional[dict] = None) -> Any:
        return runner.invoke({"log": []}, config=config, poll_interval=0.01)

    async def test_a_run_cut_by_the_recursion_limit_raises_like_native(self):
        compiled = _chain(3)
        runner = _runner(compiled, "invoke-limit")
        for limit, ok in ((2, False), (3, False), (4, True), (5, True)):
            with self.subTest(limit=limit):
                config = {"recursion_limit": limit}
                if ok:
                    out = await asyncio.to_thread(self._invoke, runner, config)
                    self.assertEqual(out["log"], ["n0", "n1", "n2"])
                else:
                    with self.assertRaises(GraphRecursionError):
                        compiled.invoke({"log": []}, config)
                    with self.assertRaises(GraphRecursionError) as ctx:
                        await asyncio.to_thread(self._invoke, runner, config)
                    self.assertIsInstance(ctx.exception, GraphRunError)

    async def test_a_limit_bound_to_the_graph_raises_too(self):
        runner = _runner(_chain(4).with_config(recursion_limit=3), "invoke-bound")
        with self.assertRaises(GraphRecursionError):
            await asyncio.to_thread(self._invoke, runner)

    async def test_a_failed_run_raises_instead_of_returning_a_partial_output(self):
        class State(TypedDict, total=False):
            log: Annotated[list, operator.add]

        graph = StateGraph(State)
        graph.add_node("a", lambda s: {"log": ["a"]})
        graph.add_node("b", lambda s: {"log": ["b"]})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", lambda s: "nowhere", {"go": "b"})
        graph.add_edge("b", END)
        runner = _runner(graph.compile(), "invoke-fail")
        with self.assertRaises(GraphRunError) as ctx:
            await asyncio.to_thread(self._invoke, runner)
        self.assertNotIsInstance(ctx.exception, GraphRecursionError)
        self.assertIn("nowhere", str(ctx.exception))

    async def test_a_config_that_cannot_travel_is_refused(self):
        runner = _runner(_chain(1), "invoke-bad-config")
        with self.assertRaisesRegex(ValueError, "callbacks"):
            await asyncio.to_thread(self._invoke, runner, {"callbacks": [object()]})
        with self.assertRaisesRegex(ValueError, "recursion_limit"):
            await asyncio.to_thread(self._invoke, runner, {"recursion_limit": 0})
