# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A graph whose whole state is one channel: ``StateGraph(Annotated[list, add])``.

LangGraph names that channel ``__root__``. The graph's input is the channel's
value, a node is handed that value and what it returns is the channel's write
(not a dict of channels), and the run's result is the value.
"""

import asyncio
import operator
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .golden_harness import _both, _FakeDaprTest, _Golden, _runner


def _graph() -> Any:
    seen = []

    def a(state: Any) -> Any:
        seen.append(("a", type(state).__name__, list(state)))
        return ["a"]

    def b(state: Any) -> Any:
        return ["b"]

    graph = StateGraph(Annotated[list, operator.add])
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    compiled = graph.compile()
    compiled.seen = seen  # type: ignore[attr-defined]
    return compiled


class TestRootState(_Golden):
    async def test_the_stream_matches_native(self):
        compiled = _graph()
        native, ours = await asyncio.wait_for(_both(compiled, "root", ["in"]), 20)
        self.assertEqual(
            [c for m, c in native if m == "updates"], [{"a": ["a"]}, {"b": ["b"]}]
        )
        self._assert_same_sequence(native, ours)
        # the node was handed the channel's value, as a list
        self.assertEqual({entry[:2] for entry in compiled.seen}, {("a", "list")})
        self.assertEqual(compiled.seen[-1][2], ["in"])


class TestRootStateInvoke(_FakeDaprTest):
    async def test_invoke_takes_the_value_and_returns_it(self):
        compiled = _graph()
        native = compiled.invoke(["in"])
        runner: DaprWorkflowGraphRunner = _runner(compiled, "root-invoke")
        out = await asyncio.to_thread(lambda: runner.invoke(["in"], poll_interval=0.01))
        self.assertEqual(native, ["in", "a", "b"])
        self.assertEqual(out, native)
