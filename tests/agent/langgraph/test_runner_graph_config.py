# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The config bound to the graph, and the failures ``invoke`` raises."""

import asyncio
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .golden_harness import _chain, _FakeDaprTest, _runner
from .streaming_helpers import _collect


class _Loop(TypedDict, total=False):
    n: int
    k: Optional[str]


def _loop(steps: int) -> Any:
    """A graph that runs ``count`` ``steps`` times and reads ``configurable['k']``."""

    def count(state: _Loop, config: RunnableConfig) -> dict:
        return {
            "n": state.get("n", 0) + 1,
            "k": (config.get("configurable") or {}).get("k"),
        }

    graph = StateGraph(_Loop)
    graph.add_node("count", count)
    graph.add_edge(START, "count")
    graph.add_conditional_edges(
        "count", lambda s: "count" if s["n"] < steps else END, ["count", END]
    )
    return graph.compile()


class TestGraphBoundConfig(_FakeDaprTest):
    def _invoke(self, runner: Any, config: Optional[dict] = None) -> Any:
        return runner.invoke({}, config=config, poll_interval=0.01)

    async def test_a_bound_recursion_limit_and_configurable_apply(self):
        compiled = _loop(150).with_config(recursion_limit=500, configurable={"k": "v"})
        native = compiled.invoke({})
        runner = _runner(compiled, "bound-loop")
        out = await asyncio.to_thread(self._invoke, runner)
        self.assertEqual((native["n"], native["k"]), (150, "v"))
        self.assertEqual((out["n"], out["k"]), (150, "v"))

    async def test_a_bound_configurable_reaches_a_streamed_run(self):
        compiled = _loop(2).with_config(configurable={"k": "v"})
        runner = _runner(compiled, "bound-stream")
        seen = await _collect(
            runner.astream({}, stream_mode=["updates"], poll_interval=0.01)
        )
        self.assertEqual(
            [c["count"]["k"] for m, c in seen if m == "updates"], ["v"] * 2
        )

    async def test_a_bound_recursion_limit_below_the_default_is_enforced(self):
        compiled = _chain(4).with_config(recursion_limit=3)
        with self.assertRaises(GraphRecursionError):
            compiled.invoke({"log": []})
        runner = _runner(compiled, "bound-low")
        with self.assertRaises(GraphRecursionError):
            await _collect(runner.astream({"log": []}, poll_interval=0.01))

    async def test_the_caller_config_wins_over_the_bound_one(self):
        compiled = _chain(4).with_config(recursion_limit=3, configurable={"k": "v"})
        runner = _runner(compiled, "bound-override")
        out = await asyncio.to_thread(
            self._invoke, runner, {"recursion_limit": 50, "configurable": {"j": "w"}}
        )
        self.assertEqual(len(out["log"]), 4)

    async def test_the_callers_config_is_not_changed(self):
        runner = _runner(_chain(1), "bound-copy")
        config = {"tags": ["t"]}
        await asyncio.to_thread(self._invoke, runner, config)
        self.assertEqual(config, {"tags": ["t"]})
