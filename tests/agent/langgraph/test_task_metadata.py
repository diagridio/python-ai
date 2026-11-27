# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A node sees the ``langgraph_*`` metadata native LangGraph gives its task.

Whether the run is streamed or not: the keys come from the config every node
and router runs under, not only from the stream handler.
"""

import re
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .golden_harness import _FakeDaprTest, _runner
from .streaming_helpers import _collect

_NS = re.compile(r"^(\w+):.+$")


class _S(TypedDict, total=False):
    log: List[str]


def _graph(seen: Dict[str, Dict[str, Any]]) -> Any:
    def a(state: _S, config: RunnableConfig) -> dict:
        seen["a"] = dict(config["metadata"])
        return {"log": ["a"]}

    def b(state: _S) -> dict:
        seen["b"] = dict(get_config()["metadata"])
        return {"log": ["b"]}

    def route(state: _S) -> str:
        seen["route"] = dict(get_config()["metadata"])
        return "b"

    graph = StateGraph(_S)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", route, ["b"])
    graph.add_edge("b", END)
    return graph.compile()


def _comparable(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The metadata with task ids (uuids natively, integers here) left out."""
    out = dict(metadata)
    out.pop("checkpoint_ns", None)
    ns = out.get("langgraph_checkpoint_ns")
    if isinstance(ns, str):
        out["langgraph_checkpoint_ns"] = _NS.sub(r"\1:<task>", ns)
    return out


CONFIG = {"configurable": {"thread_id": "t1"}, "metadata": {"foo": "bar"}}


class TestTaskMetadata(_FakeDaprTest):
    def _native(self) -> Dict[str, Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}
        _graph(seen).invoke({"log": []}, CONFIG)
        return seen

    def test_an_invoked_run_matches_native(self):
        want = self._native()
        seen: Dict[str, Dict[str, Any]] = {}
        runner = _runner(_graph(seen), "meta-invoke")
        runner.invoke(
            {"log": []},
            thread_id="t1",
            config={"metadata": {"foo": "bar"}},
            poll_interval=0.01,
            timeout=10,
        )
        self.assertEqual(set(seen), {"a", "b", "route"})
        for name in seen:
            with self.subTest(task=name):
                self.assertEqual(_comparable(seen[name]), _comparable(want[name]))

    async def test_a_streamed_run_sets_the_same_keys(self):
        want = self._native()
        seen: Dict[str, Dict[str, Any]] = {}
        runner = _runner(_graph(seen), "meta-stream")
        await _collect(
            runner.astream(
                {"log": []},
                thread_id="t1",
                config={"metadata": {"foo": "bar"}},
                stream_mode="updates",
                poll_interval=0.01,
            )
        )
        keys = [k for k in want["a"] if k.startswith("langgraph_")]
        for name in ("a", "b", "route"):
            with self.subTest(task=name):
                got = _comparable(seen[name])
                self.assertEqual(
                    {k: got.get(k) for k in keys},
                    {k: _comparable(want[name])[k] for k in keys},
                )
