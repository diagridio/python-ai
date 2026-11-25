# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A node that is a composite runnable with an async-only step inside it.

A compiled subgraph (or a ``RunnableSequence``) has no ``afunc`` of its own, but
a step inside it may be an ``async def`` that has no sync entry. Native
``ainvoke`` and ``astream`` run it; the node must go through ``ainvoke`` too.
"""

from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _RunnerTestCase, _run


class _S(TypedDict, total=False):
    x: int


async def _inner(state: _S) -> dict:
    return {"x": state["x"] + 1}


def _parent(sub: Any) -> Any:
    graph = StateGraph(_S)
    graph.add_node("sub", sub)
    graph.add_edge(START, "sub")
    graph.add_edge("sub", END)
    return graph.compile()


def _async_subgraph() -> Any:
    inner = StateGraph(_S)
    inner.add_node("inner", _inner)
    inner.add_edge(START, "inner")
    inner.add_edge("inner", END)
    return inner.compile()


def _sync_subgraph() -> Any:
    inner = StateGraph(_S)
    inner.add_node("inner", lambda s: {"x": s["x"] + 1})
    inner.add_edge(START, "inner")
    inner.add_edge("inner", END)
    return inner.compile()


class TestAsyncCompositeNodes(_RunnerTestCase):
    def _check(self, sub: Any, name: str) -> None:
        compiled = _parent(sub)
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name=name), {"x": 1})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output, {"x": 2})

    def test_a_subgraph_with_an_async_only_node(self):
        self._check(_async_subgraph(), "async-subgraph")

    def test_a_sequence_with_an_async_only_step(self):
        sequence = RunnableLambda(lambda s: s) | RunnableLambda(_inner)
        self._check(sequence, "async-sequence")

    def test_a_subgraph_of_sync_nodes_still_runs(self):
        self._check(_sync_subgraph(), "sync-subgraph")
