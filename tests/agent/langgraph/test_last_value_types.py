# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A plain (last value) channel holds the type its node wrote, not its JSON.

A ``datetime`` reached the next node as a string, and a pydantic object as a
dict, so ``state["when"].year`` or ``state["profile"].name`` raised where
LangGraph hands the node what was written.
"""

import datetime
import operator
from typing import Any, List

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _RunnerTestCase, _run


class _Profile(BaseModel):
    name: str


class _S(TypedDict, total=False):
    history: Annotated[List[AnyMessage], operator.add]
    when: datetime.datetime
    profile: _Profile
    seen: List[str]


def _graph() -> Any:
    def write(state: _S) -> dict:
        return {
            "history": [AIMessage("x")],
            "when": datetime.datetime(2020, 1, 2),
            "profile": _Profile(name="n"),
        }

    def read(state: _S) -> dict:
        return {
            "seen": [
                type(state["history"][0]).__name__,
                type(state["when"]).__name__,
                type(state["profile"]).__name__,
            ]
        }

    graph = StateGraph(_S)
    graph.add_node("write", write)
    graph.add_node("read", read)
    graph.add_edge(START, "write")
    graph.add_edge("write", "read")
    graph.add_edge("read", END)
    return graph.compile()


class TestLastValueTypes(_RunnerTestCase):
    def test_the_next_node_reads_the_types_langgraph_gives_it(self):
        compiled = _graph()
        native = compiled.invoke({"history": []})
        self.assertEqual(native["seen"], ["AIMessage", "datetime", "_Profile"])
        out = _run(
            DaprWorkflowGraphRunner(graph=compiled, name="lv-types"), {"history": []}
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["seen"], native["seen"])

    def test_the_input_keeps_its_types_too(self):
        def read(state: _S) -> dict:
            return {"seen": [type(state["when"]).__name__]}

        graph = StateGraph(_S)
        graph.add_node("read", read)
        graph.add_edge(START, "read")
        graph.add_edge("read", END)
        out = _run(
            DaprWorkflowGraphRunner(graph=graph.compile(), name="lv-input"),
            {"when": datetime.datetime(2020, 1, 2)},
        )
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["seen"], ["datetime"])

    def test_the_output_is_still_json(self):
        import json

        out = _run(
            DaprWorkflowGraphRunner(graph=_graph(), name="lv-output"), {"history": []}
        )
        json.dumps(out.output)
