# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A node and a router get the state as LangGraph shapes it for them."""

from dataclasses import dataclass
from typing import Any, List

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from .test_runner import _clear_dapr_registration, _run, _RunnerTestCase


class _Pyd(BaseModel):
    count: int = 0
    seen: str = ""


@dataclass
class _Plain:
    count: int = 0
    seen: str = ""


class _Open(TypedDict, total=False):
    log: List[str]
    secret: int
    seen: str


class _Priv(TypedDict):
    secret: int


class TestStateShapes(_RunnerTestCase):
    def _both(self, graph: StateGraph, graph_input: dict):
        _clear_dapr_registration()
        compiled = graph.compile()
        native = compiled.invoke(graph_input)
        out = _run(DaprWorkflowGraphRunner(graph=compiled, name="shapes"), graph_input)
        self.assertEqual(out.status, "completed", out.error)
        return native, out

    def _schema_graph(self, schema: Any) -> StateGraph:
        def a(state: Any) -> dict:
            return {"count": state.count + 1, "seen": type(state).__name__}

        def route(state: Any) -> str:
            return "b" if state.count > 0 else END

        graph = StateGraph(schema)
        graph.add_node("a", a)
        graph.add_node("b", lambda state: {"seen": state.seen + f":{state.count}"})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", route, ["b", END])
        graph.add_edge("b", END)
        return graph

    def test_a_pydantic_state_is_an_instance_for_a_node_and_a_router(self):
        native, out = self._both(self._schema_graph(_Pyd), {"count": 4})
        self.assertEqual(out.output["count"], native["count"])
        self.assertEqual(out.output["seen"], native["seen"])
        self.assertEqual(out.output["seen"], "_Pyd:5")

    def test_a_dataclass_state_is_an_instance_for_a_node_and_a_router(self):
        native, out = self._both(self._schema_graph(_Plain), {"count": 4})
        self.assertEqual(out.output["seen"], native["seen"])
        self.assertEqual(out.output["seen"], "_Plain:5")

    def test_a_node_with_an_input_schema_sees_only_its_channels(self):
        def b(state: Any) -> dict:
            return {"seen": ",".join(sorted(state))}

        graph = StateGraph(_Open)
        graph.add_node("a", lambda s: {"log": ["a"], "secret": 1})
        graph.add_node("b", b, input_schema=_Priv)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        native, out = self._both(graph, {})
        self.assertEqual(native["seen"], "secret")
        self.assertEqual(out.output["seen"], "secret")

    def test_a_router_sees_the_input_schema_of_its_source_node(self):
        def route(state: _Priv) -> str:
            return "b" if sorted(state) == ["secret"] else END

        graph = StateGraph(_Open)
        graph.add_node("a", lambda s: {"log": ["a"], "secret": 1}, input_schema=_Priv)
        graph.add_node("b", lambda s: {"seen": "routed"})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", route, ["b", END])
        graph.add_edge("b", END)
        native, out = self._both(graph, {})
        self.assertEqual(native["seen"], "routed")
        self.assertEqual(out.output.get("seen"), "routed")
