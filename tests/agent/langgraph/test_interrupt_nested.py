# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``interrupt()`` below a node: in a subgraph the node runs, and in its router.

LangGraph pauses the node's task for both, and a resume runs the task again.
The workflow must pause and resume the same way, never complete as if the
interrupt had not happened.
"""

import operator
from typing import Any

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _S(TypedDict, total=False):
    log: Annotated[list, operator.add]
    choice: str


def _inner() -> Any:
    def inner(s: _S) -> dict:
        return {"log": [f"inner:{interrupt('inner?')}"]}

    sub = StateGraph(_S)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")
    sub.add_edge("inner", END)
    return sub.compile()


def _subgraph_node_graph() -> Any:
    g = StateGraph(_S)
    g.add_node("sub", _inner())
    g.add_edge(START, "sub")
    g.add_edge("sub", END)
    return g.compile()


def _calls_subgraph_graph(keep_result: bool) -> Any:
    sub = _inner()

    def call(s: _S) -> dict:
        out = sub.invoke({"log": []})
        if keep_result:
            return out
        return {"log": ["ignored the subgraph"]}

    g = StateGraph(_S)
    g.add_node("call", call)
    g.add_edge(START, "call")
    g.add_edge("call", END)
    return g.compile()


def _router_graph() -> Any:
    def pick(s: _S) -> dict:
        return {"log": ["pick"]}

    def route(s: _S) -> str:
        return "yes" if interrupt("which way?") == "yes" else "no"

    def yes(s: _S) -> dict:
        return {"log": ["yes"]}

    def no(s: _S) -> dict:
        return {"log": ["no"]}

    g = StateGraph(_S)
    g.add_node("pick", pick)
    g.add_node("yes", yes)
    g.add_node("no", no)
    g.add_edge(START, "pick")
    g.add_conditional_edges("pick", route, ["yes", "no"])
    g.add_edge("yes", END)
    g.add_edge("no", END)
    return g.compile()


def _run(graph: Any, name: str) -> Any:
    runner = log_runner(graph, name)
    http = TestClient(serve_app(runner))
    got = frames(http.post("/agent/run", json={"log": ["s"], "stream": True}).text)
    return runner, http, got


def _resume(http: Any, wf: str, value: Any) -> Any:
    return frames(
        http.post(
            f"/agent/run/{wf}/resume", json={"resume": value, "stream": True}
        ).text
    )


class TestSubgraphInterrupt(_RunnerTestCase):
    def test_a_subgraph_node_pauses_the_run_and_resumes(self):
        _, http, got = _run(_subgraph_node_graph(), "nest-sub")
        self.assertEqual(got[-1][1], "workflow_interrupted")
        self.assertEqual(got[-1][2]["interrupts"][0]["value"], "inner?")
        self.assertEqual(got[-1][2]["interrupts"][0]["node"], "sub")
        rest = _resume(http, got[-1][2]["workflow_id"], "ok")
        self.assertEqual(rest[-1][1], "workflow_completed")
        # as native: the subgraph's whole state is the node's update
        self.assertEqual(rest[-1][2]["output"]["log"], ["s", "s", "inner:ok"])

    def test_a_node_that_returns_a_subgraph_result_pauses_too(self):
        _, http, got = _run(_calls_subgraph_graph(keep_result=True), "nest-call")
        self.assertEqual(got[-1][1], "workflow_interrupted")
        rest = _resume(http, got[-1][2]["workflow_id"], "ok")
        self.assertEqual(rest[-1][1], "workflow_completed")
        self.assertEqual(rest[-1][2]["output"]["log"], ["s", "inner:ok"])

    def test_a_node_that_drops_a_subgraph_interrupt_fails_loudly(self):
        _, _, got = _run(_calls_subgraph_graph(keep_result=False), "nest-drop")
        self.assertEqual(got[-1][1], "workflow_failed")
        message = got[-1][2]["error"]["message"]
        self.assertIn("paused on interrupt()", message)
        self.assertIn("'call'", message)


class TestRouterInterrupt(_RunnerTestCase):
    def test_a_router_that_interrupts_pauses_and_resumes(self):
        _, http, got = _run(_router_graph(), "nest-route")
        self.assertEqual(got[-1][1], "workflow_interrupted")
        self.assertEqual(got[-1][2]["interrupts"][0]["value"], "which way?")
        rest = _resume(http, got[-1][2]["workflow_id"], "yes")
        self.assertEqual(rest[-1][1], "workflow_completed")
        self.assertEqual(rest[-1][2]["output"]["log"], ["s", "pick", "yes"])
