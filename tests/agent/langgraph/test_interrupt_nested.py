# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``interrupt()`` below a node: in a subgraph the node runs, and in its router.

LangGraph pauses the node's task for both, and a resume runs the task again.
The workflow must pause and resume the same way, never complete as if the
interrupt had not happened.
"""

import json
import operator
import time
import unittest
from typing import Any, Dict, List

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph.interrupt_ids import partial_map_error

from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _clear_dapr_registration, _RunnerTestCase


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


def _parallel_inner(slow_on: Any) -> Any:
    """A subgraph whose nodes ``a`` and ``b`` both pause, in parallel.

    ``slow_on(name, call)`` says whether a node's ``call``-th run waits a
    moment before it asks: the order the two ask in can change between runs.
    """
    calls = {"a": 0, "b": 0}

    def asking(name: str) -> Any:
        def node(s: _S) -> dict:
            calls[name] += 1
            if slow_on(name, calls[name]):
                time.sleep(0.05)
            return {"log": [f"{name}={interrupt('q' + name)}"]}

        return node

    sub = StateGraph(_S)
    for name in ("a", "b"):
        sub.add_node(name, asking(name))
        sub.add_edge(START, name)
        sub.add_edge(name, END)
    g = StateGraph(_S)
    g.add_node("sub", sub.compile())
    g.add_edge(START, "sub")
    g.add_edge("sub", END)
    return g.compile()


def _fanout_inner() -> Any:
    """A subgraph that sends two tasks to one node, which both pause."""

    def fan(s: _S) -> dict:
        return {}

    def work(s: Any) -> dict:
        return {"log": [f"w{s['i']}={interrupt({'q': s['i']})}"]}

    sub = StateGraph(_S)
    sub.add_node("fan", fan)
    sub.add_node("work", work)
    sub.add_edge(START, "fan")
    sub.add_conditional_edges(
        "fan", lambda s: [Send("work", {"i": i}) for i in range(2)], ["work"]
    )
    sub.add_edge("work", END)
    g = StateGraph(_S)
    g.add_node("sub", sub.compile())
    g.add_edge(START, "sub")
    g.add_edge("sub", END)
    return g.compile()


def _ids(event: Any) -> Dict[Any, str]:
    return {
        json.dumps(i["value"], sort_keys=True): i["id"] for i in event["interrupts"]
    }


class TestParallelInterruptsInOneTask(_RunnerTestCase):
    """Two interrupts of one task (a subgraph's parallel nodes), resumed by id."""

    def test_a_partial_map_answers_only_the_interrupt_it_names(self):
        for slow in ("a", "b"):
            with self.subTest(slow=slow):
                _clear_dapr_registration()
                graph = _parallel_inner(lambda name, call: name == slow)
                _, http, got = _run(graph, f"par-{slow}")
                self.assertEqual(got[-1][1], "workflow_interrupted")
                ids = _ids(got[-1][2])
                wf = got[-1][2]["workflow_id"]
                rest = _resume(http, wf, {ids['"qb"']: "B"})
                self.assertEqual(rest[-1][1], "workflow_interrupted", rest[-1])
                left = rest[-1][2]["interrupts"]
                # the other one still waits, under the id the client has
                self.assertEqual(
                    [(i["value"], i["id"]) for i in left], [("qa", ids['"qa"'])]
                )
                done = _resume(http, wf, {ids['"qa"']: "A"})
                self.assertEqual(done[-1][1], "workflow_completed", done[-1])
                self.assertEqual(
                    sorted(done[-1][2]["output"]["log"]), ["a=A", "b=B", "s", "s"]
                )

    def test_a_full_map_reaches_each_interrupt_whatever_order_they_ask_in(self):
        # ``a`` asks first on the run that pauses, last on the one resumed
        graph = _parallel_inner(lambda name, call: (name == "a") == (call > 1))
        _, http, got = _run(graph, "par-flip")
        ids = _ids(got[-1][2])
        wf = got[-1][2]["workflow_id"]
        done = _resume(http, wf, {ids['"qa"']: "A", ids['"qb"']: "B"})
        self.assertEqual(done[-1][1], "workflow_completed", done[-1])
        self.assertEqual(sorted(done[-1][2]["output"]["log"]), ["a=A", "b=B", "s", "s"])

    def test_a_subgraph_node_that_asks_twice_gets_each_answer_once(self):
        def twice(s: _S) -> dict:
            first = interrupt("first?")
            return {"log": [f"{first}+{interrupt('second?')}"]}

        sub = StateGraph(_S)
        sub.add_node("twice", twice)
        sub.add_edge(START, "twice")
        sub.add_edge("twice", END)
        g = StateGraph(_S)
        g.add_node("sub", sub.compile())
        g.add_edge(START, "sub")
        g.add_edge("sub", END)
        _, http, got = _run(g.compile(), "par-twice")
        (asked,) = got[-1][2]["interrupts"]
        wf = got[-1][2]["workflow_id"]
        again = _resume(http, wf, {asked["id"]: "A"})
        self.assertEqual(again[-1][1], "workflow_interrupted", again[-1])
        (second,) = again[-1][2]["interrupts"]
        self.assertEqual(second["value"], "second?")
        done = _resume(http, wf, {second["id"]: "B"})
        self.assertEqual(done[-1][1], "workflow_completed", done[-1])
        self.assertEqual(done[-1][2]["output"]["log"], ["s", "s", "A+B"])

    def test_one_value_still_answers_them_all(self):
        _, http, got = _run(_parallel_inner(lambda name, call: False), "par-one")
        done = _resume(http, got[-1][2]["workflow_id"], "X")
        self.assertEqual(sorted(done[-1][2]["output"]["log"]), ["a=X", "b=X", "s", "s"])


class TestInterruptsWithoutStableIds(_RunnerTestCase):
    """Tasks a subgraph sends to one node cannot be told apart across runs."""

    def test_a_partial_map_is_refused(self):
        _, http, got = _run(_fanout_inner(), "fan-part")
        self.assertEqual(got[-1][1], "workflow_interrupted", got[-1])
        ids = _ids(got[-1][2])
        wf = got[-1][2]["workflow_id"]
        response = http.post(
            f"/agent/run/{wf}/resume", json={"resume": {ids['{"q": 1}']: "one"}}
        )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("every interrupt", response.text)
        done = _resume(http, wf, "both")
        self.assertEqual(done[-1][1], "workflow_completed", done[-1])
        self.assertEqual(
            sorted(done[-1][2]["output"]["log"]), ["s", "s", "w0=both", "w1=both"]
        )


class TestPartialMapRule(unittest.TestCase):
    def _pending(self, *items: Any) -> List[Dict[str, Any]]:
        return [dict(item) for item in items]

    def test_only_a_partial_map_of_a_task_without_stable_ids_is_refused(self):
        twins = self._pending(
            {"id": "x", "task": 0, "node": "sub"}, {"id": "y", "task": 0, "node": "sub"}
        )
        self.assertIsNotNone(partial_map_error(twins, {"x": 1}))
        self.assertIsNone(partial_map_error(twins, {"x": 1, "y": 2}))
        self.assertIsNone(partial_map_error(twins, "same for all"))
        stable = self._pending(
            {"id": "x", "task": 0, "path": "a"}, {"id": "y", "task": 0, "path": "b"}
        )
        self.assertIsNone(partial_map_error(stable, {"x": 1}))
        tasks = self._pending({"id": "x", "task": 0}, {"id": "y", "task": 1})
        self.assertIsNone(partial_map_error(tasks, {"x": 1}))
