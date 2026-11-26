# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A router that ran in its own activity is listed in ``missing_nodes`` when
none of its chunks arrived.

A conditional edge's function runs in the condition activity when it is a
START edge, when it raised inside its node's activity, or for an instance
older than routing inside the node. The workflow reports each such call on
its output (``branches``); the router's streamer marks it done in the sink
(an internal ``branch_done`` that never reaches a client), so a router that
ran in another process is reported like a node that did.
"""

import unittest
from typing import Any, List

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from diagrid.agent.core.workflow.stream import BRANCH_DONE, StreamEvent
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.streaming import stream_unavailable_event

from .golden_harness import _FakeDaprTest, _runner
from .streaming_helpers import INPUT, _collect
from .test_runner import _RunnerTestCase, _run
from .test_streaming_events import _chunk, _FakeRun

ROUTER = "a:branch:route"


def _final(**extra: Any) -> dict:
    return {"type": "workflow_completed", "steps": 1, "executed": [["a"]], **extra}


class TestStreamUnavailableWithRouters(unittest.TestCase):
    def test_a_router_whose_chunks_never_arrived_is_missing(self):
        event = stream_unavailable_event(
            _final(branches=[[0, ROUTER]]), {(0, "a")}, "wf"
        )
        assert event is not None
        self.assertEqual(event.data["missing_nodes"], [{"step": 1, "node": ROUTER}])
        self.assertEqual(event.data["missing_steps"], [1])

    def test_a_router_marked_done_is_not_missing(self):
        seen = {(0, "a"), (0, ROUTER)}
        self.assertIsNone(
            stream_unavailable_event(_final(branches=[[0, ROUTER]]), seen, "wf")
        )

    def test_a_start_router_is_numbered_as_the_input_step(self):
        event = stream_unavailable_event(
            _final(branches=[[-1, "__start__:branch:pick"]]), {(0, "a")}, "wf"
        )
        assert event is not None
        self.assertEqual(
            event.data["missing_nodes"], [{"step": 0, "node": "__start__:branch:pick"}]
        )


class TestBranchDoneInTheStream(_FakeRun):
    async def test_branch_done_marks_the_router_seen_and_is_not_yielded(self):
        done = StreamEvent("wf", 0, ROUTER, "r1", 0, (), BRANCH_DONE, None)
        runner = self._runner(
            "router-done",
            [_chunk(0, "a", "updates"), done],
            executed=[["a"]],
            branches=[[0, ROUTER]],
        )
        items = await _collect(
            runner.astream(INPUT, stream_mode=["updates", "custom", "messages"])
        )
        self.assertEqual([m for m, _ in items], ["updates"])

    async def test_a_router_without_branch_done_is_reported(self):
        runner = self._runner(
            "router-gone",
            [_chunk(0, "a", "updates")],
            executed=[["a"]],
            branches=[[0, ROUTER]],
        )
        items = await _collect(runner.astream(INPUT, stream_mode=["updates"]))
        self.assertEqual(items[-1][0], "stream_unavailable")
        self.assertEqual(items[-1][1]["missing_nodes"], [{"step": 1, "node": ROUTER}])


class _S(TypedDict, total=False):
    log: List[str]


def _start_router_graph() -> Any:
    def pick(state: _S) -> str:
        return "a"

    graph = StateGraph(_S)
    graph.add_node("a", lambda s: {"log": ["a"]})
    graph.add_conditional_edges(START, pick, ["a"])
    graph.add_edge("a", END)
    return graph.compile()


class TestWorkflowReportsBranches(_RunnerTestCase):
    def test_a_start_router_is_on_the_output(self):
        runner = DaprWorkflowGraphRunner(graph=_start_router_graph(), name="br-out")
        out = _run(runner, {"log": []})
        self.assertEqual(
            out.branches, [[-1, "__start__:branch:__start___pick_condition"]]
        )


class TestRouterStreamedHere(_FakeDaprTest):
    async def test_a_router_that_ran_here_is_not_reported(self):
        runner = _runner(_start_router_graph(), "br-here")
        items = await _collect(
            runner.astream(
                {"log": []}, workflow_id="wf-br-here", stream_mode=["updates"]
            )
        )
        self.assertNotIn("stream_unavailable", [m for m, _ in items])
        self.assertIn("updates", [m for m, _ in items])
