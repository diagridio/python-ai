# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A node's conditional edges run inside the node's activity.

LangGraph runs a router as one of its node's writers, inside the node's task,
so the router's chunks come before the node's ``updates`` chunk. The node
activity now does the same, behind ``ROUTE_IN_NODE_PATCH``: an instance started
before it replays the separate condition activity it scheduled. A router that
raises inside the node activity falls back to that separate activity, so the
node itself never runs twice.
"""

import operator
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessageChunk
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated, TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import (
    CONDITION_ACTIVITY,
    NODE_ACTIVITY_NAMES_PATCH,
    ROUTE_IN_NODE_PATCH,
    STEP_ORDER_PATCH,
)

from .golden_harness import _both, _canonical, _FakeDaprTest, _runner, _wire
from .streaming_helpers import _collect, _StreamingModel
from .test_runner import _InlineWorkflowContext, _run, _RunnerTestCase

ALL_PATCHES = [NODE_ACTIVITY_NAMES_PATCH, STEP_ORDER_PATCH, ROUTE_IN_NODE_PATCH]
BEFORE_ROUTE_IN_NODE = [NODE_ACTIVITY_NAMES_PATCH, STEP_ORDER_PATCH]


class _Log(TypedDict, total=False):
    log: Annotated[list, operator.add]


class _Ctx(_InlineWorkflowContext):
    """Answers each patch by name, as an instance of a given age would."""

    def __init__(self, runner: Any, patches: List[str]) -> None:
        super().__init__(runner)
        self._patches = patches

    def is_patched(self, patch_name: str) -> bool:
        return patch_name in self._patches


def _routed_graph(calls: Dict[str, int], raise_once: bool = False) -> Any:
    """START -> a -(router)-> b -> END, counting how often a and the router run."""

    def a(state: _Log) -> dict:
        calls["a"] = calls.get("a", 0) + 1
        return {"log": ["a"]}

    def router(state: _Log) -> str:
        calls["router"] = calls.get("router", 0) + 1
        get_stream_writer()("routing")
        if raise_once and calls["router"] == 1:
            raise RuntimeError("router hiccup")
        return "b"

    graph = StateGraph(_Log)
    graph.add_node("a", a)
    graph.add_node("b", lambda state: {"log": ["b"]})
    graph.add_edge(START, "a")
    graph.add_conditional_edges("a", router, ["b"])
    graph.add_edge("b", END)
    return graph.compile()


class TestTheSchedule(_RunnerTestCase):
    def _scheduled(self, compiled: Any, name: str, patches: List[str]) -> List[str]:
        runner = DaprWorkflowGraphRunner(graph=compiled, name=name)
        ctx = _Ctx(runner, patches)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["log"], ["a", "b"])
        return ctx.scheduled

    def test_a_new_instance_schedules_no_condition_activity(self):
        scheduled = self._scheduled(_routed_graph({}), "route-new", ALL_PATCHES)
        self.assertEqual(scheduled, ["a", "b"])

    def test_an_instance_from_before_the_patch_replays_the_condition_activity(self):
        scheduled = self._scheduled(
            _routed_graph({}), "route-old", BEFORE_ROUTE_IN_NODE
        )
        self.assertEqual(scheduled, ["a", CONDITION_ACTIVITY, "b"])


class TestARouterThatRaises(_RunnerTestCase):
    def test_it_falls_back_to_the_condition_activity_and_the_node_runs_once(self):
        calls: Dict[str, int] = {}
        compiled = _routed_graph(calls, raise_once=True)
        runner = DaprWorkflowGraphRunner(graph=compiled, name="route-raise")
        ctx = _Ctx(runner, ALL_PATCHES)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["log"], ["a", "b"])
        self.assertEqual(ctx.scheduled, ["a", CONDITION_ACTIVITY, "b"])
        self.assertEqual(calls, {"a": 1, "router": 2})


class TestARouterThatRaisesStreams(_FakeDaprTest):
    async def test_the_consumer_gets_a_branch_reset_for_the_router_only(self):
        calls: Dict[str, int] = {}
        runner = _runner(_routed_graph(calls, raise_once=True), "route-raise-stream")
        ours = await _collect(
            runner.astream(
                {"log": []},
                workflow_id="wf-route-raise-stream",
                stream_mode=["custom", "updates"],
                poll_interval=0.01,
            )
        )
        resets = [data for kind, data in ours if kind == "node_reset"]
        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["node"], "a")
        self.assertEqual(resets[0]["step"], 1)
        self.assertIn("router", resets[0]["branch"])
        self.assertEqual(
            [i for i in ours if i[0] != "node_reset"],
            [
                ("custom", "routing"),
                ("updates", {"a": {"log": ["a"]}}),
                ("custom", "routing"),
                ("updates", {"b": {"log": ["b"]}}),
            ],
        )
        self.assertEqual(calls, {"a": 1, "router": 2})


def _parallel_routers_graph() -> Any:
    def ra(state: _Log) -> str:
        get_stream_writer()("ra")
        return "x"

    def rb(state: _Log) -> str:
        get_stream_writer()("rb")
        return "y"

    graph = StateGraph(_Log)
    for name in ("a", "b", "x", "y"):
        graph.add_node(name, lambda state, name=name: {"log": [name]})
    graph.add_edge(START, "a")
    graph.add_edge(START, "b")
    graph.add_conditional_edges("a", ra, ["x"])
    graph.add_conditional_edges("b", rb, ["y"])
    graph.add_edge("x", END)
    graph.add_edge("y", END)
    return graph.compile()


def _before_its_update(items: List[Tuple[str, Any]], custom: str, node: str) -> bool:
    """Whether the router's ``custom`` event comes before its node's ``updates``."""
    modes = [(m, p if m == "custom" else next(iter(p))) for m, p in items]
    return modes.index(("custom", custom)) < modes.index(("updates", node))


class TestParallelStepWithTwoRouters(_FakeDaprTest):
    async def test_each_router_streams_before_its_nodes_update_like_native(self):
        native, ours = await _both(
            _parallel_routers_graph(), "two-routers", {"log": []}
        )
        self.assertEqual(_canonical(ours), _canonical(native))
        self.assertNotIn("node_reset", [m for m, _ in ours])
        for custom, node in (("ra", "a"), ("rb", "b")):
            with self.subTest(router=custom):
                self.assertTrue(_before_its_update(native, custom, node))
                self.assertTrue(_before_its_update(ours, custom, node))

    def test_neither_router_needs_an_activity_of_its_own(self):
        compiled = _parallel_routers_graph()
        runner = DaprWorkflowGraphRunner(graph=compiled, name="two-routers-run")
        ctx = _Ctx(runner, ALL_PATCHES)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(sorted(out.output["log"]), sorted(compiled.invoke({})["log"]))
        self.assertNotIn(CONDITION_ACTIVITY, ctx.scheduled)


def _send_router_graph() -> Any:
    model = _StreamingModel()

    def router(state: _Log) -> str:
        get_stream_writer()(f"route-{state['log'][-1]}")
        model.invoke("which way?")
        return END

    graph = StateGraph(_Log)
    graph.add_node(
        "fan", lambda state: Command(goto=[Send("w", {"i": i}) for i in (1, 2)])
    )
    graph.add_node("w", lambda arg: {"log": [f"w{arg['i']}"]})
    graph.add_edge(START, "fan")
    graph.add_conditional_edges("w", router, [END])
    return graph.compile()


class TestASendTasksRouter(_FakeDaprTest):
    async def test_it_streams_as_its_task_before_the_tasks_update(self):
        native, ours = await _both(_send_router_graph(), "send-router", {"log": []})

        def paths(items: List) -> List[Any]:
            return sorted(
                (tuple(md["langgraph_path"]), md["langgraph_node"])
                for m, (c, md) in [i for i in items if i[0] == "messages"]
                if isinstance(c, AIMessageChunk)
            )

        self.assertEqual(paths(ours), paths(native))
        for index in (1, 2):
            with self.subTest(task=index):
                modes = [
                    (m, p if m == "custom" else p.get("w")) for m, p in _wire(ours)
                ]
                self.assertLess(
                    modes.index(("custom", f"route-w{index}")),
                    modes.index(("updates", {"log": [f"w{index}"]})),
                )
        self.assertNotIn("node_reset", [m for m, _ in ours])

    def test_no_condition_activity_is_scheduled_for_the_send_tasks(self):
        compiled = _send_router_graph()
        runner = DaprWorkflowGraphRunner(graph=compiled, name="send-router-run")
        ctx = _Ctx(runner, ALL_PATCHES)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(sorted(out.output["log"]), ["w1", "w2"])
        self.assertEqual(ctx.scheduled, ["fan", "w", "w"])


class TestAPathMapMiss(_RunnerTestCase):
    def test_it_fails_the_run_without_running_the_router_again(self):
        calls: List[str] = []

        def router(state: _Log) -> str:
            calls.append("router")
            return "nowhere"

        graph = StateGraph(_Log)
        graph.add_node("a", lambda state: {"log": ["a"]})
        graph.add_node("b", lambda state: {"log": ["b"]})
        graph.add_edge(START, "a")
        graph.add_conditional_edges("a", router, {"x": "b"})
        graph.add_edge("b", END)
        runner = DaprWorkflowGraphRunner(graph=graph.compile(), name="route-miss")
        ctx = _Ctx(runner, ALL_PATCHES)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "error")
        self.assertIn("Condition evaluation failed", out.error or "")
        self.assertIn("nowhere", out.error or "")
        self.assertEqual(ctx.scheduled, ["a"])
        self.assertEqual(calls, ["router"])
