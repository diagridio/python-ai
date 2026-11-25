# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""An instance started before the step-order change replays as it was scheduled.

Dapr replays the orchestrator against the recorded history and fails a run whose
code schedules a different activity at a given point. The workflow used to run a
step's nodes in the order its edges were found and to end the run as soon as END
was pending; it now runs them sorted by name (LangGraph's task order) and lets
the other branches of a step finish. An instance that began under the old code
must keep the old order, so the change sits behind ``ctx.is_patched``.
"""

from typing import Any, List

from langgraph.graph import END, START, StateGraph

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import (
    NODE_ACTIVITY_NAMES_PATCH,
    STEP_ORDER_PATCH,
)

from .golden_harness import _Log
from .test_runner import _InlineWorkflowContext, _run, _RunnerTestCase


def _graph() -> Any:
    graph = StateGraph(_Log)
    for name in ("x", "a", "b"):
        graph.add_node(name, lambda state, name=name: {"log": [name]})
    graph.add_edge(START, "x")
    # a sorts before b; the test lists b's edge first (see _scheduled)
    graph.add_edge("x", "b")
    graph.add_edge("x", "a")
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    return graph.compile()


class _Ctx(_InlineWorkflowContext):
    """Answers each patch by name, as an instance of a given age would."""

    def __init__(self, runner: Any, patches: List[str]) -> None:
        super().__init__(runner)
        self._patches = patches

    def is_patched(self, patch_name: str) -> bool:
        return patch_name in self._patches


class TestStepOrderPatch(_RunnerTestCase):
    def _scheduled(self, graph: Any, name: str, patches: List[str]) -> List[str]:
        runner = DaprWorkflowGraphRunner(graph=graph, name=name)
        # The edges of an in-flight instance are in its recorded input, in the
        # order its runner found them (a set's order); fix that order here.
        runner.graph_config.edges.sort(key=lambda e: (e.source, e.target), reverse=True)
        ctx = _Ctx(runner, patches)
        out = _run(runner, {"log": []}, ctx)
        self.assertEqual(out.status, "completed", out.error)
        return ctx.scheduled

    def test_a_new_instance_runs_a_steps_nodes_sorted_by_name(self):
        scheduled = self._scheduled(
            _graph(), "order-new", [NODE_ACTIVITY_NAMES_PATCH, STEP_ORDER_PATCH]
        )
        self.assertEqual(scheduled, ["x", "a", "b"])

    def test_an_instance_from_before_the_change_keeps_the_order_it_found(self):
        scheduled = self._scheduled(_graph(), "order-old", [NODE_ACTIVITY_NAMES_PATCH])
        self.assertEqual(scheduled, ["x", "b", "a"])

    def _end_beside_a(self) -> Any:
        graph = StateGraph(_Log)
        for name in ("x", "a"):
            graph.add_node(name, lambda state, name=name: {"log": [name]})
        graph.add_edge(START, "x")
        graph.add_conditional_edges("x", lambda s: [END, "a"], ["a", END])
        graph.add_edge("a", END)
        return graph.compile()

    def test_an_instance_from_before_the_change_still_ends_at_end(self):
        # END and another node pending in one step: the old code ended the run
        scheduled = self._scheduled(
            self._end_beside_a(), "end-old", [NODE_ACTIVITY_NAMES_PATCH]
        )
        self.assertEqual(scheduled, ["x", "evaluate_condition_activity"])

    def test_a_new_instance_runs_the_other_branch_of_a_step_that_reaches_end(self):
        scheduled = self._scheduled(
            self._end_beside_a(),
            "end-new",
            [NODE_ACTIVITY_NAMES_PATCH, STEP_ORDER_PATCH],
        )
        self.assertEqual(scheduled, ["x", "evaluate_condition_activity", "a"])
