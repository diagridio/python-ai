# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A run an orchestrator starts with ``{"task": ...}`` gets the graph's own config.

Every runner entry point merges the config bound with ``with_config`` into the
run's config; the workflow an orchestrator starts takes the same.
"""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import GraphWorkflowOutput

from .test_runner import _InlineWorkflowContext, _RunnerTestCase
from .test_workflow_replay import _drive


class _S(TypedDict, total=False):
    n: int
    tenant: Optional[str]
    error: Optional[str]


def _loop(steps: int) -> Any:
    def count(state: _S, config: RunnableConfig) -> dict:
        configurable = config.get("configurable") or {}
        return {
            "n": state.get("n", 0) + 1,
            "tenant": configurable.get("tenant"),
            "error": None if "tenant" in configurable else "no tenant",
        }

    graph = StateGraph(_S)
    graph.add_node("count", count)
    graph.add_edge(START, "count")
    graph.add_conditional_edges(
        "count", lambda s: "count" if s["n"] < steps else END, ["count", END]
    )
    return graph.compile()


def _task_run(compiled: Any, name: str) -> Dict[str, Any]:
    runner = DaprWorkflowGraphRunner(graph=compiled, name=name)
    runner._setup_serve_defaults()
    return _drive(_InlineWorkflowContext(runner), {"task": "hi"})


class TestOrchestratorRun(_RunnerTestCase):
    def test_the_bound_configurable_reaches_the_nodes(self):
        compiled = _loop(2).with_config(configurable={"tenant": "x"})
        out = GraphWorkflowOutput.from_dict(_task_run(compiled, "orch-configurable"))
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["tenant"], "x")
        self.assertIsNone(out.output["error"])

    def test_the_bound_recursion_limit_bounds_the_run(self):
        compiled = _loop(50).with_config(recursion_limit=5)
        out = GraphWorkflowOutput.from_dict(_task_run(compiled, "orch-limit"))
        self.assertEqual(out.status, "max_steps_reached")
        self.assertEqual(out.steps, 5)

    def test_a_graph_without_a_bound_config_runs_as_before(self):
        out = GraphWorkflowOutput.from_dict(_task_run(_loop(3).with_config(), "plain"))
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["n"], 3)
