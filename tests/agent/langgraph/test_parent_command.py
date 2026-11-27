# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A subgraph node that hands over with ``Command(graph=Command.PARENT)``.

The swarm and supervisor patterns do it: a node inside a subgraph sends the
update and the ``goto`` to the graph the subgraph is a node of. The subgraph
runs inside the activity of its node, so the command must become that node's
result instead of failing the activity.
"""

import asyncio
from typing import Any

from langgraph.errors import ParentCommand
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.workflow import _parent_command_result

from .golden_harness import _both, _Golden, _Out
from .test_runner import _run


def _handoff(*, async_inner: bool = False, graph: Any = Command.PARENT) -> Any:
    if async_inner:

        async def inner(state: _Out) -> Any:
            return Command(goto="b", update={"out": ["handoff"]}, graph=graph)

    else:

        def inner(state: _Out) -> Any:  # type: ignore[misc]
            return Command(goto="b", update={"out": ["handoff"]}, graph=graph)

    sub = StateGraph(_Out)
    sub.add_node("inner", inner)
    sub.add_edge(START, "inner")

    parent = StateGraph(_Out)
    parent.add_node("a", sub.compile())
    parent.add_node("b", lambda state: {"out": ["b"]})
    parent.add_edge(START, "a")
    parent.add_edge("b", END)
    return parent.compile()


class TestParentCommand(_Golden):
    async def test_the_handoff_is_the_subgraph_nodes_update_and_goto(self):
        for async_inner in (False, True):
            with self.subTest(async_inner=async_inner):
                self.setUp()
                native, ours = await asyncio.wait_for(
                    _both(
                        _handoff(async_inner=async_inner),
                        f"parent-command-{async_inner}",
                        {"out": []},
                    ),
                    20,
                )
                self.assertEqual(
                    [c for m, c in native if m == "updates"],
                    [{"a": {"out": ["handoff"]}}, {"b": {"out": ["b"]}}],
                )
                self._assert_same_sequence(native, ours)

    def test_the_run_completes_with_the_handoffs_update(self):
        compiled = _handoff()
        runner = DaprWorkflowGraphRunner(graph=compiled, name="parent-command-run")
        out = _run(runner, {"out": []})
        self.assertEqual(out.status, "completed", out.error)
        self.assertEqual(out.output["out"], compiled.invoke({"out": []})["out"])
        self.assertEqual(out.output["out"], ["handoff", "b"])

    def test_a_command_addressed_above_this_graph_is_still_an_error(self):
        lost = ParentCommand(Command(goto="b", graph=Command.PARENT))
        with self.assertRaises(ParentCommand):
            _parent_command_result(lost, "a")
        for graph in ("", "a"):
            got = _parent_command_result(
                ParentCommand(Command(goto="b", update={"out": [1]}, graph=graph)), "a"
            )
            self.assertEqual(
                (got.goto, got.update, got.graph), ("b", {"out": [1]}, None)
            )
