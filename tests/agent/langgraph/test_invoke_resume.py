# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``invoke(Command(resume=...), workflow_id=...)`` resumes a paused run, as native."""

from langgraph.types import Command

from .log_helpers import log_runner
from .test_interrupt_nested import _subgraph_node_graph
from .test_runner import _RunnerTestCase


class TestInvokeCommand(_RunnerTestCase):
    def test_invoke_with_a_command_resumes_the_run(self):
        runner = log_runner(_subgraph_node_graph(), "nest-inv")
        first = runner.invoke(
            {"log": []}, workflow_id="wf-nest-inv", poll_interval=0.01, timeout=10
        )
        self.assertIn("__interrupt__", first)
        out = runner.invoke(
            Command(resume="go"),
            workflow_id="wf-nest-inv",
            poll_interval=0.01,
            timeout=10,
        )
        self.assertEqual(out["log"], ["inner:go"])

    def test_invoke_with_a_command_needs_a_paused_run(self):
        runner = log_runner(_subgraph_node_graph(), "nest-inv-bad")
        with self.assertRaises(ValueError):
            runner.invoke(Command(resume="go"), poll_interval=0.01, timeout=1)
        with self.assertRaises(ValueError):
            runner.invoke(
                Command(goto="sub"), workflow_id="wf-x", poll_interval=0.01, timeout=1
            )
