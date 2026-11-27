# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The thread's state is read and written by activities, never by workflow code.

Workflow code runs again on every replay; an activity's result comes from the
history. An instance started before thread continuation replays without the
two activities.
"""

import json
import threading
from typing import Any

from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.models import GraphWorkflowOutput
from diagrid.agent.langgraph.state import DaprMemoryCheckpointer
from diagrid.agent.langgraph.workflow import agent_workflow

from .golden_harness import _FakeDaprTest
from .test_runner import _InlineWorkflowContext
from .thread_helpers import MemoryStateStore, chat_graph, shape, turn

_in_activity = threading.local()


class _GuardedCtx(_InlineWorkflowContext):
    """Marks when an activity runs, so the store can tell who calls it."""

    instance_id = "wf-replay"

    def call_activity(self, activity: Any, *, input: Any, retry_policy: Any = None):
        _in_activity.on = True
        try:
            return super().call_activity(activity, input=input)
        finally:
            _in_activity.on = False


def _only_in_activities() -> None:
    if not getattr(_in_activity, "on", False):
        raise AssertionError("the thread store was touched from workflow code")


class TestThreadStateInActivities(_FakeDaprTest):
    def _setup(self) -> Any:
        store = MemoryStateStore()
        store.guard = _only_in_activities
        runner = DaprWorkflowGraphRunner(
            graph=chat_graph(),
            name="thread-replay",
            checkpointer=DaprMemoryCheckpointer(state_store=store),  # type: ignore[arg-type]
        )
        runner._started = True
        return runner, store

    def _drive(self, runner: Any, text: str, patched: bool = True) -> Any:
        ctx = _GuardedCtx(runner, patched=patched)
        wf_input = runner._prepare_run(turn(text), "t1", None, "wf-replay")
        workflow = agent_workflow(ctx, json.loads(json.dumps(wf_input.workflow_input)))
        result = None
        try:
            while True:
                result = workflow.send(result)
        except StopIteration as stop:
            return ctx, GraphWorkflowOutput.from_dict(stop.value)

    def test_load_comes_first_and_save_last_both_as_activities(self):
        runner, store = self._setup()
        self._drive(runner, "hi")
        ctx, out = self._drive(runner, "again")
        self.assertEqual(ctx.scheduled[0], "load_thread_state")
        self.assertEqual(ctx.scheduled[-1], "save_thread_state")
        self.assertEqual(len(shape(out.output["messages"])), 4)

    def test_an_instance_from_before_the_patch_replays_without_them(self):
        runner, store = self._setup()
        ctx, out = self._drive(runner, "hi", patched=False)
        self.assertNotIn("load_thread_state", ctx.scheduled)
        self.assertNotIn("save_thread_state", ctx.scheduled)
        self.assertEqual(store.data, {})
