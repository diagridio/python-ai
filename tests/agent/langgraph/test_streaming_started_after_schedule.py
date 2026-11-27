# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""``workflow_started`` reaches the client only once the workflow exists.

A client (or a second replica) that looks the run up the moment it sees
``workflow_started`` must find it: the event is yielded after
``schedule_new_workflow`` has returned, never before.
"""

import asyncio
import threading
import unittest
from typing import Any, List
from unittest import mock

from diagrid.agent.core.workflow.stream import WORKFLOW_STARTED, StreamEvent

from .golden_harness import _FakeWorkflowClient
from .streaming_helpers import INPUT, _chat_graph
from .test_runner import _RunnerTestCase


class _GatedClient(_FakeWorkflowClient):
    """A workflow client whose scheduling waits until the test lets it return."""

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self.release = threading.Event()
        self.returned = threading.Event()

    def schedule_new_workflow(self, *, workflow: Any, input: Any, instance_id: str):
        self.release.wait(5)
        super().schedule_new_workflow(
            workflow=workflow, input=input, instance_id=instance_id
        )
        self.returned.set()


class TestStartedAfterSchedule(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        patch = mock.patch(
            "diagrid.agent.langgraph.workflow.when_all", side_effect=list
        )
        patch.start()
        self.addCleanup(patch.stop)

    def _runner(self, name: str) -> Any:
        runner = _chat_graph(name=name)
        runner._started = True
        client = _GatedClient(runner)
        runner._workflow_client = client
        return runner, client

    async def test_nothing_is_yielded_while_the_workflow_is_being_scheduled(self):
        runner, client = self._runner("started-gated")
        events = runner._stream_events(
            INPUT, config=None, thread_id="t", workflow_id="wf-gated", poll_interval=0
        )
        first = asyncio.ensure_future(events.__anext__())
        await asyncio.sleep(0.2)
        self.assertFalse(first.done(), "workflow_started came before scheduling")
        client.release.set()
        item = await first
        self.assertTrue(client.returned.is_set())
        self.assertIsInstance(item, StreamEvent)
        self.assertEqual(item.mode, WORKFLOW_STARTED)
        self.assertEqual(item.data["workflow_id"], "wf-gated")
        rest = [i async for i in events]
        self.assertEqual(rest[-1]["type"], "workflow_completed")

    async def test_on_control_hears_of_it_after_scheduling_returned(self):
        runner, client = self._runner("started-control")
        seen: List[bool] = []
        client.release.set()
        async for _ in runner.astream(
            INPUT,
            workflow_id="wf-control",
            stream_mode="updates",
            poll_interval=0,
            on_control=lambda c: (
                seen.append(client.returned.is_set())
                if c.kind == WORKFLOW_STARTED
                else None
            ),
        ):
            pass
        self.assertEqual(seen, [True])

    async def test_a_run_that_never_scheduled_has_no_started_event(self):
        runner, client = self._runner("started-never")

        def fail(**kw: Any) -> None:
            raise RuntimeError("no sidecar")

        client.schedule_new_workflow = fail  # type: ignore[method-assign]
        items = [
            i
            async for i in runner._stream_events(
                INPUT, config=None, thread_id="t", workflow_id="wf-x", poll_interval=0
            )
        ]
        modes = [i.mode for i in items if isinstance(i, StreamEvent)]
        self.assertNotIn(WORKFLOW_STARTED, modes)
        self.assertEqual(items[-1]["type"], "workflow_failed")
