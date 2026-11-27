# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""What the public ``run_async`` yields does not change because streaming exists."""

from typing import Any

from dapr.ext.workflow import WorkflowStatus

from diagrid.agent.langgraph.models import ChannelState, GraphWorkflowOutput
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY

from .streaming_helpers import INPUT, _collect, _PollCase, _state


def _done() -> Any:
    out = GraphWorkflowOutput(
        output={"a": 1},
        channel_state=ChannelState(),
        steps=2,
        status="completed",
        executed=[["chat"], ["tool"]],
    )
    return _state(WorkflowStatus.COMPLETED, out.to_dict())


class TestRunAsyncShape(_PollCase):
    async def test_a_direct_caller_gets_no_executed_key(self):
        runner, _ = self._runner([_done()], "shape-plain")
        events = await _collect(runner.run_async(INPUT, poll_interval=0))
        self.assertEqual(
            events[-1],
            {
                "type": "workflow_completed",
                "workflow_id": events[-1]["workflow_id"],
                "output": {"a": 1},
                "steps": 2,
                "status": "completed",
            },
        )

    async def test_the_stream_asks_for_it(self):
        runner, _ = self._runner([_done()], "shape-stream")
        events = await _collect(
            runner.run_async(INPUT, config={STREAM_RUN_KEY: True}, poll_interval=0)
        )
        self.assertEqual(events[-1]["executed"], [["chat"], ["tool"]])
