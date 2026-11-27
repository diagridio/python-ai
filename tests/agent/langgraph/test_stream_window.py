# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A reader that fell behind the status's window of steps waits for the final.

The custom status lists the run's last steps only. A reader that knows some
early steps but not the ones just below the window must not learn the later
steps first: a cursor it then hands out would skip the gap for good.
"""

import asyncio
import unittest
from typing import Any, Dict, Optional

from diagrid.agent.core.workflow.stream import InProcessSink
from diagrid.agent.core.workflow.stream_cursor import StreamCursor
from diagrid.agent.core.workflow.stream_log import (
    StateLogStreamConfig,
    StateLogTransport,
)
from diagrid.agent.langgraph.stream_follow import LogFollower, units_of

from tests.agent.core.log_store_fake import FakeLogStore


def _behind(
    known: Dict[str, Any], later: Dict[str, Any], cursor: Optional[StreamCursor] = None
) -> bool:
    async def go() -> bool:
        transport = StateLogTransport(StateLogStreamConfig(), store=FakeLogStore())
        sink = InProcessSink("wf", asyncio.get_running_loop(), bounded=False)
        follower = LogFollower(
            transport.reader("wf"),
            sink,
            workflow_id="wf",
            status=lambda: {},
            poll_s=0.01,
            cursor=cursor,
        )
        follower._learn(units_of(known))
        return follower._behind_window(later)

    return asyncio.run(go())


class TestBehindTheWindow(unittest.TestCase):
    def test_a_gap_between_known_steps_and_the_window_waits(self):
        known = {"executed": [["a"]]}  # step 0
        later = {"executed_from": 200, "executed": [["z"]]}  # 1..199 are gone
        self.assertTrue(_behind(known, later))

    def test_a_reader_that_knows_nothing_waits(self):
        self.assertTrue(_behind({}, {"executed_from": 5, "executed": [["z"]]}))

    def test_a_reader_that_knows_the_step_before_the_window_reads_on(self):
        known = {"executed_from": 70, "executed": [["a"]] * 130}  # steps 70..199
        later = {"executed_from": 200, "executed": [["z"]]}
        self.assertFalse(_behind(known, later))

    def test_a_reader_that_knows_steps_inside_the_window_reads_on(self):
        known = {"executed_from": 100, "executed": [["a"]] * 150}  # 100..249
        later = {"executed_from": 200, "executed": [["z"]] * 60}
        self.assertFalse(_behind(known, later))

    def test_a_cursor_past_the_gap_reads_on(self):
        later = {"executed_from": 200, "executed": [["z"]]}
        self.assertFalse(_behind({}, later, StreamCursor(base=200)))

    def test_a_status_with_every_step_never_waits(self):
        self.assertFalse(_behind({}, {"executed": [["a"], ["b"]]}))
