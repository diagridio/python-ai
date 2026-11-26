# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A live stream reads every unit of the run before it closes on the final event.

The reader learns a run's units from its custom status, which it reads only
now and then while a unit is being written. A run that pauses on
``interrupt()`` (or fails) right after the last read has a final event that
does not list its units, so the reader must read the status once more: the
``__interrupt__`` chunk comes before ``workflow_interrupted``.
"""

import asyncio
import unittest
from typing import Any, Dict, List

from langgraph.types import Interrupt

from ..core.log_store_fake import FakeLogStore
from .follow_helpers import FollowedStream, transport, write

DRAFT = [("custom", {"token": i}) for i in range(3)] + [
    ("updates", {"draft": {"draft": "text"}})
]
ASKED = {"__interrupt__": (Interrupt(value={"question": "ship?"}, id="i-1"),)}
INTERRUPTED = {
    "type": "workflow_interrupted",
    "workflow_id": "wf",
    "status": "interrupted",
    "interrupt_seq": 1,
    "interrupts": [{"value": {"question": "ship?"}, "id": "i-1"}],
}
FAILED = {
    "type": "workflow_failed",
    "workflow_id": "wf",
    "error": {"message": "boom", "error_type": "WorkflowError"},
}


class TestFinalAfterTheLastStatusRead(unittest.IsolatedAsyncioTestCase):
    async def _stream_ending_with(self, final: Dict[str, Any]) -> List[Any]:
        log = transport(FakeLogStore())
        write(log, 0, "draft", DRAFT).finish(done=True)
        status: Dict[str, Any] = {"executed": [["draft"]]}
        # a long poll: the reader reads the status once, then only on the final
        live = FollowedStream(log, lambda: status, poll_s=30)
        before = await live.take(len(DRAFT))
        # the next step starts and pauses before the reader looks again
        status = {"executed": [["draft"], ["ask"]]}
        write(log, 1, "ask", [("updates", ASKED)]).finish(done=True)
        live.follower.finish(final)
        rest = await asyncio.wait_for(live.rest(), 10)
        await live.close()
        return before + rest

    async def test_an_interrupt_streams_its_chunk_before_the_final_event(self):
        frames = await self._stream_ending_with(INTERRUPTED)
        names = [name for name, *_ in frames]
        self.assertEqual(names[-1], "workflow_interrupted")
        self.assertEqual(names[-2], "updates")
        self.assertIn("__interrupt__", frames[-2][1])
        self.assertEqual(
            frames[-2][1]["__interrupt__"][0]["value"],
            INTERRUPTED["interrupts"][0]["value"],
        )

    async def test_a_failed_run_streams_what_its_last_step_wrote(self):
        frames = await self._stream_ending_with(FAILED)
        self.assertEqual([n for n, *_ in frames][-2:], ["updates", "workflow_failed"])
        self.assertIn("__interrupt__", frames[-2][1])


class TestFinalBehindTheWindow(unittest.IsolatedAsyncioTestCase):
    async def test_a_reader_behind_the_window_does_not_learn_later_steps_first(self):
        # the status lists only steps 5 on: steps 0..4 are in no status and no
        # final event, so step 5 now would leave a gap a cursor then skips
        log = transport(FakeLogStore())
        write(log, 5, "z", [("updates", {"z": {"n": 5}})]).finish(done=True)
        status = {"executed_from": 5, "executed": [["z"]]}
        live = FollowedStream(log, lambda: status, poll_s=30)
        live.follower.finish(INTERRUPTED)
        frames = await asyncio.wait_for(live.rest(), 10)
        await live.close()
        self.assertEqual([n for n, *_ in frames], ["workflow_interrupted"])
