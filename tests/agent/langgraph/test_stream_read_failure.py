# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A stream that cannot read the log mid-run ends with an id the client resumes from.

When the serving process's sidecar goes away (the pod is deleted), the reader
fails and the stream ends with ``workflow_failed`` and
``workflow_still_running``. The run goes on elsewhere, so that event must not
carry ``v1.end`` (the client has the end): it repeats the last chunk's id, and
a reconnect with it gets exactly the rest of the run.
"""

import asyncio
import unittest
from typing import Any, Dict

from diagrid.agent.core.workflow.stream import StreamEvent
from diagrid.agent.core.workflow.stream_cursor import END_ID, START_ID, parse_cursor
from diagrid.agent.langgraph.stream_drain import SseEncoder, encode_final
from diagrid.agent.langgraph.stream_events import stream_failure_event

from ..core.log_store_fake import FakeLogStore
from .follow_helpers import FollowedStream, chunk, transport, write

A = [("custom", {"a": i}) for i in range(3)] + [("updates", {"a": {"x": 1}})]
B_FIRST = [("custom", {"b": i}) for i in range(2)]
B_REST = [("custom", {"b": i}) for i in range(2, 5)] + [("updates", {"b": {"y": 2}})]
STATUS = {"executed": [["a"], ["b"]]}
COMPLETED = {"type": "workflow_completed", "workflow_id": "wf", **STATUS}


def _data(frames: Any) -> list:
    return [data for _, data, *_ in frames]


class TestReadFailureMidRun(unittest.IsolatedAsyncioTestCase):
    async def test_the_failure_carries_the_last_chunk_id_and_resumes_exactly(self):
        store = FakeLogStore()
        log = transport(store)
        write(log, 0, "a", A).finish(done=True)
        b = write(log, 1, "b", B_FIRST)  # still running: two chunks are in the log

        status: Dict[str, Any] = dict(STATUS)
        live = FollowedStream(log, lambda: status)
        before = await live.take(len(A) + len(B_FIRST))
        self.assertEqual(_data(before), [d for _, d in A + B_FIRST])
        store.fail_gets = 10**6  # the sidecar is gone
        rest = await asyncio.wait_for(live.rest(), 10)
        await live.close()

        name, final, event_id = rest[-1]
        self.assertEqual(name, "workflow_failed")
        self.assertEqual(final["error"]["error_type"], "StreamReadError")
        self.assertTrue(final["workflow_still_running"])
        self.assertNotEqual(event_id, END_ID)
        self.assertEqual(event_id, before[-1][2])

        # the run goes on elsewhere and ends
        store.fail_gets = 0
        for seq, (mode, data) in enumerate(B_REST, start=len(B_FIRST)):
            b.put(chunk(1, "b", mode, data, seq))
        b.finish(done=True)

        resumed = FollowedStream(log, lambda: status, cursor=parse_cursor(event_id))
        resumed.follower.finish(COMPLETED)
        after = await asyncio.wait_for(resumed.rest(), 10)
        await resumed.close()
        self.assertEqual(after[-1][0], "workflow_completed")
        self.assertEqual(_data(after[:-1]), [d for _, d in B_REST])

        whole = FollowedStream(log, lambda: status, cursor=parse_cursor(START_ID))
        whole.follower.finish(COMPLETED)
        replay = await asyncio.wait_for(whole.rest(), 10)
        await whole.close()
        # no gaps, no duplicates
        self.assertEqual(_data(before) + _data(after[:-1]), _data(replay[:-1]))


def _chunk_event(event_id: str) -> StreamEvent:
    return StreamEvent(
        workflow_id="wf",
        step=0,
        node="a",
        attempt_id="",
        seq=0,
        ns=(),
        mode="custom",
        data={},
        event_id=event_id,
    )


class TestFinalIds(unittest.TestCase):
    def _failure(self) -> Dict[str, Any]:
        return stream_failure_event(ConnectionError("gone"), "StreamError", True)

    def test_a_finished_run_ends_on_the_end_id(self):
        encoder = SseEncoder(["custom"], True)
        encoder.encode(_chunk_event("v1.0.x"))
        self.assertEqual(encoder.encode(COMPLETED)[2], END_ID)

    def test_a_failure_while_the_run_goes_on_repeats_the_last_chunk_id(self):
        encoder = SseEncoder(["custom"], True)
        encoder.encode(_chunk_event("v1.0.x"))
        self.assertEqual(encoder.encode(self._failure())[2], "v1.0.x")

    def test_before_any_chunk_it_is_where_the_client_resumed_from(self):
        cursor = "v1.3.W10"
        self.assertEqual(
            SseEncoder(["custom"], True, resume_from=cursor).encode(self._failure())[2],
            cursor,
        )
        self.assertEqual(
            SseEncoder(["custom"], True).encode(self._failure())[2], START_ID
        )

    def test_a_failure_raised_out_of_the_stream_gets_the_same_id(self):
        encoder = SseEncoder(["custom"], True)
        encoder.encode(_chunk_event("v1.0.x"))
        self.assertEqual(
            encode_final(encoder, self._failure()),
            ("workflow_failed", self._failure(), "v1.0.x"),
        )
        # no encoder yet (the request was refused before the stream): no id
        self.assertEqual(
            encode_final(None, self._failure()), ("workflow_failed", self._failure())
        )

    def test_the_in_process_transport_numbers_events_itself(self):
        self.assertEqual(len(SseEncoder(["custom"], False).encode(self._failure())), 2)
