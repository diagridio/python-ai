# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""The stream of a resume starts at the run's last chunk, not at ``v1.start``.

It carries what happens after the resume only, so its ``workflow_started``
(and a failure before its first chunk) carries the cursor of what the log held
when it started: a reconnect with that id gets the rest of the run, not the
part streamed before the interrupt again.
"""

from typing import Any, List, Tuple
from unittest import mock

from fastapi.testclient import TestClient

from diagrid.agent.core.workflow.stream_cursor import START_ID

from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_interrupt import INPUT, _approval_graph, _wait
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase

Frames = List[Tuple[Any, str, Any]]


async def _broken_drain(*args: Any, **kwargs: Any) -> Any:
    raise ConnectionError("sidecar gone")
    yield  # an async generator


class TestResumeStreamIds(_RunnerTestCase):
    def _interrupted(self, name: str) -> Tuple[Any, TestClient, Frames]:
        runner = log_runner(_approval_graph(), name)
        http = TestClient(serve_app(runner))
        got = frames(http.post("/agent/run", json={**INPUT, "stream": True}).text)
        self.assertEqual(got[-1][1], "workflow_interrupted")
        return runner, http, got

    def _reconnect(self, http: TestClient, wf: str, event_id: str) -> Frames:
        return frames(
            http.get(
                f"/agent/run/{wf}/stream", headers={"Last-Event-ID": event_id}
            ).text
        )

    def test_a_drop_right_after_workflow_started_resumes_the_rest(self):
        runner, http, got = self._interrupted("ids-started")
        wf = got[-1][2]["workflow_id"]
        rest = frames(
            http.post(
                f"/agent/run/{wf}/resume", json={"resume": "yes", "stream": True}
            ).text
        )
        started_id = rest[0][0]
        self.assertEqual(rest[0][1], "workflow_started")
        self.assertNotEqual(started_id, START_ID)
        _wait(lambda: wf in runner._workflow_client.done)
        again = self._reconnect(http, wf, started_id)
        # exactly what the resume's stream carried: nothing from before it.
        # The ids may differ: a live reader may not yet know a unit whose last
        # event it sent is done, a replay does, and both cursors resume alike.
        self.assertEqual(
            [(e, d) for _, e, d in again[1:]], [(e, d) for _, e, d in rest[1:]]
        )
        pairs = zip(rest[1:-1], again[1:-1])
        for i, ((live_id, event, _), (replay_id, _, _)) in enumerate(pairs, start=1):
            if event == "node_reset":
                continue  # a control event repeats the id of the chunk before it
            want = [(e, d) for _, e, d in rest[i + 1 :]]
            for cursor in (live_id, replay_id):
                with self.subTest(event=i, cursor=cursor):
                    got = self._reconnect(http, wf, cursor)
                    # the rest of the stream after event i: no repeat, no gap
                    self.assertEqual([(e, d) for _, e, d in got[1:]], want)

    def test_a_failure_before_the_first_chunk_carries_the_same_id(self):
        runner, http, got = self._interrupted("ids-broken")
        wf = got[-1][2]["workflow_id"]
        with mock.patch("diagrid.agent.langgraph.resume_run.drain", _broken_drain):
            broken = frames(
                http.post(
                    f"/agent/run/{wf}/resume", json={"resume": "yes", "stream": True}
                ).text
            )
        self.assertEqual(
            [e for _, e, _ in broken], ["workflow_started", "workflow_failed"]
        )
        self.assertTrue(broken[-1][2]["workflow_still_running"])
        self.assertNotEqual(broken[-1][0], START_ID)
        self.assertEqual(broken[-1][0], broken[0][0])
        _wait(lambda: wf in runner._workflow_client.done)
        after = self._reconnect(http, wf, broken[-1][0])
        # the same events as a client that kept the last chunk before the
        # interrupt gets (the ids differ: each repeats the cursor it was sent)
        kept = self._reconnect(http, wf, got[-2][0])
        self.assertEqual(
            [(e, d) for _, e, d in after[1:]], [(e, d) for _, e, d in kept[1:]]
        )
        self.assertEqual(after[-1][1], "workflow_completed")
