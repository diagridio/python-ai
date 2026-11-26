# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Two resumes of one interrupt: the run takes one, the other caller gets 409.

Both callers see the run waiting and both raise the ``resume`` event; the
workflow takes the first and ignores the other. The caller whose value was
ignored must be told, not handed a result its value had no part in.
"""

import threading
import time
from typing import Any, List

from fastapi.testclient import TestClient

from .log_helpers import LogClient, log_runner
from .streaming_helpers import serve_app
from .test_interrupt import INPUT, _approval_graph
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase


class _GatedClient(LogClient):
    """Holds the workflow back from taking a resume until ``release`` is set."""

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self.release = threading.Event()

    def wait_event(self, instance_id: str, name: str) -> Any:
        self.release.wait(timeout=30)
        return super().wait_event(instance_id, name)

    def buffered(self) -> int:
        with self._cond:
            return sum(len(v) for v in self._events.values())


def _start(name: str) -> Any:
    runner = log_runner(_approval_graph(), name)
    client = _GatedClient(runner)
    runner._workflow_client = client  # type: ignore[assignment]
    http = TestClient(serve_app(runner))
    got = frames(http.post("/agent/run", json={**INPUT, "stream": True}).text)
    assert got[-1][1] == "workflow_interrupted", got[-1]
    return client, http, got[-1][2]["workflow_id"]


def _race(client: Any, posts: List[Any]) -> List[Any]:
    results: List[Any] = [None] * len(posts)

    def run(i: int) -> None:
        results[i] = posts[i]()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(posts))]
    for t in threads:
        t.start()
    deadline = time.monotonic() + 10
    while client.buffered() < len(posts) and time.monotonic() < deadline:
        time.sleep(0.01)
    client.release.set()
    for t in threads:
        t.join(timeout=30)
    return results


class TestConcurrentResumes(_RunnerTestCase):
    def test_the_second_resume_of_one_interrupt_gets_409(self):
        client, http, wf = _start("race-plain")
        url = f"/agent/run/{wf}/resume"
        a, b = _race(
            client,
            [
                lambda: http.post(url, json={"resume": "A"}),
                lambda: http.post(url, json={"resume": "B"}),
            ],
        )
        codes = sorted([a.status_code, b.status_code])
        self.assertEqual(codes, [200, 409])
        won = a if a.status_code == 200 else b
        value = "A" if won is a else "B"
        self.assertEqual(won.json()["output"]["answer"], value)

    def test_a_streamed_resume_that_lost_gets_409_before_streaming(self):
        client, http, wf = _start("race-stream")
        url = f"/agent/run/{wf}/resume"
        a, b = _race(
            client,
            [
                lambda: http.post(url, json={"resume": "A", "stream": True}),
                lambda: http.post(url, json={"resume": "B", "stream": True}),
            ],
        )
        self.assertEqual(sorted([a.status_code, b.status_code]), [200, 409])
        won = a if a.status_code == 200 else b
        self.assertEqual(frames(won.text)[-1][1], "workflow_completed")
