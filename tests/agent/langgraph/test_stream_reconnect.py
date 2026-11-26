# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Reconnecting to a run's stream: ``GET /agent/run/{id}/stream`` and ``astream_resume``."""

import json
import time
import unittest
from typing import Any, List, Tuple

from fastapi.testclient import TestClient

from diagrid.agent.core.workflow.stream_cursor import (
    END_ID,
    START_ID,
    StreamCursor,
    encode_cursor,
)

from .golden_harness import MODES, _golden_graph, _wire
from .log_helpers import LogClient, log_runner
from .streaming_helpers import INPUT, _chat_graph, _collect, serve_app
from .test_runner import _clear_dapr_registration, _RunnerTestCase


def _frames(text: str) -> List[Tuple[str, str, Any]]:
    """``(id, event, data)`` of each SSE frame (comments left out)."""
    out = []
    for frame in text.strip().split("\n\n"):
        if frame.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in frame.split("\n"))
        out.append((fields.get("id"), fields["event"], json.loads(fields["data"])))
    return out


def _wait_done(runner: Any, workflow_id: str) -> None:
    client = runner._workflow_client
    deadline = time.monotonic() + 10
    while workflow_id not in client.done and time.monotonic() < deadline:
        time.sleep(0.01)


class _Case(_RunnerTestCase):
    def _post(self, runner: Any) -> List[Tuple[str, str, Any]]:
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"messages": INPUT["messages"], "stream": True}
        )
        self.assertEqual(resp.status_code, 200)
        return _frames(resp.text)


class TestReconnectEndpoint(_Case):
    def test_resuming_after_any_event_gives_exactly_the_rest(self):
        runner = log_runner(_golden_graph(), "re-mid")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        http = TestClient(serve_app(runner))
        for cut in (1, len(frames) // 2, len(frames) - 2):
            with self.subTest(cut=cut):
                resp = http.get(
                    f"/agent/run/{wf}/stream",
                    headers={"Last-Event-ID": frames[cut][0]},
                )
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(
                    resp.headers["content-type"].startswith("text/event-stream")
                )
                got = _frames(resp.text)
                self.assertEqual(got[0][1], "workflow_started")
                self.assertEqual(got[0][0], START_ID)
                # no duplicates, no gaps: the events after the cut, same ids
                self.assertEqual([(i, e, d) for i, e, d in got[1:]], frames[cut + 1 :])

    def test_the_query_parameter_works_like_the_header(self):
        runner = log_runner(_golden_graph(), "re-query")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        resp = TestClient(serve_app(runner)).get(
            f"/agent/run/{wf}/stream", params={"last_event_id": frames[3][0]}
        )
        self.assertEqual(_frames(resp.text)[1:], frames[4:])

    def test_no_id_replays_the_whole_run_with_its_result(self):
        runner = log_runner(_golden_graph(), "re-all")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        got = _frames(TestClient(serve_app(runner)).get(f"/agent/run/{wf}/stream").text)
        self.assertEqual(got[1:], frames[1:])
        self.assertEqual(got[-1][1], "workflow_completed")
        self.assertIn("output", got[-1][2])

    def test_the_end_id_gives_only_the_final_event(self):
        runner = log_runner(_golden_graph(), "re-end")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        got = _frames(
            TestClient(serve_app(runner))
            .get(f"/agent/run/{wf}/stream", headers={"Last-Event-ID": END_ID})
            .text
        )
        self.assertEqual([e for _, e, _ in got], ["workflow_completed"])
        self.assertEqual(got[0][0], END_ID)

    def test_a_junk_id_replays_from_the_start(self):
        runner = log_runner(_golden_graph(), "re-junk")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        got = _frames(
            TestClient(serve_app(runner))
            .get(f"/agent/run/{wf}/stream", headers={"Last-Event-ID": "nonsense"})
            .text
        )
        self.assertEqual(got[1:], frames[1:])

    def test_another_replica_serves_the_reconnect(self):
        runner = log_runner(_golden_graph(), "re-two")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        store = runner.stream_transport.store
        client = runner._workflow_client
        _clear_dapr_registration()
        other = log_runner(_golden_graph(), "re-two", store=store)
        other._workflow_client = client
        got = _frames(
            TestClient(serve_app(other))
            .get(f"/agent/run/{wf}/stream", headers={"Last-Event-ID": frames[2][0]})
            .text
        )
        self.assertEqual(got[1:], frames[3:])

    def test_an_unknown_run_is_404(self):
        runner = log_runner(_golden_graph(), "re-404")
        resp = TestClient(serve_app(runner)).get("/agent/run/nope/stream")
        self.assertEqual(resp.status_code, 404)

    def test_an_unsupported_mode_is_422_before_the_stream(self):
        runner = log_runner(_golden_graph(), "re-422")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        resp = TestClient(serve_app(runner)).get(
            f"/agent/run/{wf}/stream", params={"stream_mode": "checkpoints"}
        )
        self.assertEqual(resp.status_code, 422)

    def test_stream_mode_picks_what_is_replayed(self):
        runner = log_runner(_golden_graph(), "re-mode")
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        got = _frames(
            TestClient(serve_app(runner))
            .get(f"/agent/run/{wf}/stream", params={"stream_mode": "updates"})
            .text
        )
        self.assertEqual(
            {e for _, e, _ in got},
            {"workflow_started", "updates", "workflow_completed"},
        )


class TestReconnectInProcess(_Case):
    def test_says_reconnect_needs_a_state_store_and_still_gives_the_result(self):
        runner = _chat_graph(name="re-ip")
        runner._started = True
        runner._workflow_client = LogClient(runner)  # type: ignore[assignment]
        frames = self._post(runner)
        wf = frames[0][2]["workflow_id"]
        _wait_done(runner, wf)
        got = _frames(TestClient(serve_app(runner)).get(f"/agent/run/{wf}/stream").text)
        names = [e for _, e, _ in got]
        self.assertEqual(
            names, ["workflow_started", "stream_unavailable", "workflow_completed"]
        )
        self.assertEqual(got[1][2]["reason"], "transport_not_replayable")
        self.assertIn("state store", got[1][2]["message"])
        self.assertIn("output", got[2][2])


class TestAstreamResume(_RunnerTestCase, unittest.IsolatedAsyncioTestCase):
    async def test_replays_a_finished_run_like_astream(self):
        runner = log_runner(_golden_graph(), "ar-all")
        first = await _collect(
            runner.astream(
                INPUT, workflow_id="wf-ar", stream_mode=MODES, poll_interval=0.01
            )
        )
        again = await _collect(
            runner.astream_resume("wf-ar", stream_mode=MODES, poll_interval=0.01)
        )
        self.assertEqual(_wire(again), _wire(first))

    async def test_an_attempt_the_client_saw_part_of_is_reset(self):
        runner = log_runner(_golden_graph(), "ar-reset")
        await _collect(
            runner.astream(
                INPUT, workflow_id="wf-arr", stream_mode=MODES, poll_interval=0.01
            )
        )
        # the client saw 2 events of an attempt of "chat" that the log no longer has
        cursor = encode_cursor(
            StreamCursor(base=0, positions=((0, "chat", "deadbeef", 2),))
        )
        controls: List[Any] = []
        await _collect(
            runner.astream_resume(
                "wf-arr",
                last_event_id=cursor,
                stream_mode=MODES,
                poll_interval=0.01,
                on_control=controls.append,
            )
        )
        resets = [c.data for c in controls if c.kind == "node_reset"]
        self.assertEqual(resets, [{"node": "chat", "step": 1}])

    async def test_an_unknown_run_raises(self):
        runner = log_runner(_golden_graph(), "ar-404")
        with self.assertRaises(LookupError):
            await _collect(runner.astream_resume("nope"))
