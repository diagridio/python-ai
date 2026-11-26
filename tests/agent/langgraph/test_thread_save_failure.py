# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A run whose thread state cannot be saved still returns its output.

The nodes all ran: failing the run would lose their work. The result says the
thread was not saved and why, and a stream says so before its final event.
"""

from typing import Any

from fastapi.testclient import TestClient

from diagrid.agent.core.workflow.stream_log import StateLogTransport
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.state import DaprMemoryCheckpointer

from ..core.log_store_fake import FakeLogStore
from .log_helpers import FAST, LogClient
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase
from .thread_helpers import MemoryStateStore, chat_graph, turn


class _SaveFails(MemoryStateStore):
    def save(self, key: str, value: Any, metadata: Any = None) -> None:
        raise ConnectionError("state store unreachable")

    def save_with_etag(self, *args: Any, **kwargs: Any) -> bool:
        raise ConnectionError("state store unreachable")


def _runner(name: str) -> DaprWorkflowGraphRunner:
    runner = DaprWorkflowGraphRunner(
        graph=chat_graph(),
        name=name,
        checkpointer=DaprMemoryCheckpointer(state_store=_SaveFails()),  # type: ignore[arg-type]
        stream_transport=StateLogTransport(FAST, store=FakeLogStore()),
    )
    runner._started = True
    runner._workflow_client = LogClient(runner)  # type: ignore[assignment]
    return runner


class TestThreadSaveFailure(_RunnerTestCase):
    def test_a_plain_run_returns_its_output_and_says_the_thread_was_not_saved(self):
        http = TestClient(serve_app(_runner("save-fail-plain")))
        body = http.post("/agent/run", json={**turn("hi"), "thread_id": "t1"}).json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output"]["messages"][-1]["content"], "Hello world")
        self.assertIs(body["thread_saved"], False)
        self.assertIn("state store unreachable", body["thread_error"])

    def test_a_stream_warns_before_its_final_event(self):
        http = TestClient(serve_app(_runner("save-fail-stream")))
        got = frames(
            http.post(
                "/agent/run", json={**turn("hi"), "thread_id": "t1", "stream": True}
            ).text
        )
        names = [e for _, e, _ in got]
        self.assertEqual(names[-2:], ["thread_not_saved", "workflow_completed"])
        self.assertIn("state store unreachable", got[-2][2]["error"])
        self.assertEqual(got[-2][2]["thread_id"], "t1")
        self.assertIs(got[-1][2]["thread_saved"], False)

    def test_the_status_endpoint_reports_it_too(self):
        runner = _runner("save-fail-status")
        http = TestClient(serve_app(runner))
        body = http.post("/agent/run", json={**turn("hi"), "thread_id": "t1"}).json()
        status = http.get(f"/agent/run/{body['instance_id']}").json()
        self.assertIs(status["thread_saved"], False)
        self.assertIn("state store unreachable", status["thread_error"])
