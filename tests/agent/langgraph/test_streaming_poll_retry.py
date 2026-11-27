# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A failed status poll is not a failed workflow.

Poll retries are for transient sidecar errors, opt-in, bounded, and count
consecutive failures only; and no blocking sidecar call runs on the event loop
that delivers tokens.
"""

import ssl
import threading
import unittest
from typing import Any, Dict, List
from unittest import mock

import grpc
from dapr.ext.workflow import WorkflowStatus
from fastapi.testclient import TestClient

from diagrid.agent.core.workflow import runner as core_runner
from diagrid.agent.langgraph.streaming import GraphRunError

from .streaming_helpers import (
    INPUT,
    _collect,
    _completed_state,
    _PollCase,
    _state,
    _Unavailable,
    parse_sse,
    serve_app,
)

MAX = core_runner.POLL_MAX_ERRORS


class _Rpc(grpc.RpcError):
    """A gRPC error that carries no status, as a bare ``RpcError`` does."""


def _rpc_error(code: grpc.StatusCode) -> grpc.RpcError:
    class _Coded(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return code

    return _Coded()


# What the poll loop retries (True) and what it lets through at once (False).
CLASSIFICATION = {
    **{
        f"grpc-{code.name}": (_rpc_error(code), True)
        for code in (
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.CANCELLED,
        )
    },
    **{
        f"grpc-{code.name}": (_rpc_error(code), False)
        for code in (
            grpc.StatusCode.PERMISSION_DENIED,
            grpc.StatusCode.UNAUTHENTICATED,
            grpc.StatusCode.UNIMPLEMENTED,
            grpc.StatusCode.INVALID_ARGUMENT,
            grpc.StatusCode.NOT_FOUND,
        )
    },
    "grpc-without-a-status": (_Rpc(), False),
    "connection-reset": (ConnectionResetError("x"), True),
    "connection-refused": (ConnectionRefusedError("x"), True),
    "timeout": (TimeoutError("x"), True),
    "permission": (PermissionError("x"), False),
    "bad-socket-path": (FileNotFoundError("x"), False),
    "ssl": (ssl.SSLError("x"), False),
    "plain-os-error": (OSError("x"), False),
    "bug": (TypeError("x"), False),
}


class TestWhichPollErrorsAreRetried(_PollCase):
    def test_classification(self):
        for name, (error, transient) in CLASSIFICATION.items():
            with self.subTest(error=name):
                self.assertIs(core_runner._is_transient_poll_error(error), transient)

    async def test_a_transient_error_is_ridden_out_and_any_other_surfaces_at_once(self):
        for name, (error, transient) in CLASSIFICATION.items():
            with self.subTest(error=name):
                self.setUp()
                runner, client = self._runner([error, _completed_state()], name)
                if transient:
                    events = await _collect(runner.run_async(INPUT, poll_interval=0))
                    self.assertEqual(events[-1]["type"], "workflow_completed")
                    self.assertEqual(client.get_workflow_state.call_count, 2)
                else:
                    with self.assertRaises(type(error)):
                        await _collect(runner.run_async(INPUT, poll_interval=0))
                    self.assertEqual(client.get_workflow_state.call_count, 1)


class TestRetryIsOptIn(_PollCase):
    async def test_the_shared_poll_loop_does_not_retry_by_default(self):
        runner, client = self._runner(
            [_Unavailable("blip"), _completed_state()], "scope-default"
        )
        with self.assertRaises(_Unavailable):
            async for _ in runner._poll_workflow("wf", "s", poll_interval=0):
                pass
        self.assertEqual(client.get_workflow_state.call_count, 1)

    async def test_it_retries_when_asked(self):
        runner, client = self._runner(
            [_Unavailable("blip"), _completed_state()], "scope-asked"
        )
        events = [
            e
            async for e in runner._poll_workflow(
                "wf", "s", poll_interval=0, retry_failed_polls=True
            )
        ]
        self.assertEqual(events[-1]["type"], "workflow_completed")


class TestConsecutiveFailuresOnly(_PollCase):
    async def test_scattered_errors_never_add_up(self):
        running = _state(WorkflowStatus.RUNNING)
        polls: List[Any] = [
            *[_Unavailable("a")] * (MAX - 1),
            running,
            *[_Unavailable("b")] * (MAX - 1),
            _completed_state(),
        ]
        runner, client = self._runner(polls, "scope-scattered")
        events = await _collect(runner.run_async(INPUT, poll_interval=0))
        self.assertEqual(events[-1]["type"], "workflow_completed")
        self.assertEqual(client.get_workflow_state.call_count, len(polls))


class TestRetryBound(_PollCase):
    async def test_a_permanently_failing_poll_is_tried_exactly_max_plus_one_times(self):
        runner, client = self._runner([_Unavailable("down")] * (MAX + 5), "scope-bound")
        with self.assertRaises(_Unavailable):
            async for _ in runner._poll_workflow(
                "wf", "s", poll_interval=0, retry_failed_polls=True
            ):
                pass
        # the first try and MAX retries, then the error is let through
        self.assertEqual(client.get_workflow_state.call_count, MAX + 1)

    async def test_the_bound_is_per_stretch_of_failures(self):
        running = _state(WorkflowStatus.RUNNING)
        polls: List[Any] = [
            *[_Unavailable("a")] * MAX,
            running,
            *[_Unavailable("b")] * (MAX + 1),
            _completed_state(),
        ]
        runner, client = self._runner(polls, "scope-bound-twice")
        with self.assertRaises(_Unavailable):
            async for _ in runner._poll_workflow(
                "wf", "s", poll_interval=0, retry_failed_polls=True
            ):
                pass
        self.assertEqual(client.get_workflow_state.call_count, 2 * MAX + 2)


class TestRetryDelay(unittest.IsolatedAsyncioTestCase):
    async def test_the_delay_doubles_and_is_capped(self):
        sleeps: List[float] = []

        async def record(delay: float, *args: Any) -> None:
            sleeps.append(delay)

        client = mock.Mock()
        client.get_workflow_state.side_effect = [_Unavailable("x")] * (MAX - 1) + [
            _completed_state()
        ]
        runner = mock.Mock(spec=["_workflow_client"])
        runner._workflow_client = client
        with mock.patch.object(core_runner.asyncio, "sleep", record):
            async for _ in core_runner.BaseWorkflowRunner._poll_workflow(
                runner, "wf", "s", poll_interval=0, retry_failed_polls=True
            ):
                pass
        delays = [d for d in sleeps if d]
        self.assertEqual(
            delays[:6], [0.5, 1, 2, 4, 8, core_runner.POLL_RETRY_MAX_SECONDS]
        )
        self.assertLessEqual(max(delays), core_runner.POLL_RETRY_MAX_SECONDS)
        self.assertEqual(max(delays), core_runner.POLL_RETRY_MAX_SECONDS)


class TestTransientPollErrors(_PollCase):
    async def test_astream_rides_out_a_dropped_poll(self):
        runner, client = self._runner(
            [
                _state(WorkflowStatus.RUNNING),
                _Unavailable("sidecar restarting"),
                _state(WorkflowStatus.RUNNING),
                _completed_state(),
            ],
            "poll-ok",
        )
        items = await _collect(
            runner.astream(INPUT, stream_mode=["updates"], poll_interval=0)
        )
        self.assertEqual(items[-1][0], "stream_unavailable")  # no sink in this test
        self.assertEqual(client.get_workflow_state.call_count, 4)


class TestPersistentPollErrors(_PollCase):
    def _failing(self, name: str):
        polls = [_Unavailable("sidecar gone")] * (core_runner.POLL_MAX_ERRORS + 1)
        return self._runner(polls, name)

    async def test_astream_says_the_run_may_still_be_going(self):
        runner, _ = self._failing("poll-dead")
        with self.assertRaises(GraphRunError) as caught:
            await _collect(runner.astream(INPUT, stream_mode=["updates"]))
        final = caught.exception.final
        self.assertEqual(final["error"]["error_type"], "PollError")
        self.assertTrue(final["workflow_still_running"])
        self.assertTrue(final["workflow_id"])

    async def test_it_is_not_published_as_a_workflow_failure(self):
        runner, _ = self._failing("poll-pub")
        published: List[Any] = []
        fake = mock.Mock()
        fake.publish.side_effect = lambda topic, data: published.append(data)
        with mock.patch("diagrid.agent.core.pubsub.DaprPubSub", return_value=fake):
            app = serve_app(runner, pubsub_name="ps", publish_topic="out")
        resp = TestClient(app).post("/agent/run", json={"task": "hi", "stream": True})
        name, data = parse_sse(resp.text)[-1]
        self.assertEqual(name, "workflow_failed")
        self.assertTrue(data["workflow_still_running"])
        self.assertEqual(published, [])


class TestSidecarCallsLeaveTheLoop(_PollCase):
    async def test_polling_and_scheduling_run_in_worker_threads(self):
        runner, client = self._runner(
            [_state(WorkflowStatus.RUNNING), _completed_state()], "offloop"
        )
        loop_thread = threading.get_ident()
        threads: Dict[str, List[int]] = {"poll": [], "schedule": []}
        polls = client.get_workflow_state.side_effect

        def poll(**kw: Any) -> Any:
            threads["poll"].append(threading.get_ident())
            return next(polls_iter)

        polls_iter = iter(polls)
        client.get_workflow_state.side_effect = poll
        client.schedule_new_workflow.side_effect = lambda **kw: threads[
            "schedule"
        ].append(threading.get_ident())

        events = await _collect(runner.run_async(INPUT, poll_interval=0))
        self.assertEqual(events[-1]["type"], "workflow_completed")
        self.assertTrue(threads["poll"] and threads["schedule"])
        for name, idents in threads.items():
            self.assertNotIn(loop_thread, idents, f"{name} ran on the event loop")
