# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Which transport a runner streams through, and what the state-store log delivers."""

import json
import unittest
from typing import Any, List
from unittest import mock

from fastapi.testclient import TestClient

from diagrid.agent.core.discovery import DiscoveredComponents
from diagrid.agent.core.workflow import stream as stream_module
from diagrid.agent.core.workflow.stream import (
    InProcessTransport,
    StreamSinkRegistry,
)
from diagrid.agent.core.workflow.stream_cursor import (
    END_ID,
    START_ID,
    parse_cursor,
)
from diagrid.agent.core.workflow.stream_log import (
    StateLogStreamConfig,
    StateLogTransport,
)
from diagrid.agent.langgraph import DaprWorkflowGraphRunner

from ..core.log_store_fake import FakeLogStore
from .golden_harness import MODES, _both, _golden_graph, _Golden, _wire
from .log_helpers import FAST, log_runner, replica
from .streaming_helpers import INPUT, _collect, serve_app
from .test_runner import _RunnerTestCase


def _discovered(**kw: Any) -> Any:
    return mock.patch(
        "diagrid.agent.core.workflow.runner.discover_components",
        return_value=DiscoveredComponents(**kw),
    )


class TestTransportChoice(_RunnerTestCase):
    def test_no_state_store_streams_in_process(self):
        runner = DaprWorkflowGraphRunner(graph=_golden_graph(), name="t-none")
        self.assertIsInstance(runner.stream_transport, InProcessTransport)
        self.assertFalse(runner.stream_transport.replayable)

    def test_the_app_memory_store_turns_the_event_log_on(self):
        with _discovered(memory_store_name="agent-memory"):
            runner = DaprWorkflowGraphRunner(graph=_golden_graph(), name="t-mem")
        transport = runner.stream_transport
        self.assertIsInstance(transport, StateLogTransport)
        self.assertTrue(transport.replayable)
        self.assertEqual(transport.store_name, "agent-memory")

    def test_the_workflow_state_store_is_enough(self):
        # every workflow app has one: no component, variable or argument to add
        with _discovered(workflow_store_name="agent-workflow"):
            runner = DaprWorkflowGraphRunner(graph=_golden_graph(), name="t-wf")
        self.assertIsInstance(runner.stream_transport, StateLogTransport)
        self.assertEqual(runner.stream_transport.store_name, "agent-workflow")

    def test_a_given_transport_wins(self):
        mine = StateLogTransport(StateLogStreamConfig(store_name="mine"))
        with _discovered(memory_store_name="agent-memory"):
            runner = DaprWorkflowGraphRunner(
                graph=_golden_graph(), name="t-own", stream_transport=mine
            )
        self.assertEqual(runner.stream_transport.store_name, "mine")

    def test_a_given_registry_keeps_the_in_process_transport(self):
        registry = StreamSinkRegistry()
        with _discovered(memory_store_name="agent-memory"):
            runner = DaprWorkflowGraphRunner(
                graph=_golden_graph(), name="t-reg", stream_registry=registry
            )
        self.assertIsInstance(runner.stream_transport, InProcessTransport)
        self.assertIs(runner.stream_registry, registry)


class TestStateLogStream(_Golden):
    async def _astream(self, runner: Any, wf: str, **kw: Any) -> List[Any]:
        return await _collect(
            runner.astream(
                INPUT,
                thread_id="t1",
                config={"tags": ["x"], "metadata": {"foo": "bar"}},
                workflow_id=wf,
                stream_mode=kw.pop("modes", MODES),
                poll_interval=0.01,
                **kw,
            )
        )

    async def test_the_same_chunks_as_the_in_process_transport(self):
        _, in_process = await _both(_golden_graph(), "golden-ip", INPUT)
        runner = log_runner(_golden_graph(), "golden-log")
        ours = await self._astream(runner, "wf-golden-log")
        self.assertEqual(_wire(ours), _wire(in_process))

    async def test_chunks_of_nodes_on_another_replica_arrive(self):
        store = FakeLogStore()
        other = replica(_golden_graph(), "two", store)
        runner = log_runner(_golden_graph(), "two", store=store, activity_runner=other)
        controls: List[Any] = []
        ours = await self._astream(runner, "wf-two", on_control=controls.append)
        self.assertEqual({m for m, _ in ours}, {"messages", "updates", "custom"})
        updates = [p for m, p in ours if m == "updates"]
        self.assertEqual([list(u) for u in updates], [["chat"], ["counter"], ["quiet"]])
        self.assertNotIn("stream_unavailable", [c.kind for c in controls])

    async def test_a_slow_consumer_loses_nothing(self):
        runner = log_runner(_golden_graph(), "slow")
        with mock.patch.object(stream_module, "MAX_BUFFERED_EVENTS", 2):
            controls: List[Any] = []
            ours = await self._astream(runner, "wf-slow", on_control=controls.append)
        self.assertNotIn("stream_dropped", [c.kind for c in controls])
        self.assertEqual(len([m for m, _ in ours if m == "messages"]), 5)

    async def test_writes_that_fail_are_reported_as_write_failed(self):
        store = FakeLogStore()
        store.fail_saves = 10_000
        runner = log_runner(_golden_graph(), "wfail", store=store)
        controls: List[Any] = []
        await self._astream(runner, "wf-wfail", on_control=controls.append)
        unavailable = [c.data for c in controls if c.kind == "stream_unavailable"]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0]["reason"], "write_failed")
        self.assertEqual(
            [m["node"] for m in unavailable[0]["missing_nodes"]],
            ["chat", "counter", "quiet"],
        )

    async def test_a_store_that_cannot_be_read_is_an_error_not_an_empty_stream(self):
        store = FakeLogStore()
        runner = log_runner(_golden_graph(), "rfail", store=store)
        store.fail_gets = 10_000
        with self.assertRaises(Exception) as caught:
            await self._astream(runner, "wf-rfail")
        self.assertIn("StreamReadError", repr(getattr(caught.exception, "final", "")))

    async def test_a_run_nobody_streams_writes_no_log(self):
        store = FakeLogStore()
        runner = log_runner(_golden_graph(), "plain", store=store)
        result = [
            e
            async for e in runner.run_async(
                INPUT, workflow_id="wf-plain", poll_interval=0.01
            )
        ]
        self.assertEqual(result[-1]["type"], "workflow_completed")
        self.assertEqual(store.data, {})


class TestStateLogSse(_RunnerTestCase):
    def test_every_event_has_an_id_any_replica_can_resume_from(self):
        runner = log_runner(_golden_graph(), "sse-ids")
        resp = TestClient(serve_app(runner)).post(
            "/agent/run", json={"messages": INPUT["messages"], "stream": True}
        )
        frames = [f for f in resp.text.strip().split("\n\n") if not f.startswith(":")]
        ids = [f.split("\n")[0] for f in frames]
        self.assertTrue(all(i.startswith("id: ") for i in ids), ids)
        ids = [i[4:] for i in ids]
        self.assertEqual(ids[0], START_ID)
        self.assertEqual(ids[-1], END_ID)
        for event_id in ids[1:-1]:
            self.assertNotEqual(
                parse_cursor(event_id), parse_cursor(START_ID), event_id
            )
        names = [f.split("\n")[1] for f in frames]
        self.assertEqual(names[-1], "event: workflow_completed")
        self.assertEqual(
            json.loads(frames[-1].split("data: ", 1)[1])["status"], "completed"
        )


if __name__ == "__main__":
    unittest.main()
