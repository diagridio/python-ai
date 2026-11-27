# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""A streamed run records only the modes it was started with.

``messages``, ``updates`` and ``custom`` chunks are always written. A
``values`` chunk (the whole state) and the ``tasks``/``debug`` events (a task's
start carries the node's whole input) are written only when the run asked for
them; a reconnect or a streamed resume that asks for one the run did not
record is refused, naming what the run recorded.
"""

import asyncio
import json
import operator
import unittest
from typing import Any, List, Set

from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from diagrid.agent.core.workflow.stream import StreamSinkRegistry
from diagrid.agent.langgraph.models import ChannelState, ExecuteNodeInput
from diagrid.agent.langgraph.stream_modes import (
    STREAM_MODES_KEY,
    UnrecordedStreamMode,
    check_recorded,
    recorded_modes,
    with_stream_modes,
)
from diagrid.agent.langgraph.streaming import STREAM_RUN_KEY
from diagrid.agent.langgraph.workflow import (
    clear_registries,
    execute_node_activity,
    register_node,
)

from ..core.log_store_fake import FakeLogStore
from .golden_harness import MODES, _both, _FakeDaprTest, _golden_graph, _wire
from .log_helpers import log_runner
from .streaming_helpers import INPUT, _collect, _Ctx, _drain, serve_app
from .test_interrupt import INPUT as ASK_INPUT
from .test_interrupt import _approval_graph
from .test_reconnect_helpers import frames
from .test_runner import _RunnerTestCase
from .test_streaming_golden_tasks import _stable

ALL_MODES = ["messages", "updates", "custom", "values", "tasks", "debug"]
STATE_MODES = {"values", "tasks", "debug"}


def _logged_modes(store: FakeLogStore) -> Set[str]:
    """The modes of every event the run's log holds (batches are event lists)."""
    return {
        wire["mode"]
        for value in store.data.values()
        if isinstance(value, list)
        for wire in value
        if isinstance(wire, dict) and "mode" in wire
    }


class TestRecordedModes(unittest.TestCase):
    def test_a_config_without_modes_records_every_mode(self):
        self.assertIsNone(recorded_modes({STREAM_RUN_KEY: True}))
        self.assertIsNone(recorded_modes(None))

    def test_the_always_recorded_modes_are_added(self):
        config = with_stream_modes({}, ["values"])
        self.assertEqual(
            with_stream_modes({}, ["values", "updates"])[STREAM_MODES_KEY],
            ["updates", "values"],
        )
        self.assertEqual(
            recorded_modes(config), {"messages", "updates", "custom", "values"}
        )


class TestTheLogHoldsOnlyRequestedModes(
    _RunnerTestCase, unittest.IsolatedAsyncioTestCase
):
    async def _run(self, name: str, modes: List[str]) -> FakeLogStore:
        store = FakeLogStore()
        runner = log_runner(_golden_graph(), name, store=store)
        await _collect(
            runner.astream(
                INPUT, workflow_id=f"wf-{name}", stream_mode=modes, poll_interval=0.01
            )
        )
        return store

    async def test_default_modes_write_no_values_or_task_events(self):
        logged = _logged_modes(await self._run("lean-default", MODES))
        self.assertTrue(logged)
        self.assertFalse(logged & STATE_MODES, logged)

    async def test_values_alone_writes_no_task_events(self):
        logged = _logged_modes(await self._run("lean-values", ["values"]))
        self.assertIn("values", logged)
        self.assertFalse(logged & {"tasks", "debug"}, logged)

    async def test_tasks_alone_writes_task_and_debug_events_but_no_values(self):
        logged = _logged_modes(await self._run("lean-tasks", ["tasks"]))
        self.assertTrue({"tasks", "debug"} <= logged, logged)
        self.assertNotIn("values", logged)

    async def test_every_mode_is_written_when_requested(self):
        logged = _logged_modes(await self._run("lean-all", ALL_MODES))
        self.assertTrue(STATE_MODES <= logged, logged)

    async def test_a_replay_of_every_mode_is_what_the_live_stream_gave(self):
        runner = log_runner(_golden_graph(), "lean-replay")
        live = await _collect(
            runner.astream(
                INPUT, workflow_id="wf-lr", stream_mode=ALL_MODES, poll_interval=0.01
            )
        )
        again = await _collect(
            runner.astream_resume("wf-lr", stream_mode=ALL_MODES, poll_interval=0.01)
        )
        self.assertEqual(_wire(again), _wire(live))


class TestEveryModeMatchesNative(_FakeDaprTest):
    async def test_all_six_modes_requested_match_native(self):
        native, ours = await _both(_golden_graph(), "lean-golden", INPUT, ALL_MODES)
        self.assertTrue({m for m, _ in native} >= STATE_MODES)
        self._assert_same_sequence(_stable(native), _stable(ours))


class TestUnrecordedModeOnReconnect(_RunnerTestCase):
    def _default_run(self, name: str) -> Any:
        runner = log_runner(_golden_graph(), name)
        http = TestClient(serve_app(runner))
        got = frames(
            http.post(
                "/agent/run", json={"messages": INPUT["messages"], "stream": True}
            ).text
        )
        return runner, http, got[0][2]["workflow_id"]

    def test_values_on_a_run_without_it_is_422_naming_the_recorded_modes(self):
        _, http, wf = self._default_run("lean-422")
        resp = http.get(f"/agent/run/{wf}/stream", params={"stream_mode": "values"})
        self.assertEqual(resp.status_code, 422)
        detail = resp.json()["detail"]
        self.assertIn("['values']", detail)
        self.assertIn("['messages', 'updates', 'custom']", detail)
        self.assertIn(f"GET /agent/run/{wf}", detail)

    def test_task_modes_are_refused_too(self):
        _, http, wf = self._default_run("lean-422-tasks")
        for mode in ("tasks", "debug"):
            with self.subTest(mode=mode):
                resp = http.get(
                    f"/agent/run/{wf}/stream",
                    params={"stream_mode": ["updates", mode]},
                )
                self.assertEqual(resp.status_code, 422)

    def test_a_recorded_mode_is_still_served(self):
        _, http, wf = self._default_run("lean-ok")
        resp = http.get(f"/agent/run/{wf}/stream", params={"stream_mode": "updates"})
        self.assertEqual(resp.status_code, 200)


class TestUnrecordedModeOnAstreamResume(
    _RunnerTestCase, unittest.IsolatedAsyncioTestCase
):
    async def test_values_on_a_run_without_it_raises_value_error(self):
        runner = log_runner(_golden_graph(), "lean-ar")
        await _collect(
            runner.astream(
                INPUT, workflow_id="wf-lean-ar", stream_mode=MODES, poll_interval=0.01
            )
        )
        with self.assertRaises(ValueError) as caught:
            await _collect(
                runner.astream_resume(
                    "wf-lean-ar", stream_mode="values", poll_interval=0.01
                )
            )
        self.assertIsInstance(caught.exception, UnrecordedStreamMode)
        self.assertIn("'messages', 'updates', 'custom'", str(caught.exception))

    async def test_a_run_that_asked_for_values_replays_them(self):
        runner = log_runner(_golden_graph(), "lean-ar-ok")
        await _collect(
            runner.astream(
                INPUT,
                workflow_id="wf-lean-ok",
                stream_mode=["updates", "values"],
                poll_interval=0.01,
            )
        )
        got = await _collect(
            runner.astream_resume(
                "wf-lean-ok", stream_mode=["values"], poll_interval=0.01
            )
        )
        self.assertTrue([p for m, p in _wire(got) if m == "values"])


class TestUnrecordedModeOnResume(_RunnerTestCase):
    def test_a_streamed_resume_keeps_the_runs_modes(self):
        runner = log_runner(_approval_graph(), "lean-resume")
        http = TestClient(serve_app(runner))
        got = frames(http.post("/agent/run", json={**ASK_INPUT, "stream": True}).text)
        self.assertEqual(got[-1][1], "workflow_interrupted")
        wf = got[-1][2]["workflow_id"]
        refused = http.post(
            f"/agent/run/{wf}/resume",
            json={"resume": "yes", "stream": True, "stream_mode": ["values"]},
        )
        self.assertEqual(refused.status_code, 422)
        self.assertIn("'messages', 'updates', 'custom'", refused.json()["detail"])
        # the refusal raised nothing: the run still waits, and a resume with
        # its own modes goes through
        ok = frames(
            http.post(
                f"/agent/run/{wf}/resume",
                json={"resume": "yes", "stream": True, "stream_mode": ["updates"]},
            ).text
        )
        self.assertEqual(ok[-1][1], "workflow_completed")


class TestInProcessActivity(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        clear_registries()
        self.addCleanup(clear_registries)

    async def _modes_sent(self, config: dict) -> Set[str]:
        register_node("b", lambda state: {"log": ["b"]})
        registry = StreamSinkRegistry()
        sink = registry.open("wf-ip")
        payload = ExecuteNodeInput(
            node_name="b",
            channel_state=ChannelState(
                values={"log": ["a"]}, versions={"log": 2}, updated_channels=["log"]
            ),
            config=config,
            step=1,
        ).to_dict()
        await asyncio.to_thread(
            execute_node_activity, _Ctx("wf-ip"), payload, stream_registry=registry
        )
        sink.close()
        return {e.mode for e in await _drain(sink) if hasattr(e, "mode")}

    async def test_no_task_events_are_built_for_a_run_without_them(self):
        modes = await self._modes_sent(with_stream_modes({STREAM_RUN_KEY: True}, MODES))
        self.assertIn("updates", modes)
        self.assertFalse(modes & {"tasks", "debug"}, modes)

    async def test_a_run_without_recorded_modes_sends_task_events(self):
        modes = await self._modes_sent({STREAM_RUN_KEY: True})
        self.assertTrue({"tasks", "debug"} <= modes, modes)


class TestCheckRecorded(unittest.TestCase):
    def test_a_run_from_before_modes_were_recorded_allows_every_mode(self):
        from types import SimpleNamespace

        state = SimpleNamespace(
            serialized_input='{"config": {"diagrid_streamed": true}}'
        )
        self.assertEqual(check_recorded(["values"], state, "wf"), ["values"])


class TestUnrecordedModeOnAstreamCommand(
    _RunnerTestCase, unittest.IsolatedAsyncioTestCase
):
    async def test_a_resume_asking_for_values_raises_and_the_run_still_waits(self):
        from langgraph.types import Command

        runner = log_runner(_approval_graph(), "lean-cmd")
        await _collect(
            runner.astream(
                ASK_INPUT,
                workflow_id="wf-cmd",
                stream_mode="updates",
                poll_interval=0.01,
            )
        )
        with self.assertRaises(ValueError):
            await _collect(
                runner.astream(
                    Command(resume="go"),
                    workflow_id="wf-cmd",
                    stream_mode=["values"],
                    poll_interval=0.01,
                )
            )
        rest = await _collect(
            runner.astream(
                Command(resume="go"),
                workflow_id="wf-cmd",
                stream_mode="updates",
                poll_interval=0.01,
            )
        )
        self.assertEqual(rest[-1], {"after": {"log": ["after:go"]}})


class _Failing(TypedDict, total=False):
    log: Annotated[list, operator.add]


def _failing_graph(first_ok: bool = True) -> Any:
    """[ok ->] boom, where boom raises before it streams anything."""

    def boom(state: _Failing) -> dict:
        raise RuntimeError("boom")

    graph = StateGraph(_Failing)
    graph.add_node("boom", boom)
    if first_ok:
        graph.add_node("ok", lambda state: {"log": ["ok"]})
        graph.add_edge(START, "ok")
        graph.add_edge("ok", "boom")
    else:
        graph.add_edge(START, "boom")
    graph.add_edge("boom", END)
    return graph.compile()


class TestFailedNodeReplay(_RunnerTestCase):
    """A node that failed before its first chunk is not an expired log."""

    def _failed_run(self, name: str, first_ok: bool = True) -> Any:
        runner = log_runner(_failing_graph(first_ok), name)
        http = TestClient(serve_app(runner))
        live = frames(
            http.post(
                "/agent/run",
                json={"log": [], "stream": True, "stream_mode": ["updates"]},
            ).text
        )
        self.assertEqual(live[-1][1], "workflow_failed")
        return http, live

    def test_a_replay_from_the_start_sends_no_stream_purged(self):
        # only the failing node: no unit of the run logged anything
        http, live = self._failed_run("lean-fail-start", first_ok=False)
        wf = live[0][2]["workflow_id"]
        again = frames(
            http.get(f"/agent/run/{wf}/stream", params={"stream_mode": "updates"}).text
        )
        self.assertNotIn("stream_purged", [e for _, e, _ in again])
        self.assertEqual(
            [(e, d) for _, e, d in again[1:-1]], [(e, d) for _, e, d in live[1:-1]]
        )
        self.assertEqual(again[-1][1], "workflow_failed")

    def test_a_reconnect_at_the_failed_step_sends_no_stream_purged(self):
        http, live = self._failed_run("lean-fail-base")
        wf = live[0][2]["workflow_id"]
        # the id after ok's update: the cursor's base is the failed step
        after_ok = [i for i, e, d in live if e == "updates"][-1]
        again = frames(
            http.get(
                f"/agent/run/{wf}/stream",
                params={"stream_mode": "updates"},
                headers={"Last-Event-ID": after_ok},
            ).text
        )
        self.assertEqual(
            [e for _, e, _ in again], ["workflow_started", "workflow_failed"]
        )

    def test_the_failed_attempt_writes_a_head(self):
        store = FakeLogStore()
        runner = log_runner(_failing_graph(), "lean-fail-head", store=store)
        frames(
            TestClient(serve_app(runner))
            .post("/agent/run", json={"log": [], "stream": True})
            .text
        )
        heads = [k for k in store.data if ":boom:" in k and k.endswith(":head")]
        self.assertTrue(heads, sorted(store.data))


class TestTaskModesGoTogether(unittest.TestCase):
    def test_either_task_mode_records_both(self):
        for mode in ("tasks", "debug"):
            with self.subTest(mode=mode):
                recorded = recorded_modes(with_stream_modes({}, ["updates", mode]))
                self.assertTrue({"tasks", "debug"} <= set(recorded or ()))

    def test_the_refusal_of_a_task_mode_does_not_point_at_the_state(self):
        from types import SimpleNamespace

        state = SimpleNamespace(
            serialized_input=json.dumps(
                {"config": with_stream_modes({STREAM_RUN_KEY: True}, ["updates"])}
            )
        )
        with self.assertRaises(UnrecordedStreamMode) as caught:
            check_recorded(["tasks"], state, "wf")
        self.assertIn("started with tasks or debug", str(caught.exception))
        self.assertNotIn("final state", str(caught.exception))
        with self.assertRaises(UnrecordedStreamMode) as caught:
            check_recorded(["values"], state, "wf")
        self.assertIn("final state", str(caught.exception))
