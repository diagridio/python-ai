# Copyright (c) 2026-Present Diagrid Inc.
# SPDX-License-Identifier: BUSL-1.1

"""Which runs carry a thread's state, and the ids a run gets when it has none.

A run continues a thread only when the caller named one (``thread_id``,
``session_id`` or ``configurable["thread_id"]``): a run without one is
stateless and touches no thread state. Generated ids are full UUIDs. Thread
continuation is on by default and can be turned off on the runner.
"""

import logging
import re
from typing import Any
from unittest import mock

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from diagrid.agent.core.discovery import DiscoveredComponents
from diagrid.agent.langgraph import DaprWorkflowGraphRunner
from diagrid.agent.langgraph.state import DaprMemoryCheckpointer
from diagrid.agent.langgraph.thread_state import DaprThreadStore

from .golden_harness import _FakeDaprTest, _FakeWorkflowClient
from .log_helpers import log_runner
from .streaming_helpers import serve_app
from .test_reconnect_helpers import frames
from .thread_helpers import MemoryStateStore, chat_graph, shape, turn

FULL_UUID = re.compile(
    r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$"
)


def _runner(name: str, **kwargs: Any) -> DaprWorkflowGraphRunner:
    store = MemoryStateStore()
    runner = DaprWorkflowGraphRunner(
        graph=chat_graph(),
        name=name,
        checkpointer=DaprMemoryCheckpointer(state_store=store),  # type: ignore[arg-type]
        **kwargs,
    )
    runner._started = True
    runner._workflow_client = _FakeWorkflowClient(runner)  # type: ignore[assignment]
    return runner


def _thread_state(runner: Any, thread_id: Any = None, config: Any = None) -> bool:
    setup = runner._prepare_run(turn("hi"), thread_id, config, None)
    return bool(setup.workflow_input.get("thread_state"))


class TestWhichRunsCarryState(_FakeDaprTest):
    def test_a_run_without_a_thread_id_is_stateless(self):
        self.assertFalse(_thread_state(_runner("scope-none")))

    def test_a_named_thread_carries_state(self):
        runner = _runner("scope-named")
        self.assertTrue(_thread_state(runner, "t1"))
        self.assertTrue(
            _thread_state(runner, config={"configurable": {"thread_id": "t1"}})
        )

    def test_continuation_can_be_turned_off(self):
        runner = _runner("scope-off", continue_threads=False)
        self.assertFalse(_thread_state(runner, "t1"))

    def test_stateless_runs_leave_nothing_in_the_store(self):
        runner = _runner("scope-store")
        store = runner._thread_store._checkpointer._store  # type: ignore[union-attr]
        for _ in range(3):
            runner.invoke(turn("hi"), poll_interval=0.01, timeout=10)
        self.assertEqual(store.data, {})


class TestGeneratedIds(_FakeDaprTest):
    def test_a_generated_thread_and_workflow_id_are_full_uuids(self):
        runner = _runner("scope-ids")
        setup = runner._prepare_run(turn("hi"), None, None, None)
        self.assertRegex(setup.thread_id, FULL_UUID)
        suffix = setup.workflow_id.rsplit("-", 1)[1]
        self.assertRegex(suffix, r"^[0-9a-f]{32}$")

    def test_an_http_run_without_ids_is_stateless_with_a_full_uuid(self):
        runner = log_runner(chat_graph(), "scope-http")
        runner._thread_store = DaprThreadStore(  # a runner with a store
            DaprMemoryCheckpointer(state_store=MemoryStateStore())  # type: ignore[arg-type]
        )
        http = TestClient(serve_app(runner))
        got = frames(http.post("/agent/run", json={**turn("hi"), "stream": True}).text)
        started = got[0][2]
        self.assertRegex(started["thread_id"], FULL_UUID)
        inputs = runner._workflow_client.inputs[started["workflow_id"]]
        self.assertFalse(inputs.get("thread_state"))
        named = frames(
            http.post(
                "/agent/run", json={**turn("hi"), "stream": True, "session_id": "s1"}
            ).text
        )
        self.assertTrue(
            runner._workflow_client.inputs[named[0][2]["workflow_id"]]["thread_state"]
        )


class TestThreadIdsOverHttp(_FakeDaprTest):
    def test_a_thread_id_with_the_actor_separator_is_refused(self):
        runner = log_runner(chat_graph(), "scope-sep")
        http = TestClient(serve_app(runner))
        for key in ("thread_id", "session_id"):
            for stream in (True, False):
                with self.subTest(key=key, stream=stream):
                    response = http.post(
                        "/agent/run",
                        json={**turn("hi"), key: "a||b", "stream": stream},
                    )
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn("||", response.text)
        self.assertEqual(runner._workflow_client.inputs, {})


class TestResentHistory(_FakeDaprTest):
    def test_messages_resent_with_their_ids_are_not_duplicated(self):
        runner = _runner("scope-resend")
        first = runner.invoke(
            turn("hi"), thread_id="t1", poll_interval=0.01, timeout=10
        )
        history = [*first["messages"], {"role": "user", "content": "again"}]
        out = runner.invoke(
            {"messages": history}, thread_id="t1", poll_interval=0.01, timeout=10
        )
        self.assertEqual(
            shape(out["messages"]),
            [
                ("human", "hi"),
                ("ai", "Hello world"),
                ("human", "again"),
                ("ai", "Hello world"),
            ],
        )


class TestInMemoryCheckpointer(_FakeDaprTest):
    def test_a_graph_compiled_with_memory_saver_uses_the_apps_store(self):
        found = DiscoveredComponents(memory_store_name="agent-memory")
        with (
            mock.patch(
                "diagrid.agent.core.workflow.runner.discover_components",
                return_value=found,
            ),
            mock.patch("diagrid.agent.core.state.DaprStateStore", MemoryStateStore),
            self.assertLogs("diagrid.agent.langgraph", level=logging.WARNING) as logs,
        ):
            runner = DaprWorkflowGraphRunner(
                graph=chat_graph(MemorySaver()), name="scope-mem"
            )
        self.assertIsInstance(runner._thread_store, DaprThreadStore)
        self.assertTrue(any("MemorySaver" in line for line in logs.output))

    def test_memory_saver_without_a_store_still_works_with_a_warning(self):
        with self.assertLogs("diagrid.agent.langgraph", level=logging.WARNING) as logs:
            runner = DaprWorkflowGraphRunner(
                graph=chat_graph(MemorySaver()), name="scope-mem-only"
            )
        self.assertIsNotNone(runner._thread_store)
        self.assertTrue(any("replica" in line for line in logs.output))
